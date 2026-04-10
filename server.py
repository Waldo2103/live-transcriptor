#!/usr/bin/env python3
"""
Live Transcriptor
-----------------
  /   → UI principal (WebSocket en tiempo real)

Variables de entorno:
  WHISPER_MODEL    medium (default) | tiny | base | small | large-v3
  WHISPER_DEVICE   cpu (default) | cuda
  OLLAMA_HOST      http://host.docker.internal:11434
  OLLAMA_MODELO    llama3
  LLM_PROVIDER     ollama (default) | claude
  ANTHROPIC_API_KEY  (si LLM_PROVIDER=claude)
"""
import asyncio
import os
import threading
import queue
import time
import tempfile
from datetime import datetime
from typing import Optional

import httpx
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn

import transcriber as tr
from sources import YouTubeSource, MicrophoneSource
from llm import get_provider
from fact_checker import FactChecker

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL",  "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST",    "http://host.docker.internal:11434")
OLLAMA_MODELO  = os.getenv("OLLAMA_MODELO",  "llama3")
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",  "ollama")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY",  "")

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

# ── Estado global ─────────────────────────────────────────────────────────────
_ws_clients: set[WebSocket] = set()
_event_loop: Optional[asyncio.AbstractEventLoop] = None

session: dict = {
    "active":          False,
    "source_type":     None,    # "youtube" | "microphone"
    "url":             None,
    "started_at":      None,
    "segments":        [],
    "fact_check_on":   False,
    "show_timestamps": True,
    "language":        None,
    "error":           None,
}
_active_source = None


# ── WebSocket broadcast ───────────────────────────────────────────────────────

async def _broadcast(msg: dict):
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def broadcast(msg: dict):
    """Enviar desde cualquier thread."""
    if _event_loop:
        asyncio.run_coroutine_threadsafe(_broadcast(msg), _event_loop)


# ── Helpers ───────────────────────────────────────────────────────────────────

def secs_to_ts(s: float) -> str:
    s = int(s)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Worker de transcripción ───────────────────────────────────────────────────

def _transcription_worker(source, fact_checker: FactChecker,
                           language: Optional[str], initial_prompt: str):
    whisper = tr.get_transcriber(WHISPER_MODEL, WHISPER_DEVICE)
    offset  = 0.0

    broadcast({"type": "status", "status": "running",
               "msg": "Modelo cargado. Escuchando..."})

    try:
        while session["active"]:
            try:
                chunk = source.get_chunk(timeout=5)
            except queue.Empty:
                continue

            if chunk is None:
                if source.error:
                    session["error"] = source.error
                    broadcast({"type": "error",
                               "msg": f"Error en la fuente de audio: {source.error}"})
                break

            result = whisper.transcribe(
                chunk,
                offset   = offset,
                language = language,
                initial_prompt = initial_prompt or None,
            )

            if not result.segments:
                offset += 18   # avanzar aunque no haya habla
                continue

            for seg in result.segments:
                fc = fact_checker.check(seg.text)
                data = {
                    "type":       "segment",
                    "text":       seg.text,
                    "ts":         secs_to_ts(seg.start),
                    "start":      seg.start,
                    "end":        seg.end,
                    "language":   result.language,
                    "fact_check": fc,
                }
                session["segments"].append(data)
                broadcast(data)

            offset = result.segments[-1].end

            # Si la fuente reportó un error, propagarlo
            if source.error:
                raise RuntimeError(source.error)

    except Exception as e:
        session["error"] = str(e)
        broadcast({"type": "error", "msg": str(e)})
    finally:
        session["active"] = False
        broadcast({"type": "status", "status": "stopped",
                   "msg": "Sesión finalizada."})


# ── Groq Whisper API ──────────────────────────────────────────────────────────

async def groq_transcribe(
    audio_bytes: bytes,
    filename: str,
    language: Optional[str],
    prompt: Optional[str],
    api_key: str,
) -> str:
    """Envía audio a Groq Whisper large-v3-turbo y devuelve el texto."""
    data   = {"model": "whisper-large-v3-turbo", "response_format": "text"}
    if language: data["language"] = language
    if prompt:   data["prompt"]   = prompt
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (filename, audio_bytes, "application/octet-stream")},
            data=data,
        )
        r.raise_for_status()
        return r.text.strip()


# ── API ───────────────────────────────────────────────────────────────────────

@app.post("/api/start")
def start_session(
    source:          str = Form(...),         # "youtube" | "microphone"
    url:             str = Form(""),
    fact_check:      str = Form("no"),
    show_timestamps: str = Form("si"),
    language:        str = Form(""),          # "" = auto-detect
    initial_prompt:  str = Form(""),
    llm_provider:    str = Form(LLM_PROVIDER),
    llm_model:       str = Form(OLLAMA_MODELO),
):
    global _active_source

    if session["active"]:
        raise HTTPException(409, "Ya hay una sesión activa. Detené la actual primero.")
    if source == "youtube" and not url.strip():
        raise HTTPException(400, "Se requiere URL para YouTube.")

    provider = get_provider(llm_provider, host=OLLAMA_HOST, model=llm_model)
    fc        = FactChecker(provider, enabled=(fact_check == "si"))

    session.update({
        "active":          True,
        "source_type":     source,
        "url":             url.strip() or None,
        "started_at":      datetime.now().isoformat(),
        "segments":        [],
        "fact_check_on":   fact_check == "si",
        "show_timestamps": show_timestamps == "si",
        "language":        language.strip() or None,
        "error":           None,
    })

    if source == "youtube":
        _active_source = YouTubeSource(url.strip())
    else:
        _active_source = MicrophoneSource()

    _active_source.start()

    threading.Thread(
        target=_transcription_worker,
        args=(_active_source, fc, language.strip() or None, initial_prompt.strip()),
        daemon=True,
    ).start()

    broadcast({"type": "status", "status": "starting",
               "msg": "Iniciando fuente de audio..."})

    return {"ok": True}


@app.post("/api/stop")
def stop_session():
    global _active_source
    session["active"] = False
    if _active_source:
        _active_source.stop()
        _active_source = None
    return {"ok": True}


@app.post("/api/toggle-factcheck")
def toggle_factcheck(enabled: str = Form(...)):
    # Nota: cambia el flag en la sesión, el worker lo relee en cada segmento
    session["fact_check_on"] = (enabled == "si")
    # También actualizamos el FactChecker activo si pudiéramos — por ahora
    # el worker usa el objeto creado al inicio. Para cambio en caliente
    # habría que pasar el fc por referencia (TODO v2).
    broadcast({"type": "config", "fact_check_on": session["fact_check_on"]})
    return {"ok": True, "fact_check_on": session["fact_check_on"]}


@app.get("/api/status")
def get_status():
    return {
        "active":          session["active"],
        "source_type":     session["source_type"],
        "fact_check_on":   session["fact_check_on"],
        "show_timestamps": session["show_timestamps"],
        "segment_count":   len(session["segments"]),
        "started_at":      session["started_at"],
        "error":           session["error"],
    }


@app.get("/api/transcript")
def get_transcript(timestamps: str = "si"):
    lines = []
    for seg in session["segments"]:
        prefix = f"[{seg['ts']}] " if timestamps == "si" else ""
        lines.append(prefix + seg["text"])
    content = "\n".join(lines)
    return PlainTextResponse(
        content,
        headers={"Content-Disposition": 'attachment; filename="transcripcion.txt"'},
    )


@app.post("/api/speech-segment")
async def speech_segment(text: str = Form(...)):
    """Recibe segmentos del Web Speech API del browser y los almacena/difunde."""
    if not text.strip():
        return {"ok": True}
    now = datetime.now()
    started = session.get("started_at")
    if started:
        elapsed = (now - datetime.fromisoformat(started)).total_seconds()
    else:
        elapsed = 0.0
        session["started_at"] = now.isoformat()
        session["active"] = True
    seg_data = {
        "type":       "segment",
        "text":       text.strip(),
        "ts":         secs_to_ts(elapsed),
        "start":      elapsed,
        "end":        elapsed,
        "language":   "auto",
        "fact_check": None,
    }
    session["segments"].append(seg_data)
    broadcast(seg_data)
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """WebSocket de control: recibe JSON de control, envía segmentos transcritos."""
    await ws.accept()
    _ws_clients.add(ws)
    await ws.send_json({
        "type":    "init",
        "session": {
            "active":        session["active"],
            "source_type":   session["source_type"],
            "fact_check_on": session["fact_check_on"],
            "segments":      session["segments"],
        },
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)


@app.websocket("/ws-mic")
async def ws_mic_endpoint(ws: WebSocket):
    """
    WebSocket de audio. Motores:
      engine=groq  → audio raw a Groq API (~0.5s, Whisper large-v3-turbo)
      engine=local → ffmpeg → PCM → Whisper local (lento en CPU)
    """
    await ws.accept()

    language       = ws.query_params.get("language", "") or None
    initial_prompt = ws.query_params.get("prompt", "")   or None
    fact_check_on  = ws.query_params.get("fc", "no") == "si"
    audio_ext      = ws.query_params.get("ext", ".webm")
    engine         = ws.query_params.get("engine", "local")
    api_key        = ws.query_params.get("key", "") or GROQ_API_KEY

    whisper  = tr.get_transcriber(WHISPER_MODEL, WHISPER_DEVICE) if engine == "local" else None
    provider = get_provider(LLM_PROVIDER, host=OLLAMA_HOST, model=OLLAMA_MODELO)
    fc       = FactChecker(provider, enabled=fact_check_on)
    offset   = 0.0
    CHUNK_SECS = 6.0

    broadcast({"type": "status", "status": "running", "msg": "Micrófono conectado. Escuchando..."})
    print(f"[ws-mic] engine={engine} ext={audio_ext}")

    chunk_n = 0
    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                print("[ws-mic] timeout: no llegaron chunks en 30s")
                break

            chunk_n += 1
            print(f"[ws-mic] chunk #{chunk_n}: {len(data)} bytes")

            if engine == "groq":
                # ── Groq: audio raw → API (sin ffmpeg) ────────────────────
                if not api_key:
                    await ws.send_json({"type": "error", "msg": "Groq: falta la API key."})
                    break
                try:
                    text = await groq_transcribe(
                        data, f"audio{audio_ext}", language, initial_prompt, api_key
                    )
                except httpx.HTTPStatusError as e:
                    await ws.send_json({"type": "error",
                        "msg": f"Groq HTTP {e.response.status_code}: {e.response.text[:200]}"})
                    continue
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Groq error: {e}"})
                    continue

                print(f"[ws-mic] Groq OK: {len(text)} chars")
                if text:
                    seg_data = {
                        "type":       "segment",
                        "text":       text,
                        "ts":         secs_to_ts(offset),
                        "start":      offset,
                        "end":        offset + CHUNK_SECS,
                        "language":   language or "es",
                        "fact_check": None,
                    }
                    session["segments"].append(seg_data)
                    broadcast(seg_data)   # llega una sola vez por /ws
                    offset += CHUNK_SECS

            else:
                # ── Whisper local: ffmpeg → PCM → transcribe ───────────────
                with tempfile.NamedTemporaryFile(suffix=audio_ext, delete=False) as f:
                    f.write(data)
                    tmp_in = f.name
                tmp_out = tmp_in + ".pcm"
                try:
                    proc = await asyncio.create_subprocess_exec(
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", tmp_in,
                        "-ar", "16000", "-ac", "1", "-f", "s16le", tmp_out,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, ffmpeg_err = await proc.communicate()
                    pcm_size = os.path.getsize(tmp_out) if os.path.exists(tmp_out) else 0
                    print(f"[ws-mic] ffmpeg rc={proc.returncode} pcm={pcm_size}b")
                    if pcm_size < 512:
                        continue
                    audio_arr = (
                        np.frombuffer(open(tmp_out, "rb").read(), dtype=np.int16)
                        .astype(np.float32) / 32_768.0
                    )
                finally:
                    os.unlink(tmp_in)
                    if os.path.exists(tmp_out):
                        os.unlink(tmp_out)

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda a=audio_arr, o=offset: whisper.transcribe(
                        a, offset=o, language=language, initial_prompt=initial_prompt
                    )
                )
                print(f"[ws-mic] Whisper: {len(result.segments)} segs")
                for seg in result.segments:
                    seg_data = {
                        "type":       "segment",
                        "text":       seg.text,
                        "ts":         secs_to_ts(seg.start),
                        "start":      seg.start,
                        "end":        seg.end,
                        "language":   result.language,
                        "fact_check": fc.check(seg.text),
                    }
                    session["segments"].append(seg_data)
                    broadcast(seg_data)   # llega una sola vez por /ws
                if result.segments:
                    offset = result.segments[-1].end

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        broadcast({"type": "status", "status": "stopped", "msg": "Micrófono desconectado."})


@app.on_event("startup")
async def startup():
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    # Pre-cargar Whisper en background para que la primera sesión arranque rápido
    tr.preload(WHISPER_MODEL, WHISPER_DEVICE)


@app.get("/")
def index():
    return HTMLResponse(HTML)


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live Transcriptor</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #0f1117; color: #e2e8f0; min-height: 100vh;
  display: flex; flex-direction: column;
}
.header {
  background: #1a1d2e; border-bottom: 1px solid #2d3748;
  padding: .85rem 1.5rem; display: flex; align-items: center;
  justify-content: space-between; flex-wrap: wrap; gap: .75rem;
}
.header-left { display: flex; align-items: center; gap: .6rem; }
.header h1 { font-size: 1.1rem; font-weight: 700; }
.header p { font-size: .75rem; color: #64748b; margin-top: .1rem; }
.header-right { display: flex; align-items: center; gap: .75rem; flex-wrap: wrap; }
.layout { display: flex; flex: 1; min-height: 0; }
.sidebar {
  width: 300px; flex-shrink: 0; background: #1a1d2e;
  border-right: 1px solid #2d3748; padding: 1.25rem 1rem;
  overflow-y: auto; display: flex; flex-direction: column; gap: 1rem;
}
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.transcript-area {
  flex: 1; overflow-y: auto; padding: 1.25rem 1.5rem;
  display: flex; flex-direction: column; gap: .5rem;
}
.toolbar {
  background: #1a1d2e; border-top: 1px solid #2d3748;
  padding: .6rem 1.5rem; display: flex; align-items: center;
  gap: 1rem; flex-wrap: wrap;
}

/* Sidebar */
.section-label {
  font-size: .68rem; font-weight: 700; color: #4a5568;
  text-transform: uppercase; letter-spacing: .06em; margin-bottom: .5rem;
}
.input-group { display: flex; flex-direction: column; gap: .35rem; }
.input-group label { font-size: .75rem; color: #718096; }
input[type=text], input[type=url], select {
  background: #0d1117; border: 1px solid #2d3748; border-radius: 6px;
  padding: .5rem .75rem; color: #e2e8f0; font-size: .85rem; width: 100%;
  outline: none; transition: border .15s;
}
input:focus, select:focus { border-color: #3b82f6; }
input::placeholder { color: #4a5568; }
select option { background: #1a1d2e; }
.toggle-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: .4rem 0;
}
.toggle-row label { font-size: .82rem; color: #a0aec0; }
.toggle {
  position: relative; width: 36px; height: 20px; flex-shrink: 0;
}
.toggle input { opacity: 0; width: 0; height: 0; }
.slider {
  position: absolute; inset: 0; background: #2d3748; border-radius: 20px;
  cursor: pointer; transition: background .2s;
}
.slider:before {
  content: ""; position: absolute; height: 14px; width: 14px;
  left: 3px; bottom: 3px; background: #fff; border-radius: 50%;
  transition: transform .2s;
}
input:checked + .slider { background: #3b82f6; }
input:checked + .slider:before { transform: translateX(16px); }

/* Botones */
.btn {
  padding: .5rem 1rem; border-radius: 6px; border: none; cursor: pointer;
  font-size: .85rem; font-weight: 600; transition: all .15s; white-space: nowrap;
}
.btn:disabled { opacity: .4; cursor: not-allowed; }
.btn-start   { background: #16a34a; color: #fff; width: 100%; padding: .65rem; font-size: .9rem; }
.btn-start:hover:not(:disabled) { background: #15803d; }
.btn-stop    { background: #dc2626; color: #fff; width: 100%; padding: .65rem; font-size: .9rem; }
.btn-stop:hover:not(:disabled)  { background: #b91c1c; }
.btn-ghost   { background: #2d3748; color: #a0aec0; }
.btn-ghost:hover:not(:disabled) { background: #374151; }

/* Status badge */
.status-dot {
  width: 8px; height: 8px; border-radius: 50%; background: #4a5568; flex-shrink: 0;
}
.status-dot.running { background: #16a34a; animation: pulse 1.5s infinite; }
.status-dot.error   { background: #dc2626; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.status-text { font-size: .78rem; color: #718096; }

/* Segments */
.segment {
  display: flex; gap: .75rem; align-items: flex-start;
  padding: .5rem .6rem; border-radius: 6px;
  border-left: 2px solid transparent; transition: background .1s;
}
.segment:hover { background: #1a1d2e; }
.segment.fc-verified     { border-left-color: #16a34a; }
.segment.fc-questionable { border-left-color: #d97706; }
.segment.fc-false        { border-left-color: #dc2626; }
.segment.fc-unverifiable { border-left-color: #6366f1; }
.seg-ts {
  font-family: 'SF Mono', monospace; font-size: .72rem; color: #4a5568;
  flex-shrink: 0; padding-top: .15rem; min-width: 48px;
}
.seg-body { flex: 1; }
.seg-text { font-size: .9rem; line-height: 1.55; color: #e2e8f0; }
.seg-fc {
  margin-top: .3rem; font-size: .72rem; padding: .18rem .5rem;
  border-radius: 4px; display: inline-block;
}
.fc-verified     .seg-fc { background: #14532d; color: #86efac; }
.fc-questionable .seg-fc { background: #78350f; color: #fde68a; }
.fc-false        .seg-fc { background: #7f1d1d; color: #fca5a5; }
.fc-unverifiable .seg-fc { background: #312e81; color: #c7d2fe; }

.empty-state {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; color: #4a5568; gap: .5rem; padding: 2rem;
}
.empty-state .icon { font-size: 2.5rem; }
.empty-state p { font-size: .85rem; text-align: center; max-width: 260px; line-height: 1.5; }

/* Tabs de fuente */
.source-tabs { display: flex; gap: .4rem; margin-bottom: .75rem; }
.source-tab {
  flex: 1; padding: .45rem; border-radius: 6px; border: 1px solid #2d3748;
  background: transparent; color: #64748b; font-size: .8rem; font-weight: 600;
  cursor: pointer; text-align: center; transition: all .15s;
}
.source-tab.active { background: #1e3a5f; border-color: #3b82f6; color: #93c5fd; }

.ws-indicator {
  width: 7px; height: 7px; border-radius: 50%; background: #4a5568;
  display: inline-block; margin-right: .35rem;
}
.ws-indicator.connected { background: #16a34a; }

@media (max-width: 640px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; border-right: none; border-bottom: 1px solid #2d3748; }
}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <span style="font-size:1.4rem">📡</span>
    <div>
      <h1>Live Transcriptor</h1>
      <p>Transcripción en tiempo real</p>
    </div>
  </div>
  <div class="header-right">
    <span class="ws-indicator" id="ws-dot"></span>
    <span class="status-dot" id="status-dot"></span>
    <span class="status-text" id="status-text">Sin sesión</span>
  </div>
</div>

<div class="layout">

  <!-- Sidebar de configuración -->
  <div class="sidebar">

    <div>
      <div class="section-label">Fuente de audio</div>
      <div class="source-tabs">
        <button class="source-tab active" id="tab-mic"     onclick="setSource('microphone')">🎙 Micrófono</button>
        <button class="source-tab"        id="tab-youtube" onclick="setSource('youtube')">▶ YouTube</button>
      </div>

      <!-- Panel micrófono: selector de motor -->
      <div id="panel-mic">
        <div class="section-label" style="margin-top:.5rem">Motor</div>
        <div class="source-tabs" id="engine-tabs">
          <button class="source-tab" id="eng-groq"   onclick="setEngine('groq')">🤖 Groq</button>
          <button class="source-tab" id="eng-speech" onclick="setEngine('speech')">🌐 Web Speech</button>
          <button class="source-tab active" id="eng-local" onclick="setEngine('local')">🖥 Local</button>
        </div>
        <!-- Panel Groq -->
        <div id="panel-groq" style="display:none" class="input-group">
          <label>Groq API Key</label>
          <input type="password" id="input-groq-key" placeholder="gsk_..."
                 oninput="localStorage.setItem('groq_key', this.value)">
          <span style="font-size:.7rem;color:#4a5568;margin-top:.2rem">
            Gratis en <strong>console.groq.com</strong> · Whisper large-v3-turbo
          </span>
        </div>
        <!-- Panel Web Speech -->
        <div id="panel-speech" style="display:none">
          <p style="font-size:.75rem;color:#718096;line-height:1.5;margin-top:.35rem">
            Usa el reconocimiento de voz del browser.<br>
            ✅ Instantáneo &nbsp;✅ Sin API key<br>
            ⚠️ Solo Chrome / Edge
          </p>
        </div>
      </div>

      <!-- Panel YouTube -->
      <div id="panel-youtube" style="display:none" class="input-group">
        <label>URL del video o stream</label>
        <input type="url" id="input-url" placeholder="https://youtube.com/watch?v=...">
      </div>
    </div>

    <div>
      <div class="section-label">Opciones</div>
      <div class="toggle-row" id="row-fc">
        <label>Fact-check <span style="color:#4a5568;font-size:.7rem">(slow)</span></label>
        <label class="toggle">
          <input type="checkbox" id="tog-fc" onchange="toggleFactcheck()">
          <span class="slider"></span>
        </label>
      </div>
    </div>

    <div class="input-group">
      <label>Idioma</label>
      <select id="sel-lang">
        <option value="">Auto-detectar</option>
        <option value="es">Español</option>
        <option value="en">English</option>
        <option value="pt">Português</option>
      </select>
    </div>

    <div class="input-group" id="row-prompt">
      <label>Prompt inicial (vocabulario clave)</label>
      <input type="text" id="input-prompt"
             placeholder="ej: San Lorenzo, Boedo, elecciones">
    </div>

    <div style="margin-top:auto; display:flex; flex-direction:column; gap:.5rem;">
      <button class="btn btn-start" id="btn-start" onclick="startSession()">▶ Iniciar</button>
      <button class="btn btn-stop"  id="btn-stop"  onclick="stopSession()" disabled>■ Detener</button>
    </div>

  </div>

  <!-- Área de transcripción -->
  <div class="main">
    <div class="transcript-area" id="transcript-area">
      <div class="empty-state" id="empty-state">
        <div class="icon">📡</div>
        <p>Configurá la fuente y presioná <strong>Iniciar</strong> para comenzar la transcripción en vivo.</p>
      </div>
    </div>

    <div class="toolbar">
      <button class="btn btn-ghost" onclick="clearTranscript()">🗑 Limpiar</button>
      <button class="btn btn-ghost" onclick="copyText()" title="Copiar texto sin timestamps">📋 Copiar</button>
      <button class="btn btn-ghost" onclick="downloadTranscript()">⬇ Descargar</button>
      <label style="display:flex;align-items:center;gap:.4rem;font-size:.78rem;color:#718096;cursor:pointer;margin-left:.5rem">
        <input type="checkbox" id="tog-timestamps" checked onchange="toggleTimestamps()"
               style="width:auto;accent-color:#3b82f6">
        Timestamps
      </label>
      <span id="seg-count" style="font-size:.75rem;color:#4a5568;margin-left:auto"></span>
    </div>
  </div>

</div>

<script>
// ── Estado ──────────────────────────────────────────────────────────────────
let ws           = null;
let activeSource = 'microphone';
let activeEngine = 'local';
let showTs       = true;
let segmentCount = 0;

// Recuperar API key guardada
document.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('groq_key');
  if (saved) document.getElementById('input-groq-key').value = saved;
});

// ── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    document.getElementById('ws-dot').classList.add('connected');
  };

  ws.onclose = () => {
    document.getElementById('ws-dot').classList.remove('connected');
    setTimeout(connectWS, 3000);
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMessage(msg);
  };
}

function handleMessage(msg) {
  if (msg.type === 'init') {
    // Re-render sesión existente al reconectar
    if (msg.session.segments && msg.session.segments.length) {
      document.getElementById('empty-state')?.remove();
      msg.session.segments.forEach(addSegment);
    }
    return;
  }
  if (msg.type === 'segment') {
    document.getElementById('empty-state')?.remove();
    addSegment(msg);
    return;
  }
  if (msg.type === 'status') {
    setStatus(msg.status, msg.msg);
    if (msg.status === 'stopped') {
      document.getElementById('btn-start').disabled = false;
      document.getElementById('btn-stop').disabled  = true;
    }
    return;
  }
  if (msg.type === 'error') {
    setStatus('error', msg.msg);
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled  = true;
  }
  // type==='debug' se ignora en la UI (va a los logs del servidor)
}

// ── Segmentos ────────────────────────────────────────────────────────────────
function addSegment(seg) {
  const area = document.getElementById('transcript-area');
  const fcClass = seg.fact_check ? 'fc-' + seg.fact_check.status : '';

  const div = document.createElement('div');
  div.className = `segment ${fcClass}`;
  div.innerHTML = `
    <span class="seg-ts" ${showTs ? '' : 'style="display:none"'}>${seg.ts}</span>
    <div class="seg-body">
      <div class="seg-text">${escHtml(seg.text)}</div>
      ${seg.fact_check ? `<span class="seg-fc">${fcLabel(seg.fact_check)}</span>` : ''}
    </div>
  `;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;

  segmentCount++;
  document.getElementById('seg-count').textContent =
    segmentCount + (segmentCount === 1 ? ' segmento' : ' segmentos');
}

function fcLabel(fc) {
  const labels = {
    verified:     '✓ Verificado',
    questionable: '? Dudoso',
    false:        '✗ Falso',
    unverifiable: '~ No verificable',
  };
  const text = labels[fc.status] || fc.status;
  return fc.explanation ? `${text} — ${escHtml(fc.explanation)}` : text;
}

// ── Controles ────────────────────────────────────────────────────────────────
function setSource(s) {
  activeSource = s;
  document.getElementById('tab-mic').classList.toggle('active', s === 'microphone');
  document.getElementById('tab-youtube').classList.toggle('active', s === 'youtube');
  document.getElementById('panel-mic').style.display     = s === 'microphone' ? 'block' : 'none';
  document.getElementById('panel-youtube').style.display = s === 'youtube'    ? 'block' : 'none';
}

function setEngine(e) {
  activeEngine = e;
  ['groq','speech','local'].forEach(id => {
    document.getElementById('eng-' + id).classList.toggle('active', id === e);
  });
  document.getElementById('panel-groq').style.display   = e === 'groq'   ? 'block' : 'none';
  document.getElementById('panel-speech').style.display = e === 'speech' ? 'block' : 'none';
  // Prompt no aplica a Web Speech (el browser no lo usa)
  document.getElementById('row-prompt').style.display   = e === 'speech' ? 'none'  : '';
}

function toggleTimestamps() {
  showTs = document.getElementById('tog-timestamps').checked;
  document.querySelectorAll('.seg-ts').forEach(el => {
    el.style.display = showTs ? '' : 'none';
  });
}

async function toggleFactcheck() {
  const on = document.getElementById('tog-fc').checked;
  const fd = new FormData();
  fd.append('enabled', on ? 'si' : 'no');
  await fetch('/api/toggle-factcheck', {method: 'POST', body: fd});
}

async function startSession() {
  const url    = document.getElementById('input-url').value.trim();
  const lang   = document.getElementById('sel-lang').value;
  const prompt = document.getElementById('input-prompt').value.trim();
  const fc     = document.getElementById('tog-fc').checked;
  const ts     = document.getElementById('tog-timestamps').checked;

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-stop').disabled  = false;
  document.getElementById('empty-state')?.remove();
  setStatus('starting', 'Conectando...');

  if (activeSource === 'microphone') {
    if (activeEngine === 'speech') {
      startWebSpeech();
    } else {
      await startMic();
    }
    return;
  }

  // YouTube
  if (!url) {
    alert('Ingresá la URL del video o stream.');
    resetUI(); return;
  }

  const fd = new FormData();
  fd.append('source',          'youtube');
  fd.append('url',             url);
  fd.append('fact_check',      fc ? 'si' : 'no');
  fd.append('show_timestamps', ts ? 'si' : 'no');
  fd.append('language',        lang);
  fd.append('initial_prompt',  prompt);

  try {
    const r = await fetch('/api/start', {method: 'POST', body: fd});
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      alert(d.detail || 'Error al iniciar la sesión.');
      resetUI();
    }
  } catch(e) {
    alert('Error de red.');
    resetUI();
  }
}

async function stopSession() {
  if (activeSource === 'microphone') {
    if (activeEngine === 'speech') stopWebSpeech();
    else stopMic();
  } else {
    await fetch('/api/stop', {method: 'POST'});
  }
  resetUI();
  setStatus('idle', 'Sesión detenida');
}

function resetUI() {
  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled  = true;
}

function clearTranscript() {
  const area = document.getElementById('transcript-area');
  area.innerHTML = '<div class="empty-state" id="empty-state"><div class="icon">📡</div><p>Transcripción limpiada.</p></div>';
  segmentCount = 0;
  document.getElementById('seg-count').textContent = '';
}

function downloadTranscript() {
  const ts = document.getElementById('tog-timestamps').checked ? 'si' : 'no';
  window.location.href = '/api/transcript?timestamps=' + ts;
}

async function copyText() {
  const lines = [...document.querySelectorAll('.seg-text')].map(el => el.textContent.trim());
  const text  = lines.join('\n');
  try {
    await navigator.clipboard.writeText(text);
    const btn = event.target;
    const orig = btn.textContent;
    btn.textContent = '✓ Copiado';
    setTimeout(() => btn.textContent = orig, 1500);
  } catch(e) {
    alert('No se pudo copiar: ' + e.message);
  }
}

function setStatus(status, msg) {
  const dot  = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  dot.className  = 'status-dot ' + (status === 'running' || status === 'starting' ? 'running' : status === 'error' ? 'error' : '');
  text.textContent = msg || status;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Micrófono (Web Audio API → WebSocket) ────────────────────────────────────
let micWs        = null;
let mediaRec     = null;
let micStream    = null;

async function startMic() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch(e) {
    alert('No se pudo acceder al micrófono: ' + e.message);
    resetUI(); return;
  }

  const lang    = document.getElementById('sel-lang').value;
  const prompt  = encodeURIComponent(document.getElementById('input-prompt').value.trim());
  const fc      = document.getElementById('tog-fc').checked ? 'si' : 'no';
  const proto   = location.protocol === 'https:' ? 'wss' : 'ws';
  const groqKey = encodeURIComponent(document.getElementById('input-groq-key')?.value.trim() || '');

  const MIME_CANDIDATES = ['audio/webm;codecs=opus','audio/webm','audio/mp4','audio/ogg;codecs=opus'];
  const mimeType = MIME_CANDIDATES.find(t => MediaRecorder.isTypeSupported(t)) || '';
  const audioExt = mimeType.includes('mp4') ? '.mp4' : mimeType.includes('ogg') ? '.ogg' : '.webm';

  const url = `${proto}://${location.host}/ws-mic?language=${lang}&prompt=${prompt}&fc=${fc}` +
              `&ext=${encodeURIComponent(audioExt)}&engine=${activeEngine}&key=${groqKey}`;

  micWs = new WebSocket(url);
  micWs.binaryType = 'arraybuffer';

  micWs.onopen = () => {
    console.log(`[mic] mimeType: "${mimeType || '(default)'}" ext=${audioExt} engine=${activeEngine}`);
    function startChunk() {
      if (!micStream || !micWs || micWs.readyState !== WebSocket.OPEN) return;
      const rec = new MediaRecorder(micStream, mimeType ? { mimeType } : {});
      mediaRec  = rec;
      rec.ondataavailable = (e) => {
        if (e.data.size > 0 && micWs && micWs.readyState === WebSocket.OPEN) micWs.send(e.data);
      };
      rec.onstop = () => startChunk();
      rec.start();
      setTimeout(() => { if (rec.state === 'recording') rec.stop(); }, 6000);
    }
    startChunk();
    setStatus('running', 'Escuchando micrófono...');
  };

  micWs.onmessage = (e) => { handleMessage(JSON.parse(e.data)); };
  micWs.onclose   = () => stopMic();
}

function stopMic() {
  if (micWs && micWs.readyState === WebSocket.OPEN) micWs.close();
  micWs = null;
  if (mediaRec && mediaRec.state !== 'inactive') mediaRec.stop();
  mediaRec = null;
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
}

// ── Web Speech API ────────────────────────────────────────────────────────────
let speechRec  = null;
let interimDiv = null;

function startWebSpeech() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { alert('Web Speech API no disponible. Usá Chrome o Edge.'); resetUI(); return; }
  const lang = document.getElementById('sel-lang').value || 'es-AR';

  speechRec = new SR();
  speechRec.continuous     = true;
  speechRec.interimResults = true;
  speechRec.lang           = lang;

  speechRec.onresult = async (e) => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) {
        const text = r[0].transcript.trim();
        if (text) {
          removeInterim();
          document.getElementById('empty-state')?.remove();
          addSegment({ text, ts: '', fact_check: null });
          const fd = new FormData(); fd.append('text', text);
          fetch('/api/speech-segment', { method: 'POST', body: fd });
        }
      } else { interim += r[0].transcript; }
    }
    showInterim(interim);
  };

  speechRec.onerror = (e) => {
    if (e.error !== 'no-speech') setStatus('error', 'Web Speech: ' + e.error);
  };
  speechRec.onend = () => { if (speechRec) speechRec.start(); };
  speechRec.start();
  setStatus('running', 'Web Speech escuchando...');
}

function stopWebSpeech() {
  if (speechRec) { speechRec.onend = null; speechRec.stop(); speechRec = null; }
  removeInterim();
}

function showInterim(text) {
  if (!text) { removeInterim(); return; }
  if (!interimDiv) {
    interimDiv = document.createElement('div');
    interimDiv.style.cssText = 'padding:.5rem .6rem;color:#4a5568;font-style:italic;font-size:.9rem';
    document.getElementById('transcript-area').appendChild(interimDiv);
  }
  interimDiv.textContent = text;
  document.getElementById('transcript-area').scrollTop = 9999;
}

function removeInterim() {
  if (interimDiv) { interimDiv.remove(); interimDiv = null; }
}

// ── Init ─────────────────────────────────────────────────────────────────────
connectWS();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")

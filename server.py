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

import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
import uvicorn

import transcriber as tr
from sources import YouTubeSource, MicrophoneSource
from llm import get_provider
from fact_checker import FactChecker

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL",  "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST",    "http://host.docker.internal:11434")
OLLAMA_MODELO  = os.getenv("OLLAMA_MODELO",  "llama3")
LLM_PROVIDER   = os.getenv("LLM_PROVIDER",  "ollama")

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
    WebSocket de audio: el browser manda chunks de audio WebM/Opus,
    el server los convierte a PCM con ffmpeg y transcribe con Whisper.
    """
    await ws.accept()

    language       = ws.query_params.get("language", "") or None
    initial_prompt = ws.query_params.get("prompt", "")   or None
    fact_check_on  = ws.query_params.get("fc", "no") == "si"

    whisper  = tr.get_transcriber(WHISPER_MODEL, WHISPER_DEVICE)
    provider = get_provider(LLM_PROVIDER, host=OLLAMA_HOST, model=OLLAMA_MODELO)
    fc       = FactChecker(provider, enabled=fact_check_on)
    offset   = 0.0

    broadcast({"type": "status", "status": "running", "msg": "Micrófono conectado. Escuchando..."})

    try:
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_bytes(), timeout=30)
            except asyncio.TimeoutError:
                break

            # Convertir WebM/Opus → PCM 16kHz mono con ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(data)
                tmp_in = f.name
            tmp_out = tmp_in + ".pcm"

            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", tmp_in,
                    "-ar", "16000", "-ac", "1", "-f", "s16le", tmp_out,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()

                if not os.path.exists(tmp_out) or os.path.getsize(tmp_out) < 512:
                    continue

                audio = (
                    np.frombuffer(open(tmp_out, "rb").read(), dtype=np.int16)
                    .astype(np.float32) / 32_768.0
                )
            finally:
                os.unlink(tmp_in)
                if os.path.exists(tmp_out):
                    os.unlink(tmp_out)

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda a=audio, o=offset: whisper.transcribe(
                    a, offset=o, language=language, initial_prompt=initial_prompt
                )
            )

            for seg in result.segments:
                fc_result = fc.check(seg.text)
                seg_data  = {
                    "type":       "segment",
                    "text":       seg.text,
                    "ts":         secs_to_ts(seg.start),
                    "start":      seg.start,
                    "end":        seg.end,
                    "language":   result.language,
                    "fact_check": fc_result,
                }
                session["segments"].append(seg_data)
                await ws.send_json(seg_data)
                broadcast(seg_data)

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

      <div id="panel-youtube" style="display:none" class="input-group">
        <label>URL del video o stream</label>
        <input type="url" id="input-url" placeholder="https://youtube.com/watch?v=...">
      </div>
    </div>

    <div>
      <div class="section-label">Opciones</div>
      <div class="toggle-row">
        <label>Timestamps</label>
        <label class="toggle">
          <input type="checkbox" id="tog-timestamps" checked onchange="toggleTimestamps()">
          <span class="slider"></span>
        </label>
      </div>
      <div class="toggle-row">
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

    <div class="input-group">
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
      <button class="btn btn-ghost" onclick="downloadTranscript()">⬇ Descargar .txt</button>
      <span id="seg-count" style="font-size:.75rem;color:#4a5568;margin-left:auto"></span>
    </div>
  </div>

</div>

<script>
// ── Estado ──────────────────────────────────────────────────────────────────
let ws          = null;
let activeSource = 'microphone';
let showTs       = true;
let segmentCount = 0;

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
  document.getElementById('panel-youtube').style.display = s === 'youtube' ? 'block' : 'none';
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
    await startMic();
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
    stopMic();
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
    resetUI();
    return;
  }

  const lang   = document.getElementById('sel-lang').value;
  const prompt = encodeURIComponent(document.getElementById('input-prompt').value.trim());
  const fc     = document.getElementById('tog-fc').checked ? 'si' : 'no';
  const proto  = location.protocol === 'https:' ? 'wss' : 'ws';
  const url    = `${proto}://${location.host}/ws-mic?language=${lang}&prompt=${prompt}&fc=${fc}`;

  micWs = new WebSocket(url);
  micWs.binaryType = 'arraybuffer';

  micWs.onopen = () => {
    // MediaRecorder manda chunks de ~3s al WebSocket
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus' : 'audio/webm';

    mediaRec = new MediaRecorder(micStream, { mimeType });
    mediaRec.ondataavailable = (e) => {
      if (e.data.size > 0 && micWs.readyState === WebSocket.OPEN) {
        micWs.send(e.data);
      }
    };
    mediaRec.start(3000);  // chunk cada 3 segundos
    setStatus('running', 'Escuchando micrófono...');
  };

  micWs.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleMessage(msg);
  };

  micWs.onclose = () => stopMic(false);
}

function stopMic(sendStop = true) {
  if (mediaRec && mediaRec.state !== 'inactive') mediaRec.stop();
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (micWs && micWs.readyState === WebSocket.OPEN) micWs.close();
  micWs = null; mediaRec = null;
}

// ── Init ─────────────────────────────────────────────────────────────────────
connectWS();
</script>
</body>
</html>"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083, log_level="info")

"""
Wrapper de faster-whisper.
El modelo se carga una sola vez (singleton) en un thread de fondo al arrancar.
"""
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

SAMPLE_RATE = 16_000  # Hz requerido por Whisper


@dataclass
class Segment:
    text:  str
    start: float   # segundos desde el inicio de la sesión
    end:   float


@dataclass
class TranscriptChunk:
    segments: List[Segment] = field(default_factory=list)
    language: str = ""


class Transcriber:
    def __init__(self, model_size: str = "medium", device: str = "cpu"):
        from faster_whisper import WhisperModel
        print(f"[whisper] Cargando modelo '{model_size}' en {device}...")
        self._model = WhisperModel(
            model_size, device=device,
            compute_type="int8",          # más rápido en CPU sin pérdida notable
        )
        print("[whisper] Modelo listo.")

    def transcribe(
        self,
        audio: np.ndarray,
        offset: float = 0.0,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptChunk:
        """
        audio   : array float32 normalizado [-1, 1], mono, 16kHz
        offset  : segundos a sumar a todos los timestamps del chunk
        """
        segs, info = self._model.transcribe(
            audio,
            language=language or None,
            initial_prompt=initial_prompt,
            vad_filter=True,          # silencia fragmentos sin voz → menos alucinaciones
            word_timestamps=False,    # no los necesitamos, sería más lento
        )

        result = TranscriptChunk(language=info.language)
        for s in segs:
            text = s.text.strip()
            # Descartar segmentos sin habla (alucinaciones de Whisper)
            # Umbral alto: solo filtrar silencio claro, no voz con ruido de fondo
            if text and s.no_speech_prob < 0.85:
                result.segments.append(Segment(
                    text  = text,
                    start = offset + s.start,
                    end   = offset + s.end,
                ))

        return result


# ── Singleton con carga en background ────────────────────────────────────────

_instance: Optional[Transcriber] = None
_lock      = threading.Lock()
_loading   = False


def preload(model_size: str = "medium", device: str = "cpu"):
    """Llamar al inicio para cargar el modelo sin bloquear el server."""
    def _load():
        global _instance, _loading
        _loading = True
        with _lock:
            if _instance is None:
                _instance = Transcriber(model_size, device)
        _loading = False
    threading.Thread(target=_load, daemon=True).start()


def get_transcriber(model_size: str = "medium", device: str = "cpu") -> Transcriber:
    global _instance
    with _lock:
        if _instance is None:
            _instance = Transcriber(model_size, device)
    return _instance

"""
Fuentes de audio para transcripción en vivo.

YouTubeSource  : usa yt-dlp para obtener la URL del stream y ffmpeg para decodificarlo.
MicrophoneSource: usa sounddevice para capturar el micrófono del host.

Ambas producen chunks numpy float32 16kHz mono de ~CHUNK_SECONDS segundos
con OVERLAP_SECONDS de superposición para evitar cortar palabras en el borde.
"""
import subprocess
import threading
import queue
import time
import numpy as np
from typing import Optional

SAMPLE_RATE     = 16_000   # Hz
CHUNK_SECONDS   = 20       # duración de cada chunk a transcribir
OVERLAP_SECONDS = 2        # solapamiento entre chunks para no cortar palabras
CHUNK_SAMPLES   = SAMPLE_RATE * CHUNK_SECONDS
OVERLAP_SAMPLES = SAMPLE_RATE * OVERLAP_SECONDS


class _BaseSource:
    def __init__(self):
        self._stop  = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_chunk(self, timeout: float = 60.0) -> Optional[np.ndarray]:
        """
        Retorna el próximo chunk o None si la fuente terminó / hubo error.
        Lanza queue.Empty si se agotó el timeout.
        """
        return self._queue.get(timeout=timeout)

    def _run(self):
        raise NotImplementedError


class YouTubeSource(_BaseSource):
    """
    Descarga el audio de un video o stream de YouTube y lo entrega en chunks.
    Funciona tanto con VODs como con streams en vivo (yt-dlp maneja ambos).
    """
    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def _run(self):
        try:
            # Paso 1: obtener URL directa del audio (sin descargar el archivo)
            result = subprocess.run(
                ["yt-dlp", "-f", "bestaudio", "-g", "--no-playlist", self.url],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp error: {result.stderr.strip()}")

            audio_url = result.stdout.strip().splitlines()[0]

            # Paso 2: ffmpeg decodifica el audio a PCM 16kHz mono
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-i", audio_url,
                    "-vn",                          # sin video
                    "-ar", str(SAMPLE_RATE),        # 16kHz
                    "-ac", "1",                     # mono
                    "-f", "s16le",                  # PCM 16-bit little-endian
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            buffer     = b""
            chunk_bytes   = CHUNK_SAMPLES   * 2    # 2 bytes por muestra (int16)
            overlap_bytes = OVERLAP_SAMPLES * 2

            while not self._stop.is_set():
                needed = chunk_bytes - len(buffer)
                data   = ffmpeg.stdout.read(needed)
                if not data:
                    break
                buffer += data
                if len(buffer) >= chunk_bytes:
                    arr = (
                        np.frombuffer(buffer[:chunk_bytes], dtype=np.int16)
                        .astype(np.float32) / 32_768.0
                    )
                    self._queue.put(arr)
                    buffer = buffer[chunk_bytes - overlap_bytes:]   # mantener overlap

            # Procesar lo que quedó en el buffer (último fragmento)
            if len(buffer) > 4000 and not self._stop.is_set():
                arr = (
                    np.frombuffer(buffer, dtype=np.int16)
                    .astype(np.float32) / 32_768.0
                )
                self._queue.put(arr)

        except Exception as e:
            self.error = str(e)
        finally:
            self._queue.put(None)   # señal de fin
            try:
                ffmpeg.kill()
            except Exception:
                pass


class MicrophoneSource(_BaseSource):
    """
    Captura el micrófono del sistema.

    Nota Docker: requiere --device /dev/snd y configuración de PulseAudio.
    Funciona sin problemas en ejecución local (fuera de Docker).
    """
    def __init__(self, device: Optional[int] = None):
        super().__init__()
        self.device = device   # None = dispositivo predeterminado

    def _run(self):
        try:
            import sounddevice as sd

            buffer = np.array([], dtype=np.float32)
            ready  = threading.Event()

            def callback(indata, frames, time_info, status):
                nonlocal buffer
                buffer = np.append(buffer, indata[:, 0])
                if len(buffer) >= CHUNK_SAMPLES:
                    chunk  = buffer[:CHUNK_SAMPLES].copy()
                    buffer = buffer[CHUNK_SAMPLES - OVERLAP_SAMPLES:]
                    self._queue.put(chunk)

            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.device,
                callback=callback,
                blocksize=SAMPLE_RATE // 2,   # 0.5s de latencia mínima
            ):
                while not self._stop.is_set():
                    time.sleep(0.1)

        except Exception as e:
            self.error = str(e)
        finally:
            self._queue.put(None)

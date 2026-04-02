FROM python:3.11-slim

# ffmpeg + yt-dlp deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    libasound2-dev \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp
RUN curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Modelo Whisper se descarga en el primer uso y se cachea en el volumen
ENV WHISPER_MODEL=medium
ENV WHISPER_DEVICE=cpu
ENV OLLAMA_HOST=http://host.docker.internal:11434
ENV OLLAMA_MODELO=llama3
ENV LLM_PROVIDER=ollama

EXPOSE 8083

CMD ["python3", "server.py"]

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY audio/requirements.txt /app/audio/requirements.txt
RUN pip install --no-cache-dir -r /app/audio/requirements.txt

COPY . /app

EXPOSE 8001

CMD ["uvicorn", "audio.server:app", "--host", "0.0.0.0", "--port", "8001"]

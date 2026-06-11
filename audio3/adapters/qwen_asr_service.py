"""
FastAPI service wrapper for Qwen ASR.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from audio.adapters import qwen_asr_adapter
from audio.config.settings import get_settings
from audio.models.schemas import NormalizedAudioRecord

app = FastAPI(title="Audio Qwen ASR Adapter")


class TranscribeRequest(BaseModel):
    source_id: str
    audio_path: str
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    codec: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "audio-qwen-asr-adapter"}


@app.post("/transcribe")
def transcribe(request: TranscribeRequest) -> dict[str, Any]:
    settings = get_settings().model_copy(update={"qwen_asr_endpoint": ""})
    record = NormalizedAudioRecord(
        source_id=request.source_id,
        original_path=request.audio_path,
        normalized_path=request.audio_path,
        sample_rate=request.sample_rate,
        channels=request.channels,
        codec=request.codec,
        duration_seconds=request.duration_seconds,
        engine_name="qwen_asr_service",
        metadata=request.metadata,
    )
    segments = qwen_asr_adapter.transcribe_local(record, settings)
    return {
        "provider": "qwen_asr",
        "model": settings.qwen_asr_model,
        "segments": [segment.model_dump(mode="json") for segment in segments],
    }

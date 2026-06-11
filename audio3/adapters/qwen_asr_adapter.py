"""
Qwen ASR adapter.

This module owns model loading, device selection, raw ASR output parsing, and
conversion into the pipeline's ASRSegment schema.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import ASRSegment, NormalizedAudioRecord

logger = logging.getLogger(__name__)

_qwen_asr_pipeline = None
_qwen_asr_cache_key: tuple[str, str] | None = None
_qwen_asr_lock = Lock()


def resolve_device(settings: Settings) -> str:
    requested = str(getattr(settings, "qwen_asr_device", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_pipeline(settings: Settings, device: str):
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError("qwen-asr unavailable for Qwen ASR") from exc

    logger.info("Loading Qwen ASR model: %s on %s", settings.qwen_asr_model, device)
    device_map = device
    if device == "cuda":
        device_map = "cuda:0"

    dtype = torch.bfloat16 if str(device_map).startswith("cuda") else torch.float32

    return Qwen3ASRModel.from_pretrained(
        settings.qwen_asr_model,
        device_map=device_map,
        dtype=dtype,
        max_new_tokens=512,
    )


def load_pipeline(settings: Settings):
    global _qwen_asr_cache_key, _qwen_asr_pipeline

    device = resolve_device(settings)
    cache_key = (settings.qwen_asr_model, device)
    if _qwen_asr_pipeline is not None and _qwen_asr_cache_key == cache_key:
        return _qwen_asr_pipeline

    with _qwen_asr_lock:
        if _qwen_asr_pipeline is not None and _qwen_asr_cache_key == cache_key:
            return _qwen_asr_pipeline
        _qwen_asr_pipeline = build_pipeline(settings, device)
        _qwen_asr_cache_key = cache_key
        return _qwen_asr_pipeline


def reset_cache() -> None:
    global _qwen_asr_cache_key, _qwen_asr_pipeline

    with _qwen_asr_lock:
        _qwen_asr_pipeline = None
        _qwen_asr_cache_key = None


def _chunk_bounds(chunk: dict[str, Any], fallback_end: float) -> tuple[float, float]:
    timestamp = chunk.get("timestamp", (0.0, fallback_end))
    if not isinstance(timestamp, (list, tuple)) or len(timestamp) < 2:
        return 0.0, fallback_end
    start = timestamp[0] if timestamp[0] is not None else 0.0
    end = timestamp[1] if timestamp[1] is not None else fallback_end
    return float(start or 0.0), float(end or fallback_end)


def _normalize_output(output: Any, record: NormalizedAudioRecord) -> tuple[list[dict[str, Any]], str]:
    if isinstance(output, str):
        return [{"timestamp": (0.0, record.duration_seconds), "text": output}], ""
    if not isinstance(output, dict):
        return [], ""

    chunks = output.get("chunks") or []
    if not chunks and output.get("text"):
        chunks = [{"timestamp": (0.0, record.duration_seconds), "text": output.get("text", "")}]
    return [chunk for chunk in chunks if isinstance(chunk, dict)], str(output.get("language", ""))


def _segments_from_rows(rows: list[dict[str, Any]], source_id: str) -> list[ASRSegment]:
    segments: list[ASRSegment] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        payload = dict(row)
        payload["source_id"] = source_id
        segments.append(ASRSegment.model_validate(payload))
    return segments


def transcribe_endpoint(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment] | None:
    if not settings.qwen_asr_endpoint:
        return None

    import httpx

    response = httpx.post(
        settings.qwen_asr_endpoint,
        json={
            "source_id": record.source_id,
            "audio_path": record.normalized_path,
            "duration_seconds": record.duration_seconds,
        },
        timeout=settings.qwen_asr_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Qwen ASR endpoint must return a JSON object.")
    return _segments_from_rows(payload.get("segments") or [], record.source_id)


def transcribe_local(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    asr = load_pipeline(settings)
    results = asr.transcribe(audio=record.normalized_path)
    if not results:
        return []

    result = results[0]
    text = str(getattr(result, "text", "") or "").strip()
    language = str(getattr(result, "language", "") or "")
    if not text:
        return []

    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=0.0,
            end_time=record.duration_seconds,
            text=text,
            confidence=0.9,
            engine_name="qwen3-asr",
            language=language,
        )
    ]


def transcribe(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    endpoint_segments = transcribe_endpoint(record, settings)
    if endpoint_segments is not None:
        return endpoint_segments
    return transcribe_local(record, settings)

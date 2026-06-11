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
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        raise RuntimeError("transformers unavailable for Qwen ASR") from exc

    logger.info("Loading Qwen ASR model: %s on %s", settings.qwen_asr_model, device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        settings.qwen_asr_model,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    processor = AutoProcessor.from_pretrained(settings.qwen_asr_model, trust_remote_code=True)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
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


def transcribe(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    asr = load_pipeline(settings)
    output = asr(record.normalized_path)
    chunks, language = _normalize_output(output, record)

    segments: list[ASRSegment] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        start_time, end_time = _chunk_bounds(chunk, record.duration_seconds)
        segments.append(
            ASRSegment(
                source_id=record.source_id,
                start_time=start_time,
                end_time=end_time,
                text=text,
                confidence=0.9,
                engine_name="qwen3-asr",
                language=language,
            )
        )
    return segments


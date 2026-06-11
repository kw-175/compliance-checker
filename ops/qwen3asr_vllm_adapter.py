from __future__ import annotations

import os
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Qwen3-ASR vLLM Adapter")

MODEL_PATH = os.getenv("QWEN3ASR_MODEL", "/data/kw/compliance-checker/models/Qwen/Qwen3-ASR-0.6B")
GPU_MEMORY_UTILIZATION = float(os.getenv("QWEN3ASR_GPU_MEMORY_UTILIZATION", "0.12"))
MAX_INFERENCE_BATCH_SIZE = int(os.getenv("QWEN3ASR_MAX_INFERENCE_BATCH_SIZE", "1"))
MAX_NEW_TOKENS = int(os.getenv("QWEN3ASR_MAX_NEW_TOKENS", "2048"))
MAX_MODEL_LEN = int(os.getenv("QWEN3ASR_MAX_MODEL_LEN", "4096"))
TENSOR_PARALLEL_SIZE = int(os.getenv("QWEN3ASR_TENSOR_PARALLEL_SIZE", "1"))
DEVICE = os.getenv("QWEN3ASR_DEVICE", "cuda")
FORCED_ALIGNER = os.getenv("QWEN3ASR_FORCED_ALIGNER", "").strip() or None
RETURN_TIME_STAMPS = os.getenv("QWEN3ASR_RETURN_TIME_STAMPS", "false").lower() in {"1", "true", "yes"}

_model = None
_lock = Lock()


class TranscribeRequest(BaseModel):
    source_id: str
    audio_path: str
    duration_seconds: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    codec: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_model():
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model
        if DEVICE.strip().lower() == "cpu":
            raise RuntimeError("Qwen3-ASR vLLM backend requires a CUDA device; use the transformers backend for CPU ASR.")
        from qwen_asr import Qwen3ASRModel

        _model = Qwen3ASRModel.LLM(
            model=MODEL_PATH,
            forced_aligner=FORCED_ALIGNER,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            max_model_len=MAX_MODEL_LEN,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        )
        return _model


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "qwen3-asr-vllm-adapter",
        "model": MODEL_PATH,
        "device": DEVICE,
        "loaded": _model is not None,
        "backend": "vllm",
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "forced_aligner": FORCED_ALIGNER,
        "return_time_stamps": RETURN_TIME_STAMPS,
    }


@app.get("/ready")
async def ready() -> dict[str, Any]:
    try:
        load_model()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "status": "ready",
        "service": "qwen3-asr-vllm-adapter",
        "model": MODEL_PATH,
        "backend": "vllm",
        "loaded": True,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
    }


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _alignment_items(result: Any) -> list[Any]:
    time_stamps = getattr(result, "time_stamps", None)
    if time_stamps is None and isinstance(result, dict):
        time_stamps = result.get("time_stamps") or result.get("timestamps")
    if time_stamps is None:
        return []
    items = getattr(time_stamps, "items", None)
    if items is None and isinstance(time_stamps, dict):
        items = time_stamps.get("items")
    if not isinstance(items, list):
        return []
    return items


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text", "") or "").strip()
    return str(getattr(item, "text", "") or "").strip()


def _item_bounds(item: Any, default_start: float, default_end: float) -> tuple[float, float]:
    if isinstance(item, dict):
        start = item.get("start_time", item.get("start", default_start))
        end = item.get("end_time", item.get("end", default_end))
    else:
        start = getattr(item, "start_time", getattr(item, "start", default_start))
        end = getattr(item, "end_time", getattr(item, "end", default_end))
    return _float_or_default(start, default_start), _float_or_default(end, default_end)


def _segment_metadata(timestamp_granularity: str) -> dict[str, Any]:
    return {
        "asr_backend": "qwen3-asr-vllm",
        "timestamp_granularity": timestamp_granularity,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
    }


@app.post("/transcribe")
async def transcribe(request: TranscribeRequest) -> dict[str, Any]:
    model = load_model()
    use_time_stamps = bool(RETURN_TIME_STAMPS and FORCED_ALIGNER)
    results = model.transcribe(audio=request.audio_path, return_time_stamps=use_time_stamps)

    segments = []
    if results:
        result = results[0]
        text = str(getattr(result, "text", "") or "").strip()
        language = str(getattr(result, "language", "") or "")
        for item in _alignment_items(result):
            item_text = _item_text(item)
            if not item_text:
                continue
            start_time, end_time = _item_bounds(item, 0.0, request.duration_seconds)
            segments.append(
                {
                    "source_id": request.source_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "text": item_text,
                    "confidence": 0.9,
                    "engine_name": "qwen3-asr-vllm",
                    "language": language,
                    "metadata": _segment_metadata("forced_alignment"),
                }
            )
        if text and not segments:
            segments.append(
                {
                    "source_id": request.source_id,
                    "start_time": 0.0,
                    "end_time": request.duration_seconds,
                    "text": text,
                    "confidence": 0.9,
                    "engine_name": "qwen3-asr-vllm",
                    "language": language,
                    "metadata": _segment_metadata("whole_audio"),
                }
            )

    return {
        "provider": "qwen_asr",
        "model": MODEL_PATH,
        "backend": "vllm",
        "segments": segments,
    }

"""
FastAPI service wrapper for Qwen Guard moderation.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from audio.adapters import qwen_guard_adapter
from audio.config.settings import get_settings

app = FastAPI(title="Audio Qwen Guard Adapter")


class ModerateRequest(BaseModel):
    unit_id: str = ""
    text: str
    model: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "audio-qwen-guard-adapter"}


@app.post("/moderate")
def moderate(request: ModerateRequest) -> dict[str, Any]:
    base_settings = get_settings()
    settings = base_settings.model_copy(
        update={
            "qwen_guard_endpoint": "",
            "qwen_guard_model": request.model or base_settings.qwen_guard_model,
        }
    )
    result = qwen_guard_adapter.moderate_local(request.text, settings)
    return {
        "unit_id": request.unit_id,
        "provider": "qwen_guard",
        "model": result.model_version,
        "safety": result.level.value,
        "categories": result.categories,
        "raw_output": result.raw_output,
    }

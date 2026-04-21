"""
FastAPI service wrapper for Qwen hard-case adjudication.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from audio.adapters import qwen_hard_case_adapter
from audio.config.settings import get_settings

app = FastAPI(title="Audio Qwen Hard-case Adapter")


class AdjudicateRequest(BaseModel):
    model: str = ""
    prompt: str
    temperature: float = 0.0


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "audio-qwen-hard-case-adapter"}


@app.post("/adjudicate")
def adjudicate(request: AdjudicateRequest) -> dict[str, Any]:
    base_settings = get_settings()
    settings = base_settings.model_copy(
        update={
            "hard_case_endpoint": "",
            "hard_case_model_name": request.model or base_settings.hard_case_model_name,
        }
    )
    judgement, raw_response, note = qwen_hard_case_adapter.call_local_model(request.prompt, settings)
    if judgement is None:
        return {
            "provider": "qwen_hard_case",
            "model": request.model or settings.hard_case_model_name,
            "judgement": None,
            "raw_output": raw_response,
            "notes": [note] if note else ["local_model_returned_no_judgement"],
        }
    return {
        "provider": "qwen_hard_case",
        "model": request.model or settings.hard_case_model_name,
        "judgement": judgement.model_dump(mode="json"),
        "raw_output": raw_response,
        "notes": [note] if note else [],
    }

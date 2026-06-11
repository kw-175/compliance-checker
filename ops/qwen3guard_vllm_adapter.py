from __future__ import annotations

import os
import re
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Qwen3Guard vLLM Adapter")

VLLM_URL = os.getenv("QWEN3GUARD_VLLM_URL", "http://127.0.0.1:8212/v1/chat/completions")
API_KEY = os.getenv("QWEN3GUARD_API_KEY", "guard-token")
BACKEND_MODEL = os.getenv("QWEN3GUARD_VLLM_MODEL", "Qwen3Guard-Gen-0.6B")
DEFAULT_MODEL = os.getenv("QWEN3GUARD_MODEL", BACKEND_MODEL)
TIMEOUT_SECONDS = int(os.getenv("QWEN3GUARD_TIMEOUT_SECONDS", "120"))
MAX_TOKENS = int(os.getenv("QWEN3GUARD_MAX_TOKENS", "128"))

LABEL_SCORES = {
    "safe": 0.99,
    "controversial": 0.65,
    "unsafe": 0.95,
}


class ModerateRequest(BaseModel):
    unit_id: str = ""
    text: str
    model: str = DEFAULT_MODEL


def _extract_safety(raw_output: str) -> str:
    match = re.search(r"Safety:\s*(Safe|Unsafe|Controversial)", raw_output, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    normalized = raw_output.lower()
    if "unsafe" in normalized:
        return "unsafe"
    if "controversial" in normalized or "borderline" in normalized:
        return "controversial"
    if "safe" in normalized:
        return "safe"
    return "controversial"


def _extract_categories(raw_output: str) -> list[str]:
    match = re.search(r"Categories:\s*(.*)", raw_output, re.IGNORECASE)
    if not match:
        return []

    raw_categories = match.group(1).strip()
    if raw_categories.lower() == "none":
        return []

    return [
        item.strip()
        for item in re.split(r"[,;]", raw_categories)
        if item.strip() and item.strip().lower() != "none"
    ]


def parse_guard_output(raw_output: str) -> dict[str, Any]:
    safety = _extract_safety(raw_output)
    categories = _extract_categories(raw_output)
    return {
        "safety": safety,
        "categories": categories,
        "score": LABEL_SCORES[safety],
        "score_source": "rule_based_label_mapping",
        "raw_output": raw_output,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "qwen3guard-vllm-adapter",
        "backend_url": VLLM_URL,
        "backend_model": BACKEND_MODEL,
    }


@app.post("/moderate")
async def moderate(request: ModerateRequest) -> dict[str, Any]:
    payload = {
        "model": BACKEND_MODEL,
        "messages": [{"role": "user", "content": request.text}],
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(VLLM_URL, headers=headers, json=payload)

    response.raise_for_status()
    data = response.json()
    raw_output = data["choices"][0]["message"]["content"]
    parsed = parse_guard_output(raw_output)

    return {
        "unit_id": request.unit_id,
        "provider": "qwen_guard_endpoint",
        "model": request.model or DEFAULT_MODEL,
        **parsed,
    }

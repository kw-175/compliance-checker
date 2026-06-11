from __future__ import annotations

import os
import re
from typing import Any

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Qwen3Guard Adapter")

VLLM_URL = os.getenv("QWEN3GUARD_VLLM_URL", "http://127.0.0.1:8155/v1/chat/completions")
API_KEY = os.getenv("QWEN3GUARD_API_KEY", "guard-token")
MODEL = os.getenv("QWEN3GUARD_MODEL", "Qwen3Guard-Gen-0.6B")

LABEL_SCORES = {
    "safe": 0.99,
    "controversial": 0.65,
    "unsafe": 0.95,
}


class ModerateRequest(BaseModel):
    doc_id: str = ""
    text: str
    model: str = MODEL


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "qwen3guard-adapter",
        "backend_url": VLLM_URL,
        "model": MODEL,
    }


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


@app.post("/moderate")
async def moderate(request: ModerateRequest) -> dict[str, Any]:
    payload = {
        "model": request.model or MODEL,
        "messages": [{"role": "user", "content": request.text}],
        "temperature": 0,
        "max_tokens": 128,
    }
    headers: dict[str, str] = {}
    if API_KEY.strip():
        headers["Authorization"] = f"Bearer {API_KEY}"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(VLLM_URL, headers=headers, json=payload)

    response.raise_for_status()
    data = response.json()
    raw_output = data["choices"][0]["message"]["content"]
    parsed = parse_guard_output(raw_output)

    return {
        "doc_id": request.doc_id,
        "provider": "qwen3guard",
        "model": request.model or MODEL,
        **parsed,
    }

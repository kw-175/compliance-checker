from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Qwen3.5 Hard-case vLLM Adapter")

VLLM_URL = os.getenv("QWEN35_HARDCASE_VLLM_URL", "http://127.0.0.1:8213/v1/chat/completions")
API_KEY = os.getenv("QWEN35_HARDCASE_API_KEY", "hardcase-token")
BACKEND_MODEL = os.getenv("QWEN35_HARDCASE_VLLM_MODEL", "Qwen3.5-9B")
DEFAULT_MODEL = os.getenv("QWEN35_HARDCASE_MODEL", BACKEND_MODEL)
TIMEOUT_SECONDS = int(os.getenv("QWEN35_HARDCASE_TIMEOUT_SECONDS", "180"))
MAX_TOKENS = int(os.getenv("QWEN35_HARDCASE_MAX_TOKENS", "1024"))

SYSTEM_PROMPT = """You are a strict audio compliance adjudicator.
Return only one JSON object with no markdown and no extra commentary.
The JSON object must match this schema:
{
  "content_status": "clear|unsafe|borderline",
  "privacy_status": "clear|contains_pii|borderline",
  "confidence": 0.0,
  "rationale": "short evidence-based explanation",
  "recommended_decision": "allow|review|quarantine|reject",
  "requires_manual_review": true,
  "final_reasons": []
}
If unsure, return borderline, review, and requires_manual_review=true.
"""

CONTENT_STATUSES = {"clear", "unsafe", "borderline"}
PRIVACY_STATUSES = {"clear", "contains_pii", "borderline"}
DECISIONS = {"allow", "review", "quarantine", "reject"}


class AdjudicateRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    temperature: float = 0.0


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def extract_json(text: str) -> dict[str, Any]:
    stripped = _strip_code_fence(text)
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", stripped):
        try:
            payload, _ = decoder.raw_decode(stripped[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload

    raise ValueError("No JSON object found in model output.")


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_judgement(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("judgement", payload)
    if not isinstance(raw, dict):
        raise ValueError("Judgement payload must be a JSON object.")

    content_status = str(raw.get("content_status", "borderline")).strip().lower()
    if content_status not in CONTENT_STATUSES:
        content_status = "borderline"

    privacy_status = str(raw.get("privacy_status", "borderline")).strip().lower()
    if privacy_status not in PRIVACY_STATUSES:
        privacy_status = "borderline"

    confidence = max(0.0, min(1.0, _float_or_default(raw.get("confidence"), 0.5)))

    recommended_decision = str(
        raw.get("recommended_decision", raw.get("recommended_disposition", "review"))
    ).strip().lower()
    if recommended_decision not in DECISIONS:
        recommended_decision = "review"

    requires_manual_review = raw.get("requires_manual_review", True)
    if isinstance(requires_manual_review, str):
        requires_manual_review = requires_manual_review.strip().lower() in {"1", "true", "yes", "y"}
    else:
        requires_manual_review = bool(requires_manual_review)

    final_reasons = raw.get("final_reasons", raw.get("final_findings", []))
    if isinstance(final_reasons, str):
        final_reasons = [final_reasons]
    if not isinstance(final_reasons, list):
        final_reasons = []

    return {
        "content_status": content_status,
        "privacy_status": privacy_status,
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "Qwen returned an incomplete hard-case judgement."),
        "recommended_decision": recommended_decision,
        "requires_manual_review": requires_manual_review,
        "final_reasons": [str(item) for item in final_reasons],
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "service": "qwen35-hardcase-vllm-adapter",
        "backend_url": VLLM_URL,
        "backend_model": BACKEND_MODEL,
    }


@app.post("/adjudicate")
async def adjudicate(request: AdjudicateRequest) -> dict[str, Any]:
    payload = {
        "model": BACKEND_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request.prompt},
        ],
        "temperature": request.temperature,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}

    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        response = await client.post(VLLM_URL, headers=headers, json=payload)

    response.raise_for_status()
    data = response.json()
    try:
        raw_output = data["choices"][0]["message"]["content"]
        judgement = normalize_judgement(extract_json(raw_output))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid JSON judgement from Qwen3.5: {exc}") from exc

    return {
        "provider": "qwen_endpoint",
        "model": request.model or DEFAULT_MODEL,
        "judgement": judgement,
        "raw_output": raw_output,
        "notes": [],
    }

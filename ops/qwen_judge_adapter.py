from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from text.models.schemas import HardCaseJudgement

app = FastAPI(title="Qwen Hard-case Judge Adapter")

VLLM_URL = os.getenv("QWEN_JUDGE_VLLM_URL", "http://127.0.0.1:8200/v1/chat/completions")
API_KEY = os.getenv("QWEN_JUDGE_API_KEY", "judge-token")
MODEL = os.getenv("QWEN_JUDGE_MODEL", "Qwen3.5-9B")
TIMEOUT_SECONDS = int(os.getenv("QWEN_JUDGE_TIMEOUT_SECONDS", "180"))
MAX_TOKENS = int(os.getenv("QWEN_JUDGE_MAX_TOKENS", "1024"))

CONTENT_STATUSES = {"clear", "unsafe", "borderline"}
PRIVACY_STATUSES = {"clear", "contains_pii", "borderline"}
DISPOSITIONS = {"P0", "P1", "P2", "P3", "P4", "P5"}

SYSTEM_PROMPT = """You are a strict text compliance adjudicator.
Return only one JSON object, with no markdown and no extra commentary.
The JSON object must match this schema:
{
  "content_status": "clear|unsafe|borderline",
  "privacy_status": "clear|contains_pii|borderline",
  "confidence": 0.0,
  "rationale": "short evidence-based explanation",
  "recommended_disposition": "P0|P1|P2|P3|P4|P5",
  "requires_manual_review": true,
  "final_findings": []
}
Use P4 for clearly unsafe content, P3 for unresolved hard cases or combined identity risk,
P1/P2 for redaction or restricted delivery, and P0 only when no material risk remains.
Do not invent spans or findings. If unsure, use borderline, P3, and requires_manual_review=true.
"""


class JudgeRequest(BaseModel):
    model: str = MODEL
    prompt: str
    temperature: float = 0


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


def normalize_judgement(payload: dict[str, Any]) -> HardCaseJudgement:
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

    recommended = str(raw.get("recommended_disposition", "P3")).strip().upper()
    if recommended not in DISPOSITIONS:
        recommended = "P3"

    requires_manual_review = raw.get("requires_manual_review", True)
    if isinstance(requires_manual_review, str):
        requires_manual_review = requires_manual_review.strip().lower() in {"1", "true", "yes", "y"}
    else:
        requires_manual_review = bool(requires_manual_review)

    final_findings = raw.get("final_findings", [])
    if not isinstance(final_findings, list):
        final_findings = []

    candidate = {
        "content_status": content_status,
        "privacy_status": privacy_status,
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "Qwen returned an incomplete hard-case judgement."),
        "recommended_disposition": recommended,
        "requires_manual_review": requires_manual_review,
        "final_findings": final_findings,
    }

    try:
        return HardCaseJudgement.model_validate(candidate)
    except ValidationError:
        candidate["final_findings"] = []
        return HardCaseJudgement.model_validate(candidate)


@app.post("/adjudicate")
async def adjudicate(request: JudgeRequest) -> dict[str, Any]:
    payload = {
        "model": request.model or MODEL,
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
        raw_judgement = extract_json(raw_output)
        judgement = normalize_judgement(raw_judgement)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid JSON judgement from Qwen judge: {exc}",
        ) from exc

    return {
        "provider": "qwen_judge",
        "model": request.model or MODEL,
        "judgement": judgement.model_dump(mode="json"),
        "raw_output": raw_output,
    }

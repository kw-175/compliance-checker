"""
Qwen hard-case adjudication adapter for audio transcript units.

The adapter handles prompt construction, endpoint/local model invocation, raw
JSON extraction, and conversion into AudioHardCaseJudgement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import (
    AudioHardCaseJudgement,
    Decision,
    PrivacyResult,
    SafetyResult,
    TranscriptUnit,
)


@dataclass
class HardCaseAdapterResult:
    judgement: AudioHardCaseJudgement | None = None
    provider_name: str = ""
    raw_response: str = ""
    notes: list[str] = field(default_factory=list)


def build_prompt(
    unit: TranscriptUnit,
    privacy: PrivacyResult | None,
    safety: SafetyResult | None,
    trigger_sources: list[str],
    trigger_reasons: list[str],
    settings: Settings,
) -> str:
    payload = {
        "task": "Resolve an uncertain audio-compliance transcript unit with a final structured judgement.",
        "model_requirement": settings.hard_case_model_name,
        "audio_transcript_unit": {
            "unit_id": unit.unit_id,
            "source_id": unit.source_id,
            "start_time": unit.start_time,
            "end_time": unit.end_time,
            "speaker_id": unit.speaker_id,
            "asr_confidence": unit.confidence,
            "asr_engine": unit.engine_name,
            "language": unit.language,
            "text_excerpt": unit.text[: settings.hard_case_max_chars],
        },
        "trigger_sources": trigger_sources,
        "trigger_reasons": trigger_reasons,
        "preliminary_privacy_result": privacy.model_dump(mode="json") if privacy else None,
        "preliminary_safety_result": safety.model_dump(mode="json") if safety else None,
        "output_schema": {
            "content_status": "clear|unsafe|borderline",
            "privacy_status": "clear|contains_pii|borderline",
            "confidence": "0.0-1.0",
            "rationale": "short explanation",
            "recommended_decision": "allow|review|quarantine|reject",
            "requires_manual_review": "boolean",
            "final_reasons": "list of short reason strings",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _coerce_decision(value: Any) -> Decision:
    raw = str(value or "").strip().lower()
    if raw in {"allow", "clear", "pass", "p0"}:
        return Decision.ALLOW
    if raw in {"quarantine", "hold", "isolate", "p4"}:
        return Decision.QUARANTINE
    if raw in {"reject", "block", "deny", "unsafe", "p5"}:
        return Decision.REJECT
    return Decision.REVIEW


def extract_json(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        for key in ("judgement", "judgment", "result", "data"):
            nested = raw.get(key)
            if isinstance(nested, dict):
                return nested
            if isinstance(nested, str):
                parsed = extract_json(nested)
                if parsed is not None:
                    return parsed
        if "choices" in raw and isinstance(raw["choices"], list) and raw["choices"]:
            choice = raw["choices"][0]
            if isinstance(choice, dict):
                message = choice.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return extract_json(message["content"])
                if isinstance(choice.get("text"), str):
                    return extract_json(choice["text"])
        return raw

    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def judgement_from_payload(raw: Any) -> AudioHardCaseJudgement | None:
    payload = extract_json(raw)
    if payload is None:
        return None

    payload = dict(payload)
    if "recommended_decision" not in payload and "recommended_disposition" in payload:
        payload["recommended_decision"] = payload["recommended_disposition"]
    payload["recommended_decision"] = _coerce_decision(payload.get("recommended_decision"))
    if isinstance(payload.get("final_reasons"), str):
        payload["final_reasons"] = [payload["final_reasons"]]
    try:
        confidence = float(payload.get("confidence", 0.0))
        payload["confidence"] = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        payload["confidence"] = 0.0
    return AudioHardCaseJudgement.model_validate(payload)


def call_endpoint(prompt: str, settings: Settings) -> tuple[AudioHardCaseJudgement | None, str]:
    if not settings.hard_case_endpoint:
        return None, ""

    import httpx

    response = httpx.post(
        settings.hard_case_endpoint,
        json={
            "model": settings.hard_case_model_name,
            "prompt": prompt,
            "temperature": 0,
        },
        timeout=settings.hard_case_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return judgement_from_payload(payload), json.dumps(payload, ensure_ascii=False)


@lru_cache(maxsize=2)
def load_local_model(model_path: str, requested_device: str):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    import torch

    device = requested_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
        )
    except Exception as exc:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype="auto",
        )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def call_local_model(prompt: str, settings: Settings) -> tuple[AudioHardCaseJudgement | None, str, str]:
    model_path = settings.hard_case_local_model_path
    if not model_path:
        return None, "", "local_model_path_not_configured"
    if Path(model_path).is_absolute() and not Path(model_path).exists():
        return None, "", f"local_model_path_missing: {model_path}"

    import torch

    model, tokenizer, device = load_local_model(model_path, settings.hard_case_device)
    messages = [
        {"role": "system", "content": "Return only valid JSON matching the requested schema."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted = prompt
    inputs = tokenizer(formatted, return_tensors="pt", truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.hard_case_max_new_tokens,
            do_sample=False,
        )
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return judgement_from_payload(raw), raw, ""


def adjudicate(prompt: str, settings: Settings) -> HardCaseAdapterResult:
    notes: list[str] = []
    try:
        judgement, raw_response = call_endpoint(prompt, settings)
        if judgement is not None:
            return HardCaseAdapterResult(
                judgement=judgement,
                provider_name="qwen_endpoint",
                raw_response=raw_response,
            )
    except Exception as exc:
        notes.append(f"endpoint_failed: {exc}")

    try:
        judgement, raw_response, note = call_local_model(prompt, settings)
        if note:
            notes.append(note)
        if judgement is not None:
            return HardCaseAdapterResult(
                judgement=judgement,
                provider_name="qwen_local",
                raw_response=raw_response,
                notes=notes,
            )
    except Exception as exc:
        notes.append(f"local_model_failed: {exc}")

    return HardCaseAdapterResult(notes=notes)

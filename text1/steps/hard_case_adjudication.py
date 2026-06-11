from __future__ import annotations

import json
import logging
from functools import lru_cache

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentSafetyResult,
    DetectionFinding,
    DispositionLevel,
    HardCaseAdjudicationResult,
    HardCaseJudgement,
    IngestUnit,
    PrivacyDetectionResult,
)

logger = logging.getLogger(__name__)


def _build_prompt(
    unit: IngestUnit,
    safety_result: ContentSafetyResult | None,
    privacy_result: PrivacyDetectionResult | None,
    settings: Settings,
) -> str:
    payload = {
        "task": "Resolve difficult text-compliance cases with a final structured judgement.",
        "model_requirement": settings.hard_case_model_name,
        "document": {
            "doc_id": unit.doc_id,
            "text_excerpt": unit.text[: settings.hard_case_max_chars],
            "metadata": {
                "task_id": unit.task_id,
                "tenant_id": unit.tenant_id,
                "profile_id": unit.profile_id,
                "source_type": unit.source_type,
            },
        },
        "preliminary_content_findings": [
            finding.model_dump(mode="json") for finding in (safety_result.findings if safety_result else [])
        ],
        "preliminary_privacy_findings": [
            finding.model_dump(mode="json") for finding in (privacy_result.findings if privacy_result else [])
        ],
        "output_schema": {
            "content_status": "clear|unsafe|borderline",
            "privacy_status": "clear|contains_pii|borderline",
            "confidence": "0.0-1.0",
            "rationale": "short explanation",
            "recommended_disposition": "P0|P1|P2|P3|P4|P5",
            "requires_manual_review": "boolean",
            "final_findings": "list of finding objects compatible with DetectionFinding",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


def _heuristic_judgement(
    safety_result: ContentSafetyResult | None,
    privacy_result: PrivacyDetectionResult | None,
) -> HardCaseJudgement:
    safety_findings = safety_result.findings if safety_result else []
    privacy_findings = privacy_result.findings if privacy_result else []
    final_findings: list[DetectionFinding] = list(safety_findings) + list(privacy_findings)

    content_status = "clear"
    privacy_status = "clear"
    confidence = 0.92
    recommended = DispositionLevel.P0
    requires_manual_review = False
    rationale_parts: list[str] = []

    if safety_result and safety_result.findings:
        if any(finding.severity.value == "critical" and not finding.needs_adjudication for finding in safety_findings):
            content_status = "unsafe"
            recommended = DispositionLevel.P4
            confidence = 0.88
            rationale_parts.append("High-severity content safety findings remain unsafe after preliminary screening.")
        else:
            content_status = "borderline"
            recommended = max(recommended, DispositionLevel.P3, key=lambda item: item.value)
            requires_manual_review = True
            confidence = 0.58
            rationale_parts.append("Content findings depend on surrounding context and need manual judgement.")

    if privacy_result and privacy_result.findings:
        if any(finding.risk_type == "combined_identity" for finding in privacy_findings):
            privacy_status = "borderline"
            recommended = DispositionLevel.P3 if recommended.value < DispositionLevel.P3.value else recommended
            requires_manual_review = True
            confidence = min(confidence, 0.57)
            rationale_parts.append("Multiple identity attributes form a combined re-identification risk.")
        elif privacy_findings:
            privacy_status = "contains_pii"
            if recommended in {DispositionLevel.P0, DispositionLevel.P1}:
                recommended = DispositionLevel.P1
            confidence = min(confidence, 0.76)
            rationale_parts.append("PII is present but can likely be handled through structured redaction.")

    if not rationale_parts:
        rationale_parts.append("Preliminary detectors did not surface a lasting compliance issue.")

    return HardCaseJudgement(
        content_status=content_status,
        privacy_status=privacy_status,
        confidence=round(confidence, 4),
        rationale=" ".join(rationale_parts),
        recommended_disposition=recommended,
        requires_manual_review=requires_manual_review,
        final_findings=final_findings,
    )


def _call_endpoint(prompt: str, settings: Settings) -> HardCaseJudgement | None:
    if not settings.hard_case_endpoint:
        return None

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
    raw = payload.get("judgement") if isinstance(payload, dict) else payload
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        return None
    return HardCaseJudgement.model_validate(raw)


@lru_cache(maxsize=1)
def _load_local_model(model_path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def _call_local_model(prompt: str, settings: Settings) -> HardCaseJudgement | None:
    if not settings.hard_case_local_model_path:
        return None

    import torch

    model, tokenizer = _load_local_model(settings.hard_case_local_model_path)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    payload = json.loads(raw)
    return HardCaseJudgement.model_validate(payload)


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    privacy_results: list[PrivacyDetectionResult],
    settings: Settings | None = None,
) -> list[HardCaseAdjudicationResult]:
    settings = settings or get_settings()
    if not settings.enable_hard_case_adjudication:
        return []

    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    results: list[HardCaseAdjudicationResult] = []

    for unit in ingest_units:
        safety_result = safety_by_doc.get(unit.doc_id)
        privacy_result = privacy_by_doc.get(unit.doc_id)
        trigger_sources: list[str] = []
        if safety_result and safety_result.needs_adjudication:
            trigger_sources.append("content_safety")
        if privacy_result and privacy_result.needs_adjudication:
            trigger_sources.append("privacy")
        if not trigger_sources:
            continue

        prompt = _build_prompt(unit, safety_result, privacy_result, settings)
        judgement: HardCaseJudgement | None = None
        provider_name = "heuristic_fallback"
        is_degraded = True
        notes: list[str] = []
        raw_response = ""

        try:
            judgement = _call_endpoint(prompt, settings)
            if judgement is not None:
                provider_name = "qwen_endpoint"
                is_degraded = False
        except Exception as exc:
            notes.append(f"endpoint_failed: {exc}")

        if judgement is None:
            try:
                judgement = _call_local_model(prompt, settings)
                if judgement is not None:
                    provider_name = "qwen_local"
                    is_degraded = False
            except Exception as exc:
                notes.append(f"local_model_failed: {exc}")

        if judgement is None:
            judgement = _heuristic_judgement(safety_result, privacy_result)
            raw_response = judgement.model_dump_json()
            notes.append("heuristic fallback used because no Qwen adjudicator was available.")

        results.append(
            HardCaseAdjudicationResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                trigger_sources=trigger_sources,
                model_name=settings.hard_case_model_name,
                provider_name=provider_name,
                prompt_version=settings.hard_case_prompt_version,
                adjudicated=True,
                is_degraded=is_degraded,
                uncertainty=round(1.0 - judgement.confidence, 4),
                judgement=judgement,
                raw_response=raw_response,
                notes=notes,
            )
        )

    logger.info("Hard-case adjudication completed: %d documents", len(results))
    return results

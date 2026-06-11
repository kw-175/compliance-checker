from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings
from text.models.schemas import DetectionFinding, IngestUnit
from text.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CONTEXTUAL_TYPES = {
    "education",
    "research",
    "news",
    "legal_case",
    "literary_or_fictional",
    "quotation",
    "safety_education",
    "public_interest",
}


def run(
    unit: IngestUnit,
    findings: list[DetectionFinding],
    settings: Settings,
    context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not settings.enable_content_safety_semantic_adjudication:
        return {}

    eligible_findings = [finding for finding in findings if _needs_semantic_adjudication(finding, context)]
    if not eligible_findings:
        return {}

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.api_content_safety_semantic_prompt_path))
    try:
        payload = client.complete_json(
            task_name="content_semantic_adjudication",
            system_prompt=system_prompt,
            payload=_payload(unit, eligible_findings, settings, context),
        )
        return _normalize_response(payload, eligible_findings)
    except (OpenAICompatibleAPIError, Exception) as exc:
        logger.warning("API content semantic adjudication failed for %s: %s", unit.doc_id, exc)
        return {
            finding.finding_id: {
                "finding_id": finding.finding_id,
                "semantic_decision": "semantic_unavailable",
                "context_type": _context_type(finding, context),
                "final_risk_level": "C2",
                "final_action": "P3",
                "final_training_eligibility": "T2",
                "final_dataset_route": "safety_review_or_eval_only",
                "allow_downstream_annotation": False,
                "requires_manual_review": True,
                "downgrade_allowed": True,
                "upgrade_required": False,
                "reasoning_summary": f"Semantic adjudication unavailable: {exc}",
                "confidence": 0.0,
                "is_degraded": True,
            }
            for finding in eligible_findings
        }


def _needs_semantic_adjudication(finding: DetectionFinding, context: dict[str, Any]) -> bool:
    attrs = finding.attributes.get("content_safety", {})
    api_payload = finding.attributes.get("api_payload", {})
    context_type = _context_type(finding, context)
    if bool(context.get("force_semantic_adjudication")):
        return True
    if bool(api_payload.get("semantic_adjudication_required")):
        return True
    if bool(attrs.get("semantic_adjudication_required")):
        return True
    return bool(context_type and context_type in CONTEXTUAL_TYPES)


def _context_type(finding: DetectionFinding, context: dict[str, Any]) -> str:
    attrs = finding.attributes.get("content_safety", {})
    return str(attrs.get("api_context_type") or context.get("context_type") or context.get("scene") or "").lower()


def _payload(
    unit: IngestUnit,
    findings: list[DetectionFinding],
    settings: Settings,
    context: dict[str, Any],
) -> dict[str, Any]:
    provider = resolve_provider_config(settings)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": unit.text[: provider.max_chars],
        "metadata": context,
        "custom_policy": settings.content_safety_custom_policy,
        "training_context": settings.content_safety_training_context,
        "adjudication_goal": (
            "Determine whether each candidate unsafe span is a true compliance violation in this "
            "scene, or a contextual/educational/news/quotation case that must be routed to review."
        ),
        "findings": [_finding_payload(finding) for finding in findings],
    }


def _finding_payload(finding: DetectionFinding) -> dict[str, Any]:
    attrs = finding.attributes.get("content_safety", {})
    span = finding.span
    return {
        "finding_id": finding.finding_id,
        "risk_type": finding.risk_type,
        "policy_tag": finding.policy_tag,
        "matched_label": attrs.get("matched_label", ""),
        "severity": finding.severity.value,
        "confidence": finding.confidence,
        "explanation": finding.explanation,
        "needs_adjudication": finding.needs_adjudication,
        "hard_case_reason": finding.hard_case_reason,
        "span": {
            "start": span.start,
            "end": span.end,
            "text": span.text,
        }
        if span
        else None,
        "api_context_type": attrs.get("api_context_type", ""),
        "api_context_rationale": attrs.get("api_context_rationale", ""),
        "api_recommended_risk_level": attrs.get("api_recommended_risk_level", ""),
        "api_recommended_action": attrs.get("api_recommended_action", ""),
        "api_recommended_training_eligibility": attrs.get("api_recommended_training_eligibility", ""),
        "api_recommended_dataset_route": attrs.get("api_recommended_dataset_route", ""),
    }


def _normalize_response(payload: dict[str, Any], findings: list[DetectionFinding]) -> dict[str, dict[str, Any]]:
    raw_items = payload.get("adjudications") or payload.get("findings") or payload.get("results") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []

    normalized: dict[str, dict[str, Any]] = {}
    by_index = list(findings)
    by_id = {finding.finding_id: finding for finding in findings}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        finding_id = str(item.get("finding_id") or "")
        if not finding_id and index < len(by_index):
            finding_id = by_index[index].finding_id
        if finding_id not in by_id:
            continue
        normalized[finding_id] = _adjudication_item(finding_id, item)
    return normalized


def _adjudication_item(finding_id: str, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "context_type": str(item.get("context_type") or item.get("semantic_context_type") or ""),
        "semantic_decision": str(item.get("semantic_decision") or item.get("decision") or "contextual_review"),
        "final_risk_level": _allowed_code(item.get("final_risk_level") or item.get("risk_level_code"), {"C0", "C1", "C2", "C3"}),
        "final_action": _allowed_code(item.get("final_action") or item.get("action"), {"P0", "P1", "P2", "P3", "P4", "P5"}),
        "final_training_eligibility": _allowed_code(
            item.get("final_training_eligibility") or item.get("training_eligibility"),
            {"T0", "T1", "T2", "T3"},
        ),
        "final_dataset_route": str(item.get("final_dataset_route") or item.get("dataset_route") or ""),
        "allow_downstream_annotation": item.get("allow_downstream_annotation"),
        "requires_manual_review": item.get("requires_manual_review"),
        "downgrade_allowed": bool(item.get("downgrade_allowed", False)),
        "upgrade_required": bool(item.get("upgrade_required", False)),
        "reasoning_summary": str(item.get("reasoning_summary") or item.get("rationale") or ""),
        "confidence": _safe_float(item.get("confidence"), 0.0),
    }


def _allowed_code(value: Any, allowed: set[str]) -> str:
    text = str(value or "").strip().upper()
    return text if text in allowed else ""


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default

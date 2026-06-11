from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DocumentContextRecord,
    IngestUnit,
    PrivacyDetectionResult,
    PrivacyFragmentAdjudicationRecord,
)
from text.prompt_loader import load_prompt
from text.steps.adjudication_payloads import compact_privacy_finding_payload, snippet_window

logger = logging.getLogger(__name__)


STRONG_PRIVACY_TYPES = {
    "address",
    "bank_card",
    "credential",
    "email",
    "guardian_contact",
    "health_record",
    "id_card",
    "medical_record",
    "minor_info",
    "parent_contact",
    "password",
    "phone",
    "phone_number",
    "secret",
    "student_id",
}

STRICT_EDUCATION_DOCUMENT_TYPES = {
    "grade_record",
    "home_school_communication",
    "student_record",
}

GENERALIZE_PRIVACY_TYPES = {
    "address",
    "education_record",
    "medical_record",
    "minor_info",
    "student_id",
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _conservative_privacy_action(risk_type: str) -> str:
    if risk_type == "combined_identity":
        return "manual_review"
    if risk_type in GENERALIZE_PRIVACY_TYPES:
        return "generalize"
    return "redact"


def _requires_privacy_floor(
    finding: Any,
    document_context: DocumentContextRecord | None,
) -> bool:
    risk_type = str(getattr(finding, "risk_type", "") or "")
    if risk_type in STRONG_PRIVACY_TYPES or risk_type == "combined_identity":
        return True
    if risk_type == "person_name" and document_context:
        return document_context.document_type in STRICT_EDUCATION_DOCUMENT_TYPES or document_context.contains_minor_context
    return False


def _apply_conservative_privacy_floor(
    record: PrivacyFragmentAdjudicationRecord,
    finding: Any,
    document_context: DocumentContextRecord | None,
) -> PrivacyFragmentAdjudicationRecord:
    if not _requires_privacy_floor(finding, document_context):
        return record
    risk_type = str(getattr(finding, "risk_type", "") or record.risk_type or "")
    action = _conservative_privacy_action(risk_type)
    explanation = record.explanation or getattr(finding, "explanation", "") or (
        f"This {risk_type} span is treated conservatively because it may expose student or personal privacy."
    )
    return record.model_copy(
        update={
            "fragment_truth": "real_pii" if risk_type != "combined_identity" else "uncertain",
            "governance_action": action,
            "can_keep": False,
            "requires_manual_review": bool(record.requires_manual_review or action == "manual_review"),
            "training_impact": record.training_impact or "强隐私片段不得以原始形式进入训练样本。",
            "annotation_impact": record.annotation_impact or "下游标注应使用脱敏或泛化后的文本。",
            "explanation": explanation,
            "metadata": {
                **dict(record.metadata or {}),
                "conservative_privacy_floor": True,
                "document_type": document_context.document_type if document_context else "",
            },
        }
    )


def _heuristic_adjudication(
    unit: IngestUnit,
    result: PrivacyDetectionResult,
    document_context: DocumentContextRecord | None,
) -> list[PrivacyFragmentAdjudicationRecord]:
    records: list[PrivacyFragmentAdjudicationRecord] = []
    for finding in result.findings:
        privacy_context = dict(finding.attributes.get("privacy_context", {}) or {})
        explanation = str(privacy_context.get("context_explanation") or finding.explanation or "")
        fragment_truth = "real_pii"
        action = "redact"
        can_keep = False
        manual_review = bool(finding.needs_adjudication)

        if finding.risk_type == "combined_identity":
            action = "manual_review"
            manual_review = True
            explanation = explanation or "Multiple identity attributes combine into a stronger re-identification risk."
        elif document_context and document_context.document_type == "textbook_example":
            if finding.risk_type == "person_name":
                fragment_truth = "contextual_example"
                action = "manual_review"
                manual_review = True
                explanation = explanation or "The name appears in a textbook-style example, so it needs review before retention."
        elif document_context and document_context.document_type in {"student_record", "grade_record", "home_school_communication"}:
            fragment_truth = "real_pii"
            action = "generalize" if finding.risk_type in {"address", "education_record"} else "redact"
            explanation = explanation or (
                f"This {finding.risk_type} appears in a likely real education record and should not remain in raw form."
            )
        else:
            explanation = explanation or (
                f"The span appears to be {finding.risk_type}; without a clear retention justification it remains in privacy governance."
            )

        records.append(
            _apply_conservative_privacy_floor(
                PrivacyFragmentAdjudicationRecord(
                    run_id=unit.run_id,
                    doc_id=unit.doc_id,
                    finding_id=finding.finding_id,
                    risk_type=finding.risk_type,
                    fragment_truth=fragment_truth,
                    governance_action=action,
                    can_keep=can_keep,
                    requires_manual_review=manual_review,
                    training_impact=privacy_context.get("training_impact") or "Raw privacy data should not directly enter training samples.",
                    annotation_impact=privacy_context.get("annotation_impact") or "Masked or generalized values preserve annotation flow better than raw exposure.",
                    explanation=explanation,
                    confidence=max(0.45, finding.confidence),
                    provider_name="heuristic_privacy_fragment_adjudicator",
                    provider_version="builtin-2026.05",
                    is_degraded=False,
                    metadata={"document_type": document_context.document_type if document_context else ""},
                ),
                finding,
                document_context,
            )
        )
    return records


def _degraded_review_record(
    unit: IngestUnit,
    finding,
    document_context: DocumentContextRecord | None,
    reason: str,
) -> PrivacyFragmentAdjudicationRecord:
    privacy_context = dict(finding.attributes.get("privacy_context", {}) or {})
    return PrivacyFragmentAdjudicationRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        finding_id=finding.finding_id,
        risk_type=finding.risk_type,
        fragment_truth="uncertain",
        governance_action="manual_review",
        can_keep=False,
        requires_manual_review=True,
        training_impact=privacy_context.get("training_impact") or "Qwen3.5 contextual adjudication did not complete, so this span cannot be cleared for training automatically.",
        annotation_impact=privacy_context.get("annotation_impact") or "The span should be reviewed before any downstream annotation flow uses it.",
        explanation=(
            f"Qwen3.5 contextual adjudication did not complete for this {finding.risk_type} span. "
            "The system kept the recalled privacy evidence and routed it to manual review instead of making a direct compliance judgement."
        ),
        confidence=0.0,
        provider_name="local_privacy_fragment_adjudicator",
        provider_version="degraded",
        is_degraded=True,
        metadata={
            "document_type": document_context.document_type if document_context else "",
            "degrade_reason": reason,
        },
    )


def _payload(
    unit: IngestUnit,
    finding,
    document_context: DocumentContextRecord | None,
    provider_max_chars: int,
) -> dict[str, Any]:
    span = finding.span
    text = unit.text[:provider_max_chars]
    if span is not None:
        text = snippet_window(unit.text, span.start, span.end, size=240)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": text,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "findings": [compact_privacy_finding_payload(finding)],
    }


def _normalize_records(
    unit: IngestUnit,
    result: PrivacyDetectionResult,
    payload: dict[str, Any],
    provider_name: str,
    provider_version: str,
    document_context: DocumentContextRecord | None = None,
) -> list[PrivacyFragmentAdjudicationRecord]:
    raw_items = payload.get("adjudications") or payload.get("findings") or payload.get("results") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []

    by_finding = {finding.finding_id: finding for finding in result.findings}
    normalized: list[PrivacyFragmentAdjudicationRecord] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        finding_id = str(item.get("finding_id") or "")
        finding = by_finding.get(finding_id)
        if finding is None:
            continue
        normalized.append(
            _apply_conservative_privacy_floor(
                PrivacyFragmentAdjudicationRecord(
                    run_id=unit.run_id,
                    doc_id=unit.doc_id,
                    finding_id=finding_id,
                    risk_type=finding.risk_type,
                    fragment_truth=str(item.get("fragment_truth") or "uncertain"),
                    governance_action=str(item.get("governance_action") or "manual_review"),
                    can_keep=bool(item.get("can_keep", False)),
                    requires_manual_review=bool(item.get("requires_manual_review", True)),
                    training_impact=str(item.get("training_impact") or ""),
                    annotation_impact=str(item.get("annotation_impact") or ""),
                    explanation=str(item.get("explanation") or finding.explanation),
                    confidence=_safe_float(item.get("confidence"), finding.confidence),
                    provider_name=provider_name,
                    provider_version=provider_version,
                    is_degraded=False,
                    metadata={"raw_payload": item},
                ),
                finding,
                document_context,
            )
        )
    return normalized


def run(
    ingest_units: list[IngestUnit],
    privacy_results: list[PrivacyDetectionResult],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> list[PrivacyFragmentAdjudicationRecord]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    context_by_doc = {item.doc_id: item for item in document_contexts}

    if provider.mode != "local_model":
        records: list[PrivacyFragmentAdjudicationRecord] = []
        for unit in ingest_units:
            result = privacy_by_doc.get(unit.doc_id)
            if result is None:
                continue
            records.extend(_heuristic_adjudication(unit, result, context_by_doc.get(unit.doc_id)))
        return records

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.local_privacy_fragment_prompt_path))
    all_records: list[PrivacyFragmentAdjudicationRecord] = []
    for unit in ingest_units:
        result = privacy_by_doc.get(unit.doc_id)
        if result is None or not result.findings:
            continue
        document_context = context_by_doc.get(unit.doc_id)
        by_finding = {finding.finding_id: finding for finding in result.findings}
        for finding in result.findings:
            try:
                payload = client.complete_json(
                    task_name="privacy_fragment_adjudication",
                    system_prompt=system_prompt,
                    payload=_payload(unit, finding, document_context, provider.max_chars),
                )
                records = _normalize_records(
                    unit,
                    result.model_copy(update={"findings": [finding]}),
                    payload,
                    provider_name="local_privacy_fragment_adjudicator",
                    provider_version=provider.model,
                    document_context=document_context,
                )
                if not records:
                    records = [_degraded_review_record(unit, finding, document_context, "empty_local_adjudication")]
                all_records.extend(records)
            except Exception as exc:
                logger.warning("Local privacy fragment adjudication failed for %s/%s: %s", unit.doc_id, finding.finding_id, exc)
                all_records.append(_degraded_review_record(unit, by_finding[finding.finding_id], document_context, str(exc)))
    return all_records

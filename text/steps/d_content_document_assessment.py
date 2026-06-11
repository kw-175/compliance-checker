from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentDocumentAssessmentRecord,
    ContentFragmentAdjudicationRecord,
    ContentSafetyResult,
    DocumentContextRecord,
    IngestUnit,
)
from text.prompt_loader import load_prompt
from text.steps.adjudication_payloads import summarize_content_findings, summarize_fragment_adjudications

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _heuristic_assessment(
    unit: IngestUnit,
    result: ContentSafetyResult | None,
    fragments: list[ContentFragmentAdjudicationRecord],
    document_context: DocumentContextRecord | None,
) -> ContentDocumentAssessmentRecord:
    if not result or not result.findings:
        return ContentDocumentAssessmentRecord(
            run_id=unit.run_id,
            doc_id=unit.doc_id,
            text_hash=unit.text_hash,
            overall_stance="discussion",
            operational_risk="low",
            training_suitability="allowed",
            annotation_suitability="allowed",
            recommended_action="keep",
            requires_manual_review=False,
            explanation="No content-safety risk remained after recall.",
            confidence=0.84,
            provider_name="heuristic_content_document_assessor",
            provider_version="builtin-2026.05",
        )

    actions = {item.recommended_action for item in fragments}
    roles = {item.semantic_role for item in fragments}
    protective = any(item.protective_context for item in fragments)
    if "reject" in actions:
        action = "reject"
        stance = "actionable_guidance" if "actionable_guidance" in roles else "propagating_risk"
        operational_risk = "high"
        training = "blocked"
        annotation = "blocked"
    elif "manual_review" in actions or protective:
        action = "manual_review"
        stance = "teaching" if protective else "uncertain"
        operational_risk = "medium"
        training = "restricted"
        annotation = "restricted"
    else:
        action = "restricted_review"
        stance = next(iter(roles or {"discussion"}))
        operational_risk = "medium"
        training = "restricted"
        annotation = "restricted"

    explanation = (
        f"The document contains {len(result.findings)} content-safety findings. "
        f"The overall stance is assessed as {stance}, so recommended action is {action}."
    )
    return ContentDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_stance=stance,
        operational_risk=operational_risk,
        training_suitability=training,
        annotation_suitability=annotation,
        recommended_action=action,
        requires_manual_review=action != "reject" and action != "keep",
        explanation=explanation,
        confidence=0.74,
        provider_name="heuristic_content_document_assessor",
        provider_version="builtin-2026.05",
        metadata={"document_type": document_context.document_type if document_context else ""},
    )


def _clear_assessment(
    unit: IngestUnit,
    document_context: DocumentContextRecord | None,
    provider_name: str,
    provider_version: str,
) -> ContentDocumentAssessmentRecord:
    return ContentDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_stance="clear",
        operational_risk="low",
        training_suitability="allowed",
        annotation_suitability="allowed",
        recommended_action="keep",
        requires_manual_review=False,
        explanation="No content-safety candidate windows or localized risk fragments were produced. Any privacy-only risk is handled by the privacy compliance chain.",
        confidence=0.92,
        provider_name=provider_name,
        provider_version=provider_version,
        metadata={
            "document_type": document_context.document_type if document_context else "",
            "risk_chain_scope": "content_safety_only",
            "short_circuit_reason": "no_content_safety_evidence",
            "can_raise_disposition": False,
        },
    )


def _payload(
    unit: IngestUnit,
    result: ContentSafetyResult | None,
    fragments: list[ContentFragmentAdjudicationRecord],
    document_context: DocumentContextRecord | None,
    provider_max_chars: int,
) -> dict[str, Any]:
    content_findings = list(result.findings if result else [])
    context_payload = (
        {
            "document_type": document_context.document_type,
            "scene_type": document_context.scene_type,
            "subject_type": document_context.subject_type,
            "summary": document_context.summary[:240],
            "explanation": document_context.explanation[:320],
        }
        if document_context
        else {}
    )
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "document_context": context_payload,
        "content_findings": summarize_content_findings(content_findings, limit=10),
        "fragment_adjudications": summarize_fragment_adjudications(fragments, limit=12),
        "finding_count": len(content_findings),
        "high_risk_count": sum(
            1
            for finding in content_findings
            if getattr(getattr(finding, "severity", None), "value", "") in {"high", "critical"}
        ),
        "protective_fragment_count": sum(1 for record in fragments if getattr(record, "protective_context", False)),
        "scope_constraints": {
            "judge_only_content_safety_risks": True,
            "ignore_privacy_only_or_pii_risks": True,
            "if_no_content_evidence_return_keep": True,
        },
    }


def _normalize_record(
    unit: IngestUnit,
    payload: dict[str, Any],
    provider_name: str,
    provider_version: str,
) -> ContentDocumentAssessmentRecord:
    return ContentDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_stance=str(payload.get("overall_stance") or "uncertain"),
        operational_risk=str(payload.get("operational_risk") or "medium"),
        training_suitability=str(payload.get("training_suitability") or "restricted"),
        annotation_suitability=str(payload.get("annotation_suitability") or "restricted"),
        recommended_action=str(payload.get("recommended_action") or "manual_review"),
        requires_manual_review=bool(payload.get("requires_manual_review", True)),
        explanation=str(payload.get("explanation") or "Content document assessment completed."),
        confidence=_safe_float(payload.get("confidence"), 0.65),
        provider_name=provider_name,
        provider_version=provider_version,
        is_degraded=False,
        metadata={"raw_payload": payload},
    )


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    fragment_adjudications: list[ContentFragmentAdjudicationRecord],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> list[ContentDocumentAssessmentRecord]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    safety_by_doc = {result.doc_id: result for result in safety_results}
    context_by_doc = {item.doc_id: item for item in document_contexts}
    fragments_by_doc: dict[str, list[ContentFragmentAdjudicationRecord]] = {}
    for record in fragment_adjudications:
        fragments_by_doc.setdefault(record.doc_id, []).append(record)

    if provider.mode != "local_model":
        return [
            _heuristic_assessment(unit, safety_by_doc.get(unit.doc_id), fragments_by_doc.get(unit.doc_id, []), context_by_doc.get(unit.doc_id))
            for unit in ingest_units
        ]

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.local_content_document_prompt_path))
    assessments: list[ContentDocumentAssessmentRecord] = []
    for unit in ingest_units:
        result = safety_by_doc.get(unit.doc_id)
        fragments = fragments_by_doc.get(unit.doc_id, [])
        document_context = context_by_doc.get(unit.doc_id)
        if not result or not result.findings or not fragments:
            assessments.append(
                _clear_assessment(
                    unit,
                    document_context,
                    provider_name="content_document_scope_guard",
                    provider_version="builtin-2026.05",
                )
            )
            continue
        try:
            payload = client.complete_json(
                task_name="content_document_assessment",
                system_prompt=system_prompt,
                payload=_payload(unit, result, fragments, document_context, provider.max_chars),
            )
            assessments.append(
                _normalize_record(
                    unit,
                    payload,
                    provider_name="local_content_document_assessor",
                    provider_version=provider.model,
                )
            )
        except Exception as exc:
            logger.warning("Local content document assessment failed for %s: %s", unit.doc_id, exc)
            record = _heuristic_assessment(unit, result, fragments, document_context)
            record.is_degraded = True
            record.metadata["degrade_reason"] = str(exc)
            assessments.append(record)
    return assessments

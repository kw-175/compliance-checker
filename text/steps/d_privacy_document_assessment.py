from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DocumentContextRecord,
    IngestUnit,
    PrivacyDetectionResult,
    PrivacyDocumentAssessmentRecord,
    PrivacyFragmentAdjudicationRecord,
)
from text.prompt_loader import load_prompt
from text.steps.adjudication_payloads import compact_fragment_adjudication_payload

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _heuristic_assessment(
    unit: IngestUnit,
    result: PrivacyDetectionResult | None,
    fragments: list[PrivacyFragmentAdjudicationRecord],
    document_context: DocumentContextRecord | None,
) -> PrivacyDocumentAssessmentRecord:
    if not result or not result.findings:
        return PrivacyDocumentAssessmentRecord(
            run_id=unit.run_id,
            doc_id=unit.doc_id,
            text_hash=unit.text_hash,
            overall_risk_level="low",
            combination_risk=False,
            training_suitability="allowed",
            annotation_suitability="allowed",
            recommended_action="keep",
            requires_manual_review=False,
            explanation="No material privacy risk remained after recall and adjudication.",
            confidence=0.85,
            provider_name="heuristic_privacy_document_assessor",
            provider_version="builtin-2026.05",
        )

    actions = {item.governance_action for item in fragments}
    combination_risk = any(finding.risk_type == "combined_identity" for finding in result.findings)
    if combination_risk or "exclude_from_training" in actions:
        recommended_action = "exclude_from_training"
        overall_risk_level = "high"
        training = "blocked"
        annotation = "restricted"
    elif "manual_review" in actions:
        recommended_action = "manual_review"
        overall_risk_level = "medium"
        training = "restricted"
        annotation = "restricted"
    elif "generalize" in actions:
        recommended_action = "generalize"
        overall_risk_level = "medium"
        training = "restricted"
        annotation = "allowed"
    else:
        recommended_action = "redact"
        overall_risk_level = "medium" if document_context and document_context.contains_minor_context else "low"
        training = "restricted" if overall_risk_level == "medium" else "allowed"
        annotation = "allowed"

    explanation = (
        f"The document contains {len(result.findings)} privacy findings. "
        f"{'It forms a stronger re-identification profile. ' if combination_risk else ''}"
        f"Recommended action is {recommended_action} for training governance."
    )
    return PrivacyDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_risk_level=overall_risk_level,
        combination_risk=combination_risk,
        training_suitability=training,
        annotation_suitability=annotation,
        recommended_action=recommended_action,
        requires_manual_review=recommended_action == "manual_review",
        explanation=explanation,
        confidence=0.72,
        provider_name="heuristic_privacy_document_assessor",
        provider_version="builtin-2026.05",
        metadata={"document_type": document_context.document_type if document_context else ""},
    )


def _clear_assessment(
    unit: IngestUnit,
    document_context: DocumentContextRecord | None,
    provider_name: str,
    provider_version: str,
) -> PrivacyDocumentAssessmentRecord:
    return PrivacyDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_risk_level="low",
        combination_risk=False,
        training_suitability="allowed",
        annotation_suitability="allowed",
        recommended_action="keep",
        requires_manual_review=False,
        explanation="No privacy findings were produced. Any content-safety risk is handled by the content safety compliance chain.",
        confidence=0.92,
        provider_name=provider_name,
        provider_version=provider_version,
        metadata={
            "document_type": document_context.document_type if document_context else "",
            "risk_chain_scope": "privacy_only",
            "short_circuit_reason": "no_privacy_evidence",
            "can_raise_disposition": False,
        },
    )


def _payload(
    unit: IngestUnit,
    result: PrivacyDetectionResult | None,
    fragments: list[PrivacyFragmentAdjudicationRecord],
    document_context: DocumentContextRecord | None,
    provider_max_chars: int,
) -> dict[str, Any]:
    findings = list(result.findings if result else [])
    risk_type_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for finding in findings:
        risk_type_counts[str(getattr(finding, "risk_type", "") or "unknown")] = (
            risk_type_counts.get(str(getattr(finding, "risk_type", "") or "unknown"), 0) + 1
        )
        severity = str(getattr(getattr(finding, "severity", None), "value", "") or "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    context_payload = (
        {
            "document_type": document_context.document_type,
            "scene_type": document_context.scene_type,
            "subject_type": document_context.subject_type,
            "contains_education_context": document_context.contains_education_context,
            "contains_minor_context": document_context.contains_minor_context,
            "summary": document_context.summary[:240],
            "explanation": document_context.explanation[:320],
        }
        if document_context
        else {}
    )
    fragment_summary = [
        {
            key: value
            for key, value in compact_fragment_adjudication_payload(item).items()
            if key
            in {
                "finding_id",
                "risk_type",
                "governance_action",
                "requires_manual_review",
                "training_impact",
                "explanation",
                "confidence",
            }
        }
        for item in fragments[:8]
    ]
    finding_summary = []
    for finding in findings[:8]:
        span = getattr(finding, "span", None)
        finding_summary.append(
            {
                "finding_id": getattr(finding, "finding_id", ""),
                "risk_type": getattr(finding, "risk_type", ""),
                "policy_tag": getattr(finding, "policy_tag", ""),
                "severity": getattr(getattr(finding, "severity", None), "value", str(getattr(finding, "severity", ""))),
                "confidence": getattr(finding, "confidence", 0.0),
                "text": (span.text[:80] if span and span.text else ""),
            }
        )
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text_excerpt": unit.text[: min(provider_max_chars, 900)],
        "document_context": context_payload,
        "privacy_findings": finding_summary,
        "fragment_adjudications": fragment_summary,
        "finding_count": len(findings),
        "risk_type_counts": risk_type_counts,
        "severity_counts": severity_counts,
        "high_risk_count": sum(
            1
            for finding in findings
            if getattr(getattr(finding, "severity", None), "value", "") in {"high", "critical"}
        ),
        "has_combined_identity": any(getattr(finding, "risk_type", "") == "combined_identity" for finding in findings),
        "scope_constraints": {
            "judge_only_privacy_or_reidentification_risks": True,
            "ignore_content_safety_only_risks": True,
            "if_no_privacy_evidence_return_keep": True,
        },
    }


def _normalize_record(
    unit: IngestUnit,
    payload: dict[str, Any],
    provider_name: str,
    provider_version: str,
) -> PrivacyDocumentAssessmentRecord:
    return PrivacyDocumentAssessmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        overall_risk_level=str(payload.get("overall_risk_level") or "medium"),
        combination_risk=bool(payload.get("combination_risk", False)),
        training_suitability=str(payload.get("training_suitability") or "restricted"),
        annotation_suitability=str(payload.get("annotation_suitability") or "restricted"),
        recommended_action=str(payload.get("recommended_action") or "manual_review"),
        requires_manual_review=bool(payload.get("requires_manual_review", True)),
        explanation=str(payload.get("explanation") or "Privacy document assessment completed."),
        confidence=_safe_float(payload.get("confidence"), 0.65),
        provider_name=provider_name,
        provider_version=provider_version,
        is_degraded=False,
        metadata={"raw_payload": payload},
    )


def run(
    ingest_units: list[IngestUnit],
    privacy_results: list[PrivacyDetectionResult],
    fragment_adjudications: list[PrivacyFragmentAdjudicationRecord],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> list[PrivacyDocumentAssessmentRecord]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    context_by_doc = {item.doc_id: item for item in document_contexts}
    fragments_by_doc: dict[str, list[PrivacyFragmentAdjudicationRecord]] = {}
    for record in fragment_adjudications:
        fragments_by_doc.setdefault(record.doc_id, []).append(record)

    if provider.mode != "local_model":
        return [
            _heuristic_assessment(unit, privacy_by_doc.get(unit.doc_id), fragments_by_doc.get(unit.doc_id, []), context_by_doc.get(unit.doc_id))
            for unit in ingest_units
        ]

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.local_privacy_document_prompt_path))
    assessments: list[PrivacyDocumentAssessmentRecord] = []
    for unit in ingest_units:
        result = privacy_by_doc.get(unit.doc_id)
        fragments = fragments_by_doc.get(unit.doc_id, [])
        document_context = context_by_doc.get(unit.doc_id)
        if not result or not result.findings:
            assessments.append(
                _clear_assessment(
                    unit,
                    document_context,
                    provider_name="privacy_document_scope_guard",
                    provider_version="builtin-2026.05",
                )
            )
            continue
        try:
            payload = client.complete_json(
                task_name="privacy_document_assessment",
                system_prompt=system_prompt,
                payload=_payload(unit, result, fragments, document_context, provider.max_chars),
            )
            assessments.append(
                _normalize_record(
                    unit,
                    payload,
                    provider_name="local_privacy_document_assessor",
                    provider_version=provider.model,
                )
            )
        except Exception as exc:
            logger.warning("Local privacy document assessment failed for %s: %s", unit.doc_id, exc)
            record = _heuristic_assessment(unit, result, fragments, document_context)
            record.is_degraded = True
            record.metadata["degrade_reason"] = str(exc)
            assessments.append(record)
    return assessments

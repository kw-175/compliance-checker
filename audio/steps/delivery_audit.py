"""
Delivery and audit packaging for the active audio privacy/safety workflow.
"""

from __future__ import annotations

import logging

from audio.models.schemas import (
    AudioAnnotationRecord,
    AudioAuditRecord,
    AudioHardCaseResult,
    Decision,
    DeliveryStatus,
    EvidenceBundle,
    PolicyDecision,
    PrivacyResult,
    RedactionSpan,
    SafetyResult,
    TranscriptUnit,
    UnitDecision,
)

logger = logging.getLogger(__name__)


def _delivery_status(decision: Decision) -> DeliveryStatus:
    if decision == Decision.REJECT:
        return DeliveryStatus.BLOCK
    if decision in {Decision.REVIEW, Decision.QUARANTINE}:
        return DeliveryStatus.HOLD
    return DeliveryStatus.DELIVER


def _review_priority(decision: Decision) -> str:
    if decision == Decision.REJECT:
        return "critical"
    if decision == Decision.QUARANTINE:
        return "high"
    if decision == Decision.REVIEW:
        return "normal"
    return "low"


def _spans_by_unit(spans: list[RedactionSpan]) -> dict[str, list[RedactionSpan]]:
    grouped: dict[str, list[RedactionSpan]] = {}
    for span in spans:
        grouped.setdefault(span.unit_id, []).append(span)
    return grouped


def run(
    transcript_units: list[TranscriptUnit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    redaction_spans: list[RedactionSpan],
    evidence_bundle: EvidenceBundle,
    decision: PolicyDecision,
    hard_case_results: list[AudioHardCaseResult] | None = None,
) -> tuple[list[AudioAnnotationRecord], list[AudioAuditRecord]]:
    privacy_by_unit = {item.unit_id: item for item in privacy_results}
    safety_by_unit = {item.unit_id: item for item in safety_results}
    hard_case_by_unit = {item.unit_id: item for item in (hard_case_results or [])}
    evidence_by_unit = {item.unit_id: item for item in evidence_bundle.transcript_units}
    decision_by_unit: dict[str, UnitDecision] = {item.unit_id: item for item in decision.unit_decisions}
    redaction_spans_by_unit = _spans_by_unit(redaction_spans)

    annotation_records: list[AudioAnnotationRecord] = []
    audit_records: list[AudioAuditRecord] = []

    for unit in transcript_units:
        unit_decision = decision_by_unit.get(unit.unit_id, UnitDecision(unit_id=unit.unit_id))
        privacy = privacy_by_unit.get(unit.unit_id)
        safety = safety_by_unit.get(unit.unit_id)
        hard_case = hard_case_by_unit.get(unit.unit_id)
        spans = redaction_spans_by_unit.get(unit.unit_id, [])
        hints: list[str] = []
        if privacy and privacy.pii_count:
            hints.append(privacy.provider_name or "privacy detector")
        if safety and safety.safety_level.value != "safe":
            hints.append(safety.raw_output or f"safety={safety.safety_level.value}")
        if hard_case:
            hints.append(f"hard_case={hard_case.judgement.recommended_decision.value}")

        annotation_records.append(
            AudioAnnotationRecord(
                run_id=evidence_bundle.pipeline_run_id,
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                original_text=unit.text,
                redacted_view=privacy.redacted_text if privacy and privacy.redacted_text else unit.text,
                delivery_status=_delivery_status(unit_decision.decision),
                decision=unit_decision.decision,
                review_priority=_review_priority(unit_decision.decision),
                start_time=unit.start_time,
                end_time=unit.end_time,
                speaker_id=unit.speaker_id,
                redaction_spans=spans,
                annotation_hints=hints[:5],
                metadata={
                    "language": unit.language,
                    "engine_name": unit.engine_name,
                    "confidence": unit.confidence,
                },
            )
        )

        audit_records.append(
            AudioAuditRecord(
                run_id=evidence_bundle.pipeline_run_id,
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                transcript=unit,
                privacy_result=privacy,
                safety_result=safety,
                hard_case_result=hard_case,
                redaction_spans=spans,
                evidence=evidence_by_unit.get(unit.unit_id),
                decision=unit_decision,
                provider_manifest={
                    "privacy": privacy.provider_name if privacy else "",
                    "content_safety": safety.provider_name if safety else "",
                    "hard_case": hard_case.provider_name if hard_case else "",
                    "asr": unit.engine_name,
                },
                trust_level=evidence_bundle.trust_level,
                audit_summary=f"{unit_decision.decision.value} with {len(spans)} redaction span(s).",
                metadata={
                    "source_id": unit.source_id,
                    "start_time": unit.start_time,
                    "end_time": unit.end_time,
                    "speaker_id": unit.speaker_id,
                },
            )
        )

    logger.info(
        "Audio delivery and audit packaging completed: %d annotation records, %d audit records",
        len(annotation_records),
        len(audit_records),
    )
    return annotation_records, audit_records

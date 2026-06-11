from __future__ import annotations

import logging

from common.enums import TrustLevel
from text.models.schemas import (
    AnnotationPackageRecord,
    AuditPackageRecord,
    ContentSafetyResult,
    DeliveryStatus,
    DispositionLevel,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PolicyDecisionRecord,
    PrivacyDetectionResult,
    SpanConflictResolutionResult,
)

logger = logging.getLogger(__name__)


def _apply_redactions(text: str, decision: PolicyDecisionRecord) -> str:
    result = text
    for target in sorted(decision.redaction_targets, key=lambda item: item.start, reverse=True):
        result = result[: target.start] + (target.replacement or "<REDACTED>") + result[target.end :]
    return result


def _delivery_status(disposition: DispositionLevel) -> DeliveryStatus:
    if disposition in {DispositionLevel.P4, DispositionLevel.P5}:
        return DeliveryStatus.BLOCK
    if disposition in {DispositionLevel.P2, DispositionLevel.P3}:
        return DeliveryStatus.HOLD
    return DeliveryStatus.DELIVER


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    privacy_results: list[PrivacyDetectionResult],
    redaction_plans: list[SpanConflictResolutionResult],
    adjudications: list[HardCaseAdjudicationResult],
    events: list[EvidenceEvent],
    decisions: list[PolicyDecisionRecord],
) -> tuple[list[AnnotationPackageRecord], list[AuditPackageRecord]]:
    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    redaction_plan_by_doc = {result.doc_id: result for result in redaction_plans}
    adjudication_by_doc = {result.doc_id: result for result in adjudications}
    events_by_doc: dict[str, list[EvidenceEvent]] = {}
    for event in events:
        events_by_doc.setdefault(event.doc_id, []).append(event)
    decisions_by_doc = {decision.doc_id: decision for decision in decisions}

    annotation_records: list[AnnotationPackageRecord] = []
    audit_records: list[AuditPackageRecord] = []

    for unit in ingest_units:
        decision = decisions_by_doc[unit.doc_id]
        delivery_status = _delivery_status(decision.disposition_level)
        redacted_view = _apply_redactions(unit.text, decision)
        doc_events = events_by_doc.get(unit.doc_id, [])
        adjudication = adjudication_by_doc.get(unit.doc_id)
        redaction_plan = redaction_plan_by_doc.get(unit.doc_id)
        degrade_reasons = list(adjudication.notes) if adjudication and adjudication.is_degraded else []

        annotation_records.append(
            AnnotationPackageRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                original_text=unit.text,
                redacted_view=redacted_view,
                delivery_status=delivery_status,
                disposition_level=decision.disposition_level,
                unified_decision=decision.unified_decision,
                review_priority=decision.review_priority,
                span_annotations=decision.redaction_targets,
                evidence_event_ids=decision.evidence_event_ids,
                annotation_hints=[event.explanation for event in doc_events[:5]],
                metadata={
                    "task_id": unit.task_id,
                    "tenant_id": unit.tenant_id,
                    "profile_id": unit.profile_id,
                    "source_path": unit.source_path,
                    "redaction_plan_provider": redaction_plan.provider_name if redaction_plan else "",
                    "redaction_conflict_count": len(redaction_plan.conflicts) if redaction_plan else 0,
                },
            )
        )

        audit_records.append(
            AuditPackageRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                ingest_unit=unit,
                safety_result=safety_by_doc.get(unit.doc_id),
                privacy_result=privacy_by_doc.get(unit.doc_id),
                redaction_plan=redaction_plan,
                hard_case_result=adjudication,
                evidence_events=doc_events,
                decision=decision,
                provider_manifest={
                    "content_safety": safety_by_doc.get(unit.doc_id).provider_name if safety_by_doc.get(unit.doc_id) else "",
                    "privacy": privacy_by_doc.get(unit.doc_id).provider_name if privacy_by_doc.get(unit.doc_id) else "",
                    "redaction_plan": redaction_plan.provider_name if redaction_plan else "",
                    "hard_case": adjudication.provider_name if adjudication else "",
                },
                trust_level=TrustLevel.DEGRADED if degrade_reasons else decision.trust_level,
                degrade_reasons=degrade_reasons,
                audit_summary=f"{decision.disposition_level.value} / {decision.unified_decision.value} with {len(doc_events)} evidence events.",
            )
        )

    logger.info(
        "Delivery and audit packaging completed: %d annotation records, %d audit records",
        len(annotation_records),
        len(audit_records),
    )
    return annotation_records, audit_records

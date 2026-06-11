from __future__ import annotations

import logging

from common.enums import TrustLevel
from text.models.schemas import (
    AnnotationPackageRecord,
    AuditPackageRecord,
    ContentCandidateWindowRecord,
    ContentDocumentAssessmentRecord,
    ContentFragmentAdjudicationRecord,
    ContentLocalizedFragmentRecord,
    ContentSafetyResult,
    DeliveryStatus,
    DocumentContextRecord,
    DispositionLevel,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PolicyDecisionRecord,
    PrivacyDocumentAssessmentRecord,
    PrivacyFragmentAdjudicationRecord,
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
    document_contexts: list[DocumentContextRecord] | None = None,
    content_candidate_windows: list[ContentCandidateWindowRecord] | None = None,
    content_localized_fragments: list[ContentLocalizedFragmentRecord] | None = None,
    privacy_fragment_adjudications: list[PrivacyFragmentAdjudicationRecord] | None = None,
    content_fragment_adjudications: list[ContentFragmentAdjudicationRecord] | None = None,
    privacy_document_assessments: list[PrivacyDocumentAssessmentRecord] | None = None,
    content_document_assessments: list[ContentDocumentAssessmentRecord] | None = None,
) -> tuple[list[AnnotationPackageRecord], list[AuditPackageRecord]]:
    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    document_context_by_doc = {result.doc_id: result for result in (document_contexts or [])}
    redaction_plan_by_doc = {result.doc_id: result for result in redaction_plans}
    adjudication_by_doc = {result.doc_id: result for result in adjudications}
    privacy_doc_assessment_by_doc = {result.doc_id: result for result in (privacy_document_assessments or [])}
    content_doc_assessment_by_doc = {result.doc_id: result for result in (content_document_assessments or [])}
    events_by_doc: dict[str, list[EvidenceEvent]] = {}
    for event in events:
        events_by_doc.setdefault(event.doc_id, []).append(event)
    candidate_windows_by_doc: dict[str, list[ContentCandidateWindowRecord]] = {}
    for item in content_candidate_windows or []:
        candidate_windows_by_doc.setdefault(item.doc_id, []).append(item)
    localized_fragments_by_doc: dict[str, list[ContentLocalizedFragmentRecord]] = {}
    for item in content_localized_fragments or []:
        localized_fragments_by_doc.setdefault(item.doc_id, []).append(item)
    privacy_fragments_by_doc: dict[str, list[PrivacyFragmentAdjudicationRecord]] = {}
    for item in privacy_fragment_adjudications or []:
        privacy_fragments_by_doc.setdefault(item.doc_id, []).append(item)
    content_fragments_by_doc: dict[str, list[ContentFragmentAdjudicationRecord]] = {}
    for item in content_fragment_adjudications or []:
        content_fragments_by_doc.setdefault(item.doc_id, []).append(item)
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
                document_context=document_context_by_doc.get(unit.doc_id),
                redaction_plan=redaction_plan,
                hard_case_result=adjudication,
                content_candidate_windows=candidate_windows_by_doc.get(unit.doc_id, []),
                content_localized_fragments=localized_fragments_by_doc.get(unit.doc_id, []),
                privacy_fragment_adjudications=privacy_fragments_by_doc.get(unit.doc_id, []),
                content_fragment_adjudications=content_fragments_by_doc.get(unit.doc_id, []),
                privacy_document_assessment=privacy_doc_assessment_by_doc.get(unit.doc_id),
                content_document_assessment=content_doc_assessment_by_doc.get(unit.doc_id),
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

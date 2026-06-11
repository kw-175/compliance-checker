from __future__ import annotations

import logging

from common.enums import TrustLevel, UnifiedDecision
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentDocumentAssessmentRecord,
    ContentCandidateWindowRecord,
    ContentFragmentAdjudicationRecord,
    ContentLocalizedFragmentRecord,
    DocumentContextRecord,
    DispositionLevel,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PolicyDecisionRecord,
    PrivacyDocumentAssessmentRecord,
    PrivacyFragmentAdjudicationRecord,
    RedactionTarget,
    SpanConflictResolutionResult,
)

logger = logging.getLogger(__name__)

DISPOSITION_PRIORITY = {
    DispositionLevel.P0: 0,
    DispositionLevel.P1: 1,
    DispositionLevel.P2: 2,
    DispositionLevel.P3: 3,
    DispositionLevel.P4: 4,
    DispositionLevel.P5: 5,
}
DISPOSITION_RISK = {
    DispositionLevel.P0: 0.05,
    DispositionLevel.P1: 0.30,
    DispositionLevel.P2: 0.55,
    DispositionLevel.P3: 0.72,
    DispositionLevel.P4: 0.92,
    DispositionLevel.P5: 1.00,
}
DISPOSITION_TO_DECISION = {
    DispositionLevel.P0: UnifiedDecision.ALLOW,
    DispositionLevel.P1: UnifiedDecision.ALLOW,
    DispositionLevel.P2: UnifiedDecision.QUARANTINE,
    DispositionLevel.P3: UnifiedDecision.REVIEW,
    DispositionLevel.P4: UnifiedDecision.REJECT,
    DispositionLevel.P5: UnifiedDecision.REJECT,
}


def _max_disposition(current: DispositionLevel, candidate: DispositionLevel) -> DispositionLevel:
    if DISPOSITION_PRIORITY[candidate] > DISPOSITION_PRIORITY[current]:
        return candidate
    return current


def _event_floor(event: EvidenceEvent) -> DispositionLevel:
    if event.category == "content_safety":
        if event.disputed:
            return DispositionLevel.P3
        if event.severity.value == "critical":
            return DispositionLevel.P4
        if event.severity.value == "high":
            return DispositionLevel.P4
        if event.severity.value == "medium":
            return DispositionLevel.P3
        return DispositionLevel.P1

    if event.risk_type == "combined_identity":
        return DispositionLevel.P3 if event.disputed else DispositionLevel.P2
    if event.risk_type == "api_unavailable":
        return DispositionLevel.P3 if event.disputed else DispositionLevel.P2
    if event.risk_type in {"id_card", "bank_card"}:
        return DispositionLevel.P2
    if event.risk_type in {"address", "student_id", "parent_contact"}:
        return DispositionLevel.P2 if event.disputed else DispositionLevel.P1
    return DispositionLevel.P1


def _build_redaction_targets(events: list[EvidenceEvent]) -> list[RedactionTarget]:
    targets: list[RedactionTarget] = []
    for event in events:
        if event.category != "privacy" or event.primary_span is None:
            continue
        targets.append(
            RedactionTarget(
                finding_id=event.finding_refs[0] if event.finding_refs else "",
                event_id=event.event_id,
                start=event.primary_span.start,
                end=event.primary_span.end,
                original_text=event.primary_span.text,
                replacement=event.remediation_suggestion or "<REDACTED>",
                pii_type=event.risk_type,
            )
        )
    return targets


def _build_redaction_targets_from_plan(
    plan: SpanConflictResolutionResult,
    events: list[EvidenceEvent],
) -> list[RedactionTarget]:
    event_id_by_finding_id: dict[str, str] = {}
    for event in events:
        for finding_id in event.finding_refs:
            event_id_by_finding_id.setdefault(finding_id, event.event_id)

    targets: list[RedactionTarget] = []
    for target in plan.redaction_targets:
        targets.append(
            target.model_copy(
                update={
                    "event_id": event_id_by_finding_id.get(target.finding_id, target.event_id),
                }
            )
        )
    return targets


def _priority_for(disposition: DispositionLevel) -> str:
    if disposition in {DispositionLevel.P4, DispositionLevel.P5}:
        return "critical"
    if disposition == DispositionLevel.P3:
        return "high"
    if disposition == DispositionLevel.P2:
        return "normal"
    return "low"


def _actions_for(disposition: DispositionLevel, has_redactions: bool) -> tuple[list[str], str, str]:
    if disposition == DispositionLevel.P0:
        return ["release"], "none", ""
    if disposition == DispositionLevel.P1:
        return ["apply_redaction", "release"], "structured_masking", ""
    if disposition == DispositionLevel.P2:
        return ["apply_redaction", "restrict_internal_access", "hold_external_delivery"], "structured_masking", ""
    if disposition == DispositionLevel.P3:
        return ["manual_review_required", "hold_delivery"], "manual_redaction_plan", "hard_case_or_boundary_risk"
    if disposition == DispositionLevel.P4:
        return ["block_delivery", "escalate_compliance"], "block", "high_risk_content_or_sensitive_data"
    return ["block_delivery", "traceback_and_incident_response"], "block_and_traceback", "circulated_high_risk_data"


def _disposition_from_privacy_assessment(
    assessment: PrivacyDocumentAssessmentRecord | None,
) -> DispositionLevel:
    if assessment is None:
        return DispositionLevel.P0
    if assessment.metadata.get("can_raise_disposition") is False:
        return DispositionLevel.P0
    action = assessment.recommended_action
    if action == "keep":
        return DispositionLevel.P0
    if action == "redact":
        return DispositionLevel.P1
    if action == "generalize":
        return DispositionLevel.P2
    if action in {"manual_review", "exclude_from_training"}:
        return DispositionLevel.P3
    return DispositionLevel.P1


def _disposition_from_content_assessment(
    assessment: ContentDocumentAssessmentRecord | None,
) -> DispositionLevel:
    if assessment is None:
        return DispositionLevel.P0
    if assessment.metadata.get("can_raise_disposition") is False:
        return DispositionLevel.P0
    action = assessment.recommended_action
    if action == "keep":
        return DispositionLevel.P0
    if action == "restricted_review":
        return DispositionLevel.P2
    if action == "manual_review":
        return DispositionLevel.P3
    if action in {"exclude_from_training", "reject"}:
        return DispositionLevel.P4
    return DispositionLevel.P3


def _fragment_disposition_floor(
    privacy_fragments: list[PrivacyFragmentAdjudicationRecord],
    content_fragments: list[ContentFragmentAdjudicationRecord],
) -> DispositionLevel:
    disposition = DispositionLevel.P0
    for item in privacy_fragments:
        if item.governance_action == "redact":
            disposition = _max_disposition(disposition, DispositionLevel.P1)
        elif item.governance_action == "generalize":
            disposition = _max_disposition(disposition, DispositionLevel.P2)
        elif item.governance_action in {"manual_review", "exclude_from_training"}:
            disposition = _max_disposition(disposition, DispositionLevel.P3)
    for item in content_fragments:
        if item.recommended_action == "restricted_review":
            disposition = _max_disposition(disposition, DispositionLevel.P2)
        elif item.recommended_action == "manual_review":
            disposition = _max_disposition(disposition, DispositionLevel.P3)
        elif item.recommended_action in {"exclude_from_training", "reject"}:
            disposition = _max_disposition(disposition, DispositionLevel.P4)
    return disposition


def _localized_fragment_floor(
    localized_fragments: list[ContentLocalizedFragmentRecord],
    content_fragments: list[ContentFragmentAdjudicationRecord],
) -> DispositionLevel:
    """Use localization as a fallback signal when adjudication did not cover it."""
    if not localized_fragments:
        return DispositionLevel.P0
    if content_fragments:
        if any(item.is_degraded for item in localized_fragments):
            return DispositionLevel.P3
        return DispositionLevel.P0

    disposition = DispositionLevel.P0
    for item in localized_fragments:
        if item.is_degraded:
            disposition = _max_disposition(disposition, DispositionLevel.P3)
        elif item.severity.value in {"critical", "high"}:
            disposition = _max_disposition(disposition, DispositionLevel.P3)
        elif item.severity.value == "medium":
            disposition = _max_disposition(disposition, DispositionLevel.P2)
        else:
            disposition = _max_disposition(disposition, DispositionLevel.P1)
    return disposition


def _localized_fragment_metadata(items: list[ContentLocalizedFragmentRecord]) -> list[dict]:
    records: list[dict] = []
    for item in items[:20]:
        span = item.span
        records.append(
            {
                "fragment_id": item.fragment_id,
                "window_id": item.window_id,
                "risk_type": item.risk_type,
                "policy_tag": item.policy_tag,
                "severity": item.severity.value,
                "confidence": item.confidence,
                "source_tool": item.source_tool,
                "is_degraded": item.is_degraded,
                "text": span.text if span else "",
                "start": span.start if span else None,
                "end": span.end if span else None,
                "explanation": item.explanation,
            }
        )
    return records


def _decision_explanation(
    privacy_assessment: PrivacyDocumentAssessmentRecord | None,
    content_assessment: ContentDocumentAssessmentRecord | None,
    document_context: DocumentContextRecord | None,
    doc_events: list[EvidenceEvent],
) -> str:
    parts: list[str] = []
    if document_context is not None:
        parts.append(
            f"Document context: {document_context.document_type}/{document_context.scene_type}. "
            f"{document_context.explanation}"
        )
    if privacy_assessment is not None:
        parts.append(f"Privacy assessment: {privacy_assessment.explanation}")
    if content_assessment is not None:
        parts.append(f"Content assessment: {content_assessment.explanation}")
    if not parts:
        parts.append(f"Decision derived from {len(doc_events)} evidence events.")
    return " ".join(part.strip() for part in parts if part.strip())


def run(
    ingest_units: list[IngestUnit],
    events: list[EvidenceEvent],
    adjudications: list[HardCaseAdjudicationResult],
    settings: Settings | None = None,
    redaction_plans: list[SpanConflictResolutionResult] | None = None,
    document_contexts: list[DocumentContextRecord] | None = None,
    privacy_fragment_adjudications: list[PrivacyFragmentAdjudicationRecord] | None = None,
    content_fragment_adjudications: list[ContentFragmentAdjudicationRecord] | None = None,
    content_candidate_windows: list[ContentCandidateWindowRecord] | None = None,
    content_localized_fragments: list[ContentLocalizedFragmentRecord] | None = None,
    privacy_document_assessments: list[PrivacyDocumentAssessmentRecord] | None = None,
    content_document_assessments: list[ContentDocumentAssessmentRecord] | None = None,
) -> list[PolicyDecisionRecord]:
    settings = settings or get_settings()
    events_by_doc: dict[str, list[EvidenceEvent]] = {}
    for event in events:
        events_by_doc.setdefault(event.doc_id, []).append(event)
    adjudication_by_doc = {item.doc_id: item for item in adjudications}
    redaction_plan_by_doc = {item.doc_id: item for item in redaction_plans or []}
    context_by_doc = {item.doc_id: item for item in (document_contexts or [])}
    privacy_doc_assessment_by_doc = {item.doc_id: item for item in (privacy_document_assessments or [])}
    content_doc_assessment_by_doc = {item.doc_id: item for item in (content_document_assessments or [])}
    privacy_fragments_by_doc: dict[str, list[PrivacyFragmentAdjudicationRecord]] = {}
    for item in privacy_fragment_adjudications or []:
        privacy_fragments_by_doc.setdefault(item.doc_id, []).append(item)
    content_fragments_by_doc: dict[str, list[ContentFragmentAdjudicationRecord]] = {}
    for item in content_fragment_adjudications or []:
        content_fragments_by_doc.setdefault(item.doc_id, []).append(item)
    candidate_windows_by_doc: dict[str, list[ContentCandidateWindowRecord]] = {}
    for item in content_candidate_windows or []:
        candidate_windows_by_doc.setdefault(item.doc_id, []).append(item)
    localized_fragments_by_doc: dict[str, list[ContentLocalizedFragmentRecord]] = {}
    for item in content_localized_fragments or []:
        localized_fragments_by_doc.setdefault(item.doc_id, []).append(item)

    decisions: list[PolicyDecisionRecord] = []
    for unit in ingest_units:
        doc_events = events_by_doc.get(unit.doc_id, [])
        disposition = DispositionLevel.P0
        reason_codes: list[str] = []
        privacy_assessment = privacy_doc_assessment_by_doc.get(unit.doc_id)
        content_assessment = content_doc_assessment_by_doc.get(unit.doc_id)
        privacy_fragments = privacy_fragments_by_doc.get(unit.doc_id, [])
        content_fragments = content_fragments_by_doc.get(unit.doc_id, [])
        candidate_windows = candidate_windows_by_doc.get(unit.doc_id, [])
        localized_fragments = localized_fragments_by_doc.get(unit.doc_id, [])
        document_context = context_by_doc.get(unit.doc_id)

        disposition = _max_disposition(disposition, _fragment_disposition_floor(privacy_fragments, content_fragments))
        disposition = _max_disposition(disposition, _localized_fragment_floor(localized_fragments, content_fragments))
        privacy_doc_can_raise = bool(privacy_assessment and privacy_assessment.metadata.get("can_raise_disposition", True))
        content_doc_can_raise = bool(content_assessment and content_assessment.metadata.get("can_raise_disposition", True))
        disposition = _max_disposition(disposition, _disposition_from_privacy_assessment(privacy_assessment))
        disposition = _max_disposition(disposition, _disposition_from_content_assessment(content_assessment))
        if localized_fragments:
            reason_codes.append(f"content_localized_fragments:{len(localized_fragments)}")
        if any(item.is_degraded for item in localized_fragments):
            reason_codes.append("content_localization:degraded")

        for event in doc_events:
            reason_codes.append(event.policy_tag)
            if event.category in {"privacy_document", "content_document"}:
                continue
            if event.category == "content_safety" and content_assessment is not None and content_doc_can_raise:
                continue
            if event.category == "privacy" and privacy_assessment is not None and privacy_doc_can_raise:
                continue
            disposition = _max_disposition(disposition, _event_floor(event))
        if privacy_assessment is not None:
            reason_codes.append(f"privacy_doc:{privacy_assessment.recommended_action}")
        if content_assessment is not None:
            reason_codes.append(f"content_doc:{content_assessment.recommended_action}")

        adjudication = adjudication_by_doc.get(unit.doc_id)
        if adjudication:
            disposition = _max_disposition(disposition, adjudication.judgement.recommended_disposition)
            reason_codes.append(f"hard_case:{adjudication.provider_name}")

        circulation_state = str(unit.metadata.get("circulation_status", "")).lower()
        already_released = bool(unit.metadata.get("already_released")) or circulation_state in {"released", "published", "circulated"}
        if already_released and disposition == DispositionLevel.P4:
            disposition = DispositionLevel.P5

        redaction_plan = redaction_plan_by_doc.get(unit.doc_id)
        redaction_targets = (
            _build_redaction_targets_from_plan(redaction_plan, doc_events)
            if redaction_plan is not None
            else _build_redaction_targets(doc_events)
        )
        required_actions, redaction_method, blocked_reason = _actions_for(
            disposition,
            has_redactions=bool(redaction_targets),
        )
        trust_level = TrustLevel.DEGRADED if adjudication and adjudication.is_degraded else TrustLevel.FULL

        decisions.append(
            PolicyDecisionRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                disposition_level=disposition,
                unified_decision=DISPOSITION_TO_DECISION[disposition],
                risk_score=DISPOSITION_RISK[disposition],
                required_actions=required_actions,
                redaction_targets=redaction_targets,
                redaction_method=redaction_method,
                blocked_reason=blocked_reason,
                review_priority=_priority_for(disposition),
                reason_codes=sorted(set(reason_codes)),
                evidence_event_ids=[event.event_id for event in doc_events],
                summary=f"{disposition.value} decision derived from {len(doc_events)} evidence events.",
                explanation=_decision_explanation(privacy_assessment, content_assessment, document_context, doc_events),
                policy_version=settings.policy_version,
                trust_level=trust_level,
                metadata={
                    "document_context": document_context.model_dump(mode="json") if document_context else {},
                    "privacy_document_assessment": privacy_assessment.model_dump(mode="json") if privacy_assessment else {},
                    "content_document_assessment": content_assessment.model_dump(mode="json") if content_assessment else {},
                    "privacy_fragment_adjudication_count": len(privacy_fragments),
                    "content_candidate_window_count": len(candidate_windows),
                    "content_fragment_adjudication_count": len(content_fragments),
                    "content_localized_fragment_count": len(localized_fragments),
                    "content_localized_fragments": _localized_fragment_metadata(localized_fragments),
                },
            )
        )

    logger.info("Policy decision completed: %d documents", len(decisions))
    return decisions

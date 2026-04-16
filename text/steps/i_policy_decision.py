from __future__ import annotations

import logging

from common.enums import TrustLevel, UnifiedDecision
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DispositionLevel,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PolicyDecisionRecord,
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


def run(
    ingest_units: list[IngestUnit],
    events: list[EvidenceEvent],
    adjudications: list[HardCaseAdjudicationResult],
    settings: Settings | None = None,
    redaction_plans: list[SpanConflictResolutionResult] | None = None,
) -> list[PolicyDecisionRecord]:
    settings = settings or get_settings()
    events_by_doc: dict[str, list[EvidenceEvent]] = {}
    for event in events:
        events_by_doc.setdefault(event.doc_id, []).append(event)
    adjudication_by_doc = {item.doc_id: item for item in adjudications}
    redaction_plan_by_doc = {item.doc_id: item for item in redaction_plans or []}

    decisions: list[PolicyDecisionRecord] = []
    for unit in ingest_units:
        doc_events = events_by_doc.get(unit.doc_id, [])
        disposition = DispositionLevel.P0
        reason_codes: list[str] = []

        for event in doc_events:
            disposition = _max_disposition(disposition, _event_floor(event))
            reason_codes.append(event.policy_tag)

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
                policy_version=settings.policy_version,
                trust_level=trust_level,
            )
        )

    logger.info("Policy decision completed: %d documents", len(decisions))
    return decisions

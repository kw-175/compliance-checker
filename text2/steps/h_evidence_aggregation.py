from __future__ import annotations

import logging

from text.models.schemas import (
    ContentSafetyResult,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PrivacyDetectionResult,
)

logger = logging.getLogger(__name__)


def _doc_events(
    unit: IngestUnit,
    safety_result: ContentSafetyResult | None,
    privacy_result: PrivacyDetectionResult | None,
    adjudication: HardCaseAdjudicationResult | None,
) -> list[EvidenceEvent]:
    events: list[EvidenceEvent] = []
    findings = []
    if safety_result:
        findings.extend(safety_result.findings)
    if privacy_result:
        findings.extend(privacy_result.findings)

    for finding in findings:
        disputed = bool(adjudication and adjudication.judgement.requires_manual_review)
        explanation = finding.explanation
        if adjudication:
            explanation = f"{explanation} Adjudication rationale: {adjudication.judgement.rationale}"

        events.append(
            EvidenceEvent(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                category=finding.finding_type,
                risk_type=finding.risk_type,
                policy_tag=finding.policy_tag,
                severity=finding.severity,
                confidence_summary=finding.confidence,
                source_tools=[finding.source_tool] + ([adjudication.provider_name] if adjudication else []),
                finding_refs=[finding.finding_id],
                disputed=disputed,
                hard_case_applied=adjudication is not None,
                remediation_suggestion=finding.redaction_suggestion or finding.remediation_suggestion,
                explanation=explanation,
                primary_span=finding.span,
                metadata={
                    "task_id": unit.task_id,
                    "tenant_id": unit.tenant_id,
                    "profile_id": unit.profile_id,
                    "source_type": unit.source_type,
                },
            )
        )

    if adjudication and not events:
        events.append(
            EvidenceEvent(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                category="hard_case",
                risk_type="adjudication_only",
                policy_tag="hard_case.final_judgement",
                severity="medium",
                confidence_summary=adjudication.judgement.confidence,
                source_tools=[adjudication.provider_name],
                finding_refs=[],
                disputed=adjudication.judgement.requires_manual_review,
                hard_case_applied=True,
                remediation_suggestion="manual_review",
                explanation=adjudication.judgement.rationale,
                primary_span=None,
                metadata={"recommended_disposition": adjudication.judgement.recommended_disposition.value},
            )
        )

    return events


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    privacy_results: list[PrivacyDetectionResult],
    adjudications: list[HardCaseAdjudicationResult],
) -> list[EvidenceEvent]:
    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    adjudication_by_doc = {result.doc_id: result for result in adjudications}

    events: list[EvidenceEvent] = []
    for unit in ingest_units:
        events.extend(
            _doc_events(
                unit,
                safety_by_doc.get(unit.doc_id),
                privacy_by_doc.get(unit.doc_id),
                adjudication_by_doc.get(unit.doc_id),
            )
        )

    logger.info("Evidence aggregation completed: %d events", len(events))
    return events

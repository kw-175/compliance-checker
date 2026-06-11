from __future__ import annotations

import logging

from text.models.schemas import (
    ContentDocumentAssessmentRecord,
    ContentFragmentAdjudicationRecord,
    ContentSafetyResult,
    DocumentContextRecord,
    EvidenceEvent,
    HardCaseAdjudicationResult,
    IngestUnit,
    PrivacyDocumentAssessmentRecord,
    PrivacyFragmentAdjudicationRecord,
    PrivacyDetectionResult,
)

logger = logging.getLogger(__name__)


def _doc_events(
    unit: IngestUnit,
    safety_result: ContentSafetyResult | None,
    privacy_result: PrivacyDetectionResult | None,
    adjudication: HardCaseAdjudicationResult | None,
    document_context: DocumentContextRecord | None,
    privacy_fragment_adjudications: list[PrivacyFragmentAdjudicationRecord],
    content_fragment_adjudications: list[ContentFragmentAdjudicationRecord],
    privacy_document_assessment: PrivacyDocumentAssessmentRecord | None,
    content_document_assessment: ContentDocumentAssessmentRecord | None,
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
                    "document_context": document_context.model_dump(mode="json") if document_context else {},
                    "privacy_fragment_adjudication": next(
                        (item.model_dump(mode="json") for item in privacy_fragment_adjudications if item.finding_id == finding.finding_id),
                        {},
                    ),
                    "content_fragment_adjudication": next(
                        (item.model_dump(mode="json") for item in content_fragment_adjudications if item.finding_id == finding.finding_id),
                        {},
                    ),
                },
            )
        )

    if privacy_document_assessment is not None:
        events.append(
            EvidenceEvent(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                category="privacy_document",
                risk_type="privacy_document_assessment",
                policy_tag="privacy.document_assessment",
                severity="medium" if privacy_document_assessment.requires_manual_review else "low",
                confidence_summary=privacy_document_assessment.confidence,
                source_tools=[privacy_document_assessment.provider_name],
                finding_refs=[],
                disputed=privacy_document_assessment.requires_manual_review,
                hard_case_applied=False,
                remediation_suggestion=privacy_document_assessment.recommended_action,
                explanation=privacy_document_assessment.explanation,
                primary_span=None,
                metadata=privacy_document_assessment.model_dump(mode="json"),
            )
        )

    if content_document_assessment is not None:
        events.append(
            EvidenceEvent(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                category="content_document",
                risk_type="content_document_assessment",
                policy_tag="content.document_assessment",
                severity="medium" if content_document_assessment.requires_manual_review else "low",
                confidence_summary=content_document_assessment.confidence,
                source_tools=[content_document_assessment.provider_name],
                finding_refs=[],
                disputed=content_document_assessment.requires_manual_review,
                hard_case_applied=False,
                remediation_suggestion=content_document_assessment.recommended_action,
                explanation=content_document_assessment.explanation,
                primary_span=None,
                metadata=content_document_assessment.model_dump(mode="json"),
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
    document_contexts: list[DocumentContextRecord] | None = None,
    privacy_fragment_adjudications: list[PrivacyFragmentAdjudicationRecord] | None = None,
    content_fragment_adjudications: list[ContentFragmentAdjudicationRecord] | None = None,
    privacy_document_assessments: list[PrivacyDocumentAssessmentRecord] | None = None,
    content_document_assessments: list[ContentDocumentAssessmentRecord] | None = None,
) -> list[EvidenceEvent]:
    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    adjudication_by_doc = {result.doc_id: result for result in adjudications}
    context_by_doc = {result.doc_id: result for result in (document_contexts or [])}
    privacy_doc_assessment_by_doc = {result.doc_id: result for result in (privacy_document_assessments or [])}
    content_doc_assessment_by_doc = {result.doc_id: result for result in (content_document_assessments or [])}
    privacy_fragments_by_doc: dict[str, list[PrivacyFragmentAdjudicationRecord]] = {}
    for item in privacy_fragment_adjudications or []:
        privacy_fragments_by_doc.setdefault(item.doc_id, []).append(item)
    content_fragments_by_doc: dict[str, list[ContentFragmentAdjudicationRecord]] = {}
    for item in content_fragment_adjudications or []:
        content_fragments_by_doc.setdefault(item.doc_id, []).append(item)

    events: list[EvidenceEvent] = []
    for unit in ingest_units:
        events.extend(
            _doc_events(
                unit,
                safety_by_doc.get(unit.doc_id),
                privacy_by_doc.get(unit.doc_id),
                adjudication_by_doc.get(unit.doc_id),
                context_by_doc.get(unit.doc_id),
                privacy_fragments_by_doc.get(unit.doc_id, []),
                content_fragments_by_doc.get(unit.doc_id, []),
                privacy_doc_assessment_by_doc.get(unit.doc_id),
                content_doc_assessment_by_doc.get(unit.doc_id),
            )
        )

    logger.info("Evidence aggregation completed: %d events", len(events))
    return events

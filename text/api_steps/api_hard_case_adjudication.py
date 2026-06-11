from __future__ import annotations

import json
import logging

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentDocumentAssessmentRecord,
    ContentSafetyResult,
    DocumentContextRecord,
    HardCaseAdjudicationResult,
    HardCaseJudgement,
    IngestUnit,
    PrivacyDocumentAssessmentRecord,
    PrivacyDetectionResult,
)
from text.prompt_loader import load_prompt
from text.steps.hard_case_adjudication import _heuristic_judgement

logger = logging.getLogger(__name__)

def _payload(
    unit: IngestUnit,
    safety_result: ContentSafetyResult | None,
    privacy_result: PrivacyDetectionResult | None,
    settings: Settings,
    document_context: DocumentContextRecord | None,
) -> dict:
    provider = resolve_provider_config(settings)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text_excerpt": unit.text[: provider.max_chars],
        "metadata": unit.metadata,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "preliminary_content_findings": [
            finding.model_dump(mode="json") for finding in (safety_result.findings if safety_result else [])
        ],
        "preliminary_privacy_findings": [
            finding.model_dump(mode="json") for finding in (privacy_result.findings if privacy_result else [])
        ],
        "content_needs_adjudication": bool(safety_result and safety_result.needs_adjudication),
        "privacy_needs_adjudication": bool(privacy_result and privacy_result.needs_adjudication),
    }


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    privacy_results: list[PrivacyDetectionResult],
    settings: Settings | None = None,
    document_contexts: list[DocumentContextRecord] | None = None,
    content_document_assessments: list[ContentDocumentAssessmentRecord] | None = None,
    privacy_document_assessments: list[PrivacyDocumentAssessmentRecord] | None = None,
) -> list[HardCaseAdjudicationResult]:
    settings = settings or get_settings()
    if not settings.enable_hard_case_adjudication:
        return []

    client = OpenAICompatibleComplianceClient(settings)
    provider = resolve_provider_config(settings)
    system_prompt = load_prompt(str(settings.api_hard_case_prompt_path))
    safety_by_doc = {result.doc_id: result for result in safety_results}
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    context_by_doc = {item.doc_id: item for item in (document_contexts or [])}
    content_assessment_by_doc = {item.doc_id: item for item in (content_document_assessments or [])}
    privacy_assessment_by_doc = {item.doc_id: item for item in (privacy_document_assessments or [])}
    results: list[HardCaseAdjudicationResult] = []

    for unit in ingest_units:
        safety_result = safety_by_doc.get(unit.doc_id)
        privacy_result = privacy_by_doc.get(unit.doc_id)
        document_context = context_by_doc.get(unit.doc_id)
        content_assessment = content_assessment_by_doc.get(unit.doc_id)
        privacy_assessment = privacy_assessment_by_doc.get(unit.doc_id)
        trigger_sources: list[str] = []
        if provider.mode == "local_model":
            if safety_result and safety_result.needs_adjudication and content_assessment is None:
                trigger_sources.append("content_safety")
            if privacy_result and privacy_result.needs_adjudication and privacy_assessment is None:
                trigger_sources.append("privacy")
        else:
            if safety_result and safety_result.needs_adjudication:
                trigger_sources.append("content_safety")
            if privacy_result and privacy_result.needs_adjudication:
                trigger_sources.append("privacy")
        if not trigger_sources:
            continue

        judgement: HardCaseJudgement | None = None
        notes: list[str] = []
        raw_response = ""
        provider_name = "local_hard_case_adjudicator" if provider.mode == "local_model" else "api_hard_case_adjudicator"
        is_degraded = False

        try:
            response_payload = client.complete_json(
                task_name="hard_case_adjudication",
                system_prompt=system_prompt,
                payload=_payload(unit, safety_result, privacy_result, settings, document_context),
            )
            judgement = HardCaseJudgement.model_validate(response_payload)
            raw_response = json.dumps(response_payload, ensure_ascii=False)
        except (OpenAICompatibleAPIError, Exception) as exc:
            logger.warning("API hard-case adjudication failed for %s: %s", unit.doc_id, exc)
            judgement = _heuristic_judgement(safety_result, privacy_result)
            provider_name = "api_hard_case_heuristic_fallback"
            is_degraded = True
            raw_response = judgement.model_dump_json()
            notes.append(f"api_hard_case_failed: {exc}")
            notes.append("heuristic fallback used because API hard-case adjudicator was unavailable.")

        results.append(
            HardCaseAdjudicationResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                trigger_sources=trigger_sources,
                model_name=provider.model,
                provider_name=provider_name,
                prompt_version="local-hard-case-v1" if provider.mode == "local_model" else "api-hard-case-v1",
                adjudicated=True,
                is_degraded=is_degraded,
                uncertainty=round(1.0 - judgement.confidence, 4),
                judgement=judgement,
                raw_response=raw_response,
                notes=notes,
            )
        )

    logger.info("API hard-case adjudication completed: %d documents", len(results))
    return results

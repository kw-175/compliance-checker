from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentFragmentAdjudicationRecord,
    ContentSafetyResult,
    DocumentContextRecord,
    IngestUnit,
)
from text.prompt_loader import load_prompt
from text.steps.adjudication_payloads import compact_content_finding_payload, snippet_window

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _heuristic_adjudication(
    unit: IngestUnit,
    result: ContentSafetyResult,
    document_context: DocumentContextRecord | None,
) -> list[ContentFragmentAdjudicationRecord]:
    records: list[ContentFragmentAdjudicationRecord] = []
    for finding in result.findings:
        semantic_role = "propagating_risk"
        operationality = "high" if finding.severity.value in {"high", "critical"} else "medium"
        protective_context = bool(document_context and document_context.document_type in {"textbook_example", "news_report"})
        action = "reject" if finding.severity.value in {"high", "critical"} and not protective_context else "manual_review"
        explanation = finding.explanation

        if protective_context:
            semantic_role = "teaching" if document_context.document_type == "textbook_example" else "discussion"
            action = "restricted_review"
            explanation = (
                f"The fragment matches {finding.risk_type}, but the surrounding document looks like "
                f"{document_context.document_type}, so it should be reviewed as protective context rather than silently allowed."
            )
        elif finding.needs_adjudication:
            semantic_role = "uncertain"
            action = "manual_review"
            explanation = explanation or "The fragment needs contextual review before a final training decision."

        records.append(
            ContentFragmentAdjudicationRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                finding_id=finding.finding_id,
                risk_type=finding.risk_type,
                semantic_role=semantic_role,
                operationality=operationality,
                audience_risk="minor_sensitive" if document_context and document_context.contains_minor_context else "normal",
                protective_context=protective_context,
                recommended_action=action,
                training_eligibility="blocked" if action == "reject" else "restricted",
                allow_downstream_annotation=False,
                requires_manual_review=action != "reject",
                explanation=explanation,
                confidence=max(0.45, finding.confidence),
                provider_name="heuristic_content_fragment_adjudicator",
                provider_version="builtin-2026.05",
                is_degraded=False,
                metadata={"document_type": document_context.document_type if document_context else ""},
            )
        )
    return records


def _payload(
    unit: IngestUnit,
    finding,
    document_context: DocumentContextRecord | None,
    provider_max_chars: int,
) -> dict[str, Any]:
    span = finding.span
    text = unit.text[:provider_max_chars]
    if span is not None:
        text = snippet_window(unit.text, span.start, span.end, size=240)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": text,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "findings": [compact_content_finding_payload(finding)],
    }


def _normalize_records(
    unit: IngestUnit,
    result: ContentSafetyResult,
    payload: dict[str, Any],
    provider_name: str,
    provider_version: str,
) -> list[ContentFragmentAdjudicationRecord]:
    raw_items = payload.get("adjudications") or payload.get("findings") or payload.get("results") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []

    by_finding = {finding.finding_id: finding for finding in result.findings}
    normalized: list[ContentFragmentAdjudicationRecord] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        finding_id = str(item.get("finding_id") or "")
        finding = by_finding.get(finding_id)
        if finding is None:
            continue
        normalized.append(
            ContentFragmentAdjudicationRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                finding_id=finding_id,
                risk_type=finding.risk_type,
                semantic_role=str(item.get("semantic_role") or "uncertain"),
                operationality=str(item.get("operationality") or "medium"),
                audience_risk=str(item.get("audience_risk") or "normal"),
                protective_context=bool(item.get("protective_context", False)),
                recommended_action=str(item.get("recommended_action") or "manual_review"),
                training_eligibility=str(item.get("training_eligibility") or "restricted"),
                allow_downstream_annotation=bool(item.get("allow_downstream_annotation", False)),
                requires_manual_review=bool(item.get("requires_manual_review", True)),
                explanation=str(item.get("explanation") or finding.explanation),
                confidence=_safe_float(item.get("confidence"), finding.confidence),
                provider_name=provider_name,
                provider_version=provider_version,
                is_degraded=False,
                metadata={"raw_payload": item},
            )
        )
    return normalized


def run(
    ingest_units: list[IngestUnit],
    safety_results: list[ContentSafetyResult],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> list[ContentFragmentAdjudicationRecord]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    safety_by_doc = {result.doc_id: result for result in safety_results}
    context_by_doc = {item.doc_id: item for item in document_contexts}

    if provider.mode != "local_model":
        records: list[ContentFragmentAdjudicationRecord] = []
        for unit in ingest_units:
            result = safety_by_doc.get(unit.doc_id)
            if result is None:
                continue
            records.extend(_heuristic_adjudication(unit, result, context_by_doc.get(unit.doc_id)))
        return records

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.local_content_fragment_prompt_path))
    all_records: list[ContentFragmentAdjudicationRecord] = []
    for unit in ingest_units:
        result = safety_by_doc.get(unit.doc_id)
        if result is None or not result.findings:
            continue
        document_context = context_by_doc.get(unit.doc_id)
        for finding in result.findings:
            try:
                payload = client.complete_json(
                    task_name="content_fragment_adjudication",
                    system_prompt=system_prompt,
                    payload=_payload(unit, finding, document_context, provider.max_chars),
                )
                records = _normalize_records(
                    unit,
                    result.model_copy(update={"findings": [finding]}),
                    payload,
                    provider_name="local_content_fragment_adjudicator",
                    provider_version=provider.model,
                )
                if not records:
                    records = _heuristic_adjudication(unit, result.model_copy(update={"findings": [finding]}), document_context)
                    for record in records:
                        record.is_degraded = True
                        record.metadata["degrade_reason"] = "empty_local_adjudication"
                all_records.extend(records)
            except Exception as exc:
                logger.warning("Local content fragment adjudication failed for %s/%s: %s", unit.doc_id, finding.finding_id, exc)
                records = _heuristic_adjudication(unit, result.model_copy(update={"findings": [finding]}), document_context)
                for record in records:
                    record.is_degraded = True
                    record.metadata["degrade_reason"] = str(exc)
                all_records.extend(records)
    return all_records

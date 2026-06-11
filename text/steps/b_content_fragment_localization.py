from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentCandidateWindowRecord,
    ContentLocalizedFragmentRecord,
    ContentSafetyResult,
    DetectionStatus,
    DetectionFinding,
    DocumentContextRecord,
    IngestUnit,
    Severity,
    TextSpan,
)
from text.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def _severity_from_text(value: str) -> Severity:
    text = str(value or "").strip().lower()
    if text == "critical":
        return Severity.CRITICAL
    if text == "high":
        return Severity.HIGH
    if text == "low":
        return Severity.LOW
    return Severity.MEDIUM


def _normalize_content_label(fragment_text: str, risk_type: str, policy_tag: str, explanation: str) -> tuple[str, str, str]:
    text = f"{fragment_text} {explanation}".lower()
    risk = str(risk_type or "general_content_safety").strip() or "general_content_safety"
    policy = str(policy_tag or "content.localized").strip() or "content.localized"
    self_harm_terms = ("自杀", "自残", "伤害自己", "不想活", "轻生", "suicide", "self-harm", "kill myself")
    public_safety_terms = ("公共事件", "隐瞒真相", "市中心集合", "别报备", "集合", "通知", "游行", "protest", "assembly")
    illegal_terms = ("改成绩", "考试答案", "校园账号", "私聊", "保证录取", "3999", "黑客", "作弊", "诈骗")
    if risk == "self_harm" and not any(term in text for term in self_harm_terms):
        if any(term in fragment_text for term in public_safety_terms):
            return "political_sensitive", "content.political", "该片段涉及公共事件、组织聚集或规避报备等公共安全敏感表达，已从自伤自杀纠正为政治与公共安全敏感风险。"
        if any(term in fragment_text for term in illegal_terms):
            return "illegal_instruction", "content.illegal_instruction", "该片段涉及违法、欺诈或规避规则行为，已从自伤自杀纠正为违法危险教程风险。"
        return "general_content_safety", "content.localized", "该片段不包含自伤自杀语义，已降级为一般内容安全风险并等待复核。"
    return risk, policy, explanation


def _rule_fragment(window: ContentCandidateWindowRecord, hit: dict[str, Any], unit: IngestUnit) -> ContentLocalizedFragmentRecord:
    start = int(hit.get("start", window.start))
    end = int(hit.get("end", window.end))
    return ContentLocalizedFragmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        window_id=window.window_id,
        risk_type=str(hit.get("risk_type") or "general_content_safety"),
        policy_tag=str(hit.get("policy_tag") or "content.rule"),
        severity=_severity_from_text(str(hit.get("severity") or "medium")),
        confidence=float(hit.get("score") or 0.85),
        explanation=str(hit.get("reason") or "Rule-matched risky phrase."),
        span=TextSpan(start=start, end=end, text=unit.text[start:end], context_before=unit.text[max(0, start - 40):start], context_after=unit.text[end:min(len(unit.text), end + 40)]),
        source_tool="content_rule_localizer",
        metadata={"recall_source": "rule_engine"},
    )


def _align_fragment(unit: IngestUnit, window: ContentCandidateWindowRecord, fragment_text: str) -> tuple[int, int] | None:
    if not fragment_text:
        return None
    index = window.text.find(fragment_text)
    if index < 0:
        return None
    start = window.start + index
    end = start + len(fragment_text)
    return start, end


def _payload(window: ContentCandidateWindowRecord, document_context: DocumentContextRecord | None) -> dict[str, Any]:
    return {
        "doc_id": window.doc_id,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "candidate_window": {
            "window_id": window.window_id,
            "text": window.text,
            "candidate_labels": window.candidate_labels,
            "candidate_score": window.candidate_score,
            "recall_sources": window.recall_sources,
        },
    }


def _normalize_model_fragments(
    unit: IngestUnit,
    window: ContentCandidateWindowRecord,
    payload: dict[str, Any],
) -> list[ContentLocalizedFragmentRecord]:
    raw_items = payload.get("fragments") or payload.get("findings") or []
    if isinstance(raw_items, dict):
        raw_items = [raw_items]
    if not isinstance(raw_items, list):
        raw_items = []
    records: list[ContentLocalizedFragmentRecord] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        fragment_text = str(item.get("text") or "").strip()
        aligned = _align_fragment(unit, window, fragment_text)
        if aligned is None:
            continue
        start, end = aligned
        risk_type, policy_tag, explanation = _normalize_content_label(
            fragment_text,
            str(item.get("risk_type") or "general_content_safety"),
            str(item.get("policy_tag") or "content.localized"),
            str(item.get("explanation") or "Localized content fragment."),
        )
        records.append(
            ContentLocalizedFragmentRecord(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                window_id=window.window_id,
                risk_type=risk_type,
                policy_tag=policy_tag,
                severity=_severity_from_text(str(item.get("severity") or "medium")),
                confidence=float(item.get("confidence") or 0.7),
                explanation=explanation,
                span=TextSpan(start=start, end=end, text=unit.text[start:end], context_before=unit.text[max(0, start - 40):start], context_after=unit.text[end:min(len(unit.text), end + 40)]),
                source_tool="qwen3.5_content_localizer",
                metadata={"recall_source": "localized_fragment"},
            )
        )
    return records


def _degraded_window_fragment(unit: IngestUnit, window: ContentCandidateWindowRecord, reason: str) -> ContentLocalizedFragmentRecord:
    return ContentLocalizedFragmentRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        window_id=window.window_id,
        risk_type="general_content_safety",
        policy_tag="content.window_candidate",
        severity=Severity.MEDIUM,
        confidence=0.0,
        explanation="The candidate window could not be localized to a smaller risky phrase and should be reviewed as a whole candidate window.",
        span=TextSpan(start=window.start, end=window.end, text=window.text, context_before="", context_after=""),
        source_tool="content_window_fallback",
        is_degraded=True,
        metadata={"degrade_reason": reason, "recall_sources": window.recall_sources},
    )


def run(
    ingest_units: list[IngestUnit],
    candidate_windows: list[ContentCandidateWindowRecord],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> tuple[list[ContentLocalizedFragmentRecord], list[ContentSafetyResult]]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    windows_by_doc: dict[str, list[ContentCandidateWindowRecord]] = {}
    for window in candidate_windows:
        windows_by_doc.setdefault(window.doc_id, []).append(window)
    unit_by_doc = {unit.doc_id: unit for unit in ingest_units}
    context_by_doc = {item.doc_id: item for item in document_contexts}

    client = OpenAICompatibleComplianceClient(settings) if provider.mode == "local_model" else None
    system_prompt = load_prompt(str(settings.local_content_localization_prompt_path)) if provider.mode == "local_model" else ""

    all_fragments: list[ContentLocalizedFragmentRecord] = []
    safety_results: list[ContentSafetyResult] = []

    for unit in ingest_units:
        windows = windows_by_doc.get(unit.doc_id, [])
        document_context = context_by_doc.get(unit.doc_id)
        doc_fragments: list[ContentLocalizedFragmentRecord] = []
        for window in windows:
            if window.rule_hits:
                for hit in window.rule_hits:
                    doc_fragments.append(_rule_fragment(window, hit, unit))
                continue
            if provider.mode == "local_model" and client is not None:
                try:
                    payload = client.complete_json(
                        task_name="content_fragment_localization",
                        system_prompt=system_prompt,
                        payload=_payload(window, document_context),
                    )
                    localized = _normalize_model_fragments(unit, window, payload)
                    if localized:
                        doc_fragments.extend(localized)
                    else:
                        doc_fragments.append(_degraded_window_fragment(unit, window, "empty_localization"))
                except Exception as exc:
                    logger.warning("Content fragment localization failed for %s/%s: %s", unit.doc_id, window.window_id, exc)
                    doc_fragments.append(_degraded_window_fragment(unit, window, str(exc)))
            else:
                doc_fragments.append(_degraded_window_fragment(unit, window, "non_local_provider"))

        all_fragments.extend(doc_fragments)
        findings: list[DetectionFinding] = []
        for fragment in doc_fragments:
            span = fragment.span
            findings.append(
                DetectionFinding(
                    doc_id=unit.doc_id,
                    finding_type="content_safety",
                    risk_type=fragment.risk_type,
                    policy_tag=fragment.policy_tag,
                    severity=fragment.severity,
                    confidence=fragment.confidence,
                    explanation=fragment.explanation,
                    source_tool=fragment.source_tool,
                    remediation_suggestion="manual_review",
                    needs_adjudication=True,
                    hard_case_reason="localized_candidate",
                    span=span,
                    attributes={
                        "content_safety": {
                            "matched_label": fragment.policy_tag,
                            "candidate_window_id": fragment.window_id,
                            "requires_manual_review": True,
                        },
                        "localized_fragment": fragment.model_dump(mode="json"),
                    },
                )
            )
        safety_results.append(
            ContentSafetyResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                status=DetectionStatus.HARD_CASE if findings else DetectionStatus.CLEAR,
                risk_score=max((finding.confidence for finding in findings), default=0.0),
                summary="Localized content safety candidate recall completed." if findings else "No localized content risk fragments were produced.",
                findings=findings,
                needs_adjudication=bool(findings),
                hard_case_reasons=["localized_candidate"] if findings else [],
                provider_name="local_content_candidate_recall" if provider.mode == "local_model" else "content_candidate_recall",
                provider_version=provider.model if provider.mode == "local_model" else "builtin-2026.05",
                is_degraded=any(fragment.is_degraded for fragment in doc_fragments),
            )
        )

    logger.info("Content fragment localization completed: %d fragments", len(all_fragments))
    return all_fragments, safety_results

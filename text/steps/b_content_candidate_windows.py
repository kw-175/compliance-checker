from __future__ import annotations

import logging
import re
from typing import Any

from text.config.settings import Settings, get_settings
from text.engines.content_rule_engine import ContentRuleHit, recall_content_rules
from text.models.schemas import ContentCandidateWindowRecord, DocumentContextRecord, IngestUnit

logger = logging.getLogger(__name__)

_PRIVACY_ONLY_GUARD_LABELS = {
    "pii",
    "privacy",
    "personal_information",
    "personal information",
    "personal_data",
    "personal data",
    "sensitive_personal_information",
}


def _is_privacy_only_guard_label(label: str) -> bool:
    normalized = label.strip().lower().replace("-", "_")
    return normalized in _PRIVACY_ONLY_GUARD_LABELS


def _sentence_windows(text: str) -> list[tuple[int, int, str]]:
    pattern = re.compile(r".+?(?:[。！？!?]|$)", re.S)
    windows: list[tuple[int, int, str]] = []
    for match in pattern.finditer(text):
        snippet = match.group(0).strip()
        if not snippet:
            continue
        windows.append((match.start(), match.end(), text[match.start():match.end()]))
    if windows:
        return windows
    return [(0, len(text), text)]


def _merge_short_windows(
    windows: list[tuple[int, int, str]],
    max_chars: int,
) -> list[tuple[int, int, str]]:
    merged: list[tuple[int, int, str]] = []
    cursor_start = None
    cursor_end = None
    cursor_text = ""
    for start, end, text in windows:
        if cursor_start is None:
            cursor_start, cursor_end, cursor_text = start, end, text
            continue
        if len(cursor_text) + len(text) <= max_chars:
            cursor_end = end
            cursor_text += text
            continue
        merged.append((cursor_start, cursor_end or cursor_start, cursor_text))
        cursor_start, cursor_end, cursor_text = start, end, text
    if cursor_start is not None:
        merged.append((cursor_start, cursor_end or cursor_start, cursor_text))
    return merged


def _rule_hits_by_window(
    rule_hits: list[ContentRuleHit],
    start: int,
    end: int,
) -> list[ContentRuleHit]:
    return [hit for hit in rule_hits if start <= hit.start < end or start < hit.end <= end or (hit.start <= start and hit.end >= end)]


def _guard_window_labels(window_text: str, settings: Settings, doc_id: str) -> tuple[list[str], float, list[str]]:
    if not settings.enable_qwen3guard or not settings.qwen3guard_endpoint:
        return [], 0.0, []
    try:
        import httpx

        response = httpx.post(
            settings.qwen3guard_endpoint,
            json={
                "doc_id": doc_id,
                "text": window_text[: settings.qwen3guard_max_chars],
                "model": settings.qwen3guard_model_name,
            },
            timeout=settings.qwen3guard_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return [], 0.0, []
        safety = str(payload.get("safety") or "").strip().lower()
        categories = payload.get("categories") or []
        if isinstance(categories, str):
            categories = [item.strip() for item in categories.split(",") if item.strip()]
        categories = [str(item).strip() for item in categories if str(item).strip()]
        score = float(payload.get("score") or payload.get("confidence") or 0.0)
        content_categories = [item for item in categories if not _is_privacy_only_guard_label(item)]
        if categories and not content_categories:
            return [], score, []
        if safety == "safe" and not content_categories:
            return [], score, []
        labels = [f"content.qwen3guard.{safety}" if safety else "content.qwen3guard"]
        labels.extend(content_categories)
        return labels, score, ["qwen3guard"]
    except Exception as exc:
        logger.warning("Qwen3Guard window recall failed for %s: %s", doc_id, exc)
        return [], 0.0, []


def run(
    ingest_units: list[IngestUnit],
    document_contexts: list[DocumentContextRecord],
    settings: Settings | None = None,
) -> list[ContentCandidateWindowRecord]:
    settings = settings or get_settings()
    context_by_doc = {item.doc_id: item for item in document_contexts}
    records: list[ContentCandidateWindowRecord] = []

    for unit in ingest_units:
        windows = _merge_short_windows(_sentence_windows(unit.text), settings.content_candidate_window_max_chars)
        context = context_by_doc.get(unit.doc_id)
        selected_labels: list[str] = []
        rule_hits = recall_content_rules(unit.text, settings.content_rules_path, selected_labels)
        for index, (start, end, text) in enumerate(windows):
            hits = _rule_hits_by_window(rule_hits, start, end)
            labels = [hit.policy_tag for hit in hits]
            score = max((hit.score for hit in hits), default=0.0)
            recall_sources = ["rule_engine"] if hits else []
            guard_labels, guard_score, guard_sources = _guard_window_labels(text, settings, unit.doc_id)
            labels.extend(guard_labels)
            score = max(score, guard_score)
            recall_sources.extend(guard_sources)
            if not labels:
                continue
            records.append(
                ContentCandidateWindowRecord(
                    run_id=unit.run_id,
                    doc_id=unit.doc_id,
                    start=start,
                    end=end,
                    text=text,
                    candidate_labels=sorted(dict.fromkeys(labels)),
                    candidate_score=round(score, 4),
                    recall_sources=sorted(dict.fromkeys(recall_sources)),
                    rule_hits=[hit.as_dict() for hit in hits],
                    metadata={
                        "window_index": index,
                        "document_type": context.document_type if context else "",
                        "scene_type": context.scene_type if context else "",
                    },
                )
            )

    logger.info("Content candidate window recall completed: %d windows", len(records))
    return records

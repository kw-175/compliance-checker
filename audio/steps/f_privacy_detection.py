"""
Step F: privacy detection and redaction span generation.
"""

from __future__ import annotations

import logging

from audio.adapters import pii_service_adapter
from audio.config.settings import Settings
from audio.models.schemas import DedupTranscriptUnit, PIIEntity, PrivacyResult, RedactionSpan, TranscriptUnit
from audio.steps.pii_local_engine import detect as local_detect

logger = logging.getLogger(__name__)

_REPLACEMENTS = {
    "PERSON": "<PERSON>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "CN_PHONE_NUMBER": "<PHONE>",
    "US_SSN": "<SSN>",
    "CN_ID_CARD": "<ID_CARD>",
    "ID_CARD": "<ID_CARD>",
    "PASSPORT": "<PASSPORT>",
    "DRIVER_LICENSE": "<DRIVER_LICENSE>",
    "CREDIT_CARD": "<CREDIT_CARD>",
    "BANK_ACCOUNT": "<BANK_ACCOUNT>",
    "STUDENT_ID": "<STUDENT_ID>",
    "WECHAT_ID": "<WECHAT_ID>",
    "QQ_NUMBER": "<QQ>",
    "PARENT_CONTACT": "<PARENT_CONTACT>",
    "LICENSE_PLATE": "<LICENSE_PLATE>",
    "IP_ADDRESS": "<IP_ADDRESS>",
    "URL": "<URL>",
    "LOCATION": "<LOCATION>",
    "ORGANIZATION": "<ORGANIZATION>",
}


def _to_time(unit: DedupTranscriptUnit | TranscriptUnit, start: int, end: int) -> tuple[float, float]:
    if not unit.text:
        return unit.start_time, unit.end_time
    duration = max(unit.end_time - unit.start_time, 0.0)
    text_length = max(len(unit.text), 1)
    start_ratio = max(0.0, min(1.0, start / text_length))
    end_ratio = max(0.0, min(1.0, end / text_length))
    return (
        unit.start_time + duration * start_ratio,
        unit.start_time + duration * end_ratio,
    )


def _redact_text(text: str, entities: list[PIIEntity]) -> str:
    if not entities:
        return text
    parts: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: (item.start, item.end)):
        if entity.start < cursor:
            continue
        parts.append(text[cursor:entity.start])
        parts.append(_REPLACEMENTS.get(entity.entity_type, "<REDACTED>"))
        cursor = entity.end
    parts.append(text[cursor:])
    return "".join(parts)


def run(
    units: list[DedupTranscriptUnit] | list[TranscriptUnit],
    settings: Settings | None = None,
    target_entity_types: list[str] | None = None,
) -> tuple[list[PrivacyResult], list[RedactionSpan]]:
    if settings is None:
        from audio.config.settings import get_settings

        settings = get_settings()

    results: list[PrivacyResult] = []
    spans: list[RedactionSpan] = []
    target_entities = [str(item).strip().upper() for item in (target_entity_types or []) if str(item).strip()]
    for unit in units:
        if getattr(unit, "is_duplicate", False):
            continue

        language = getattr(unit, "language", "")
        try:
            if target_entities:
                detection = pii_service_adapter.detect(unit.text, settings, language=language, target_entity_types=target_entities)
            else:
                detection = pii_service_adapter.detect(unit.text, settings, language=language)
        except Exception as exc:
            logger.warning("PII endpoint failed for %s, using local PII engine: %s", unit.unit_id, exc)
            detection = None
        if detection is None:
            detection = local_detect(unit.text, settings, language=language, target_entity_types=target_entities)
        entities = detection.entities
        redacted = _redact_text(unit.text, entities)

        results.append(
            PrivacyResult(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                original_text=unit.text,
                redacted_text=redacted,
                pii_entities=entities,
                pii_count=len(entities),
                provider_name=detection.provider_name,
                provider_version=detection.provider_version,
                is_degraded=detection.is_degraded,
            )
        )

        for entity in entities:
            start_time, end_time = _to_time(unit, entity.start, entity.end)
            spans.append(
                RedactionSpan(
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    start_time=start_time,
                    end_time=end_time,
                    entity_type=entity.entity_type,
                    original_text=entity.original_text,
                    replacement=_REPLACEMENTS.get(entity.entity_type, "<REDACTED>"),
                )
            )
    return results, spans

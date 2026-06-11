"""
HTTP client for the audio PII service.
"""

from __future__ import annotations

from audio.config.settings import Settings
from audio.models.schemas import PIIEntity
from audio.steps.pii_local_engine import LocalPIIDetection


def _entity_from_row(text: str, row: dict) -> PIIEntity | None:
    try:
        start = int(row["start"])
        end = int(row["end"])
        score = float(row.get("score", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not 0 <= start < end <= len(text):
        return None
    return PIIEntity(
        entity_type=str(row.get("entity_type", "")),
        start=start,
        end=end,
        score=round(max(0.0, min(1.0, score)), 4),
        original_text=text[start:end][:100],
    )


def detect(text: str, settings: Settings, language: str = "", target_entity_types: list[str] | None = None) -> LocalPIIDetection | None:
    if not settings.pii_endpoint:
        return None
    target_entities = {str(item).strip().upper() for item in (target_entity_types or []) if str(item).strip()}

    import httpx

    response = httpx.post(
        settings.pii_endpoint,
        json={
            "text": text,
            "language": language or "en",
            "score_threshold": settings.pii_score_threshold,
            "target_entity_types": sorted(target_entities),
        },
        timeout=settings.pii_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError("PII endpoint must return a JSON list.")

    entities: list[PIIEntity] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        entity = _entity_from_row(text, row)
        if entity is not None:
            entities.append(entity)
    if target_entities:
        entities = [entity for entity in entities if str(entity.entity_type).strip().upper() in target_entities]

    return LocalPIIDetection(
        entities=entities,
        provider_name="pii_endpoint",
        provider_version=settings.pii_endpoint,
        is_degraded=False,
    )

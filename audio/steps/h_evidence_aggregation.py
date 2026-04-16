"""
Step H: privacy and safety evidence aggregation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from audio.models.schemas import EvidenceBundle, PrivacyResult, SafetyResult, TranscriptEvidence, TranscriptUnit


def run(
    units: list[TranscriptUnit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    pipeline_run_id: str,
    extension_events: list[dict[str, Any]] | None = None,
) -> EvidenceBundle:
    privacy_by_unit = {item.unit_id: item for item in privacy_results}
    safety_by_unit = {item.unit_id: item for item in safety_results}

    evidence_units: list[TranscriptEvidence] = []
    for unit in units:
        evidence_units.append(
            TranscriptEvidence(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                text=unit.text,
                speaker_id=unit.speaker_id,
                privacy=privacy_by_unit.get(unit.unit_id),
                safety=safety_by_unit.get(unit.unit_id),
            )
        )

    safety_counts = Counter(item.safety_level.value for item in safety_results)
    return EvidenceBundle(
        pipeline_run_id=pipeline_run_id,
        transcript_units=evidence_units,
        summary={
            "total_units": len(units),
            "distinct_sources": len({unit.source_id for unit in units}),
            "total_pii_entities": sum(item.pii_count for item in privacy_results),
            "unsafe_units": safety_counts.get("unsafe", 0),
            "controversial_units": safety_counts.get("controversial", 0),
            "safe_units": safety_counts.get("safe", 0),
            "extension_event_count": len(extension_events or []),
        },
        degrade_events=[],
        trust_level="full",
    )

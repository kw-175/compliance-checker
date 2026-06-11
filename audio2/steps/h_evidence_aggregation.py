"""
Step H: privacy and safety evidence aggregation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from audio.models.schemas import AudioHardCaseResult, EvidenceBundle, PrivacyResult, SafetyResult, TranscriptEvidence, TranscriptUnit


def run(
    units: list[TranscriptUnit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    pipeline_run_id: str,
    hard_case_results: list[AudioHardCaseResult] | None = None,
    extension_events: list[dict[str, Any]] | None = None,
) -> EvidenceBundle:
    privacy_by_unit = {item.unit_id: item for item in privacy_results}
    safety_by_unit = {item.unit_id: item for item in safety_results}
    hard_case_by_unit = {item.unit_id: item for item in (hard_case_results or [])}

    evidence_units: list[TranscriptEvidence] = []
    degrade_events: list[dict[str, Any]] = []
    for unit in units:
        hard_case = hard_case_by_unit.get(unit.unit_id)
        unit_degrade_events: list[dict[str, Any]] = []
        if hard_case and hard_case.is_degraded:
            event = {
                "unit_id": unit.unit_id,
                "source": "hard_case_adjudication",
                "provider": hard_case.provider_name,
                "notes": hard_case.notes,
            }
            unit_degrade_events.append(event)
            degrade_events.append(event)
        evidence_units.append(
            TranscriptEvidence(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                text=unit.text,
                speaker_id=unit.speaker_id,
                privacy=privacy_by_unit.get(unit.unit_id),
                safety=safety_by_unit.get(unit.unit_id),
                hard_case=hard_case,
                degrade_events=unit_degrade_events,
                trust_level="degraded" if unit_degrade_events else "full",
            )
        )

    safety_counts = Counter(item.safety_level.value for item in safety_results)
    hard_case_count = len(hard_case_results or [])
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
            "hard_case_units": hard_case_count,
            "hard_case_degraded_units": sum(1 for item in (hard_case_results or []) if item.is_degraded),
            "extension_event_count": len(extension_events or []),
        },
        degrade_events=degrade_events,
        trust_level="degraded" if degrade_events else "full",
    )

"""
Step L: release packaging.
"""

from __future__ import annotations

from audio.models.schemas import EvidenceBundle, NormalizedAudioRecord, PolicyDecision, RedactedAudioRecord, ReleasePackage


def run(
    pipeline_run_id: str,
    original_audio: list[NormalizedAudioRecord],
    evidence_bundle: EvidenceBundle,
    decision: PolicyDecision,
    redacted_audio: list[RedactedAudioRecord],
    audit_metadata: dict,
) -> ReleasePackage:
    summary = evidence_bundle.summary
    return ReleasePackage(
        pipeline_run_id=pipeline_run_id,
        original_audio=original_audio,
        transcript_summary={
            "total_units": summary.get("total_units", 0),
            "duplicate_units": summary.get("duplicate_units", 0),
            "unsafe_units": summary.get("unsafe_units", 0),
            "controversial_units": summary.get("controversial_units", 0),
            "safe_units": summary.get("safe_units", 0),
        },
        decision=decision,
        evidence_summary=summary,
        redacted_audio=redacted_audio,
        audit_metadata=audit_metadata,
    )

"""
Step H: evidence aggregation.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from audio.models.schemas import ComplianceHit, DedupTranscriptUnit, EvidenceBundle, KeywordHit, PrivacyResult, RegexHit, SafetyResult, SecretHit, TranscriptEvidence


def run(
    units: list[DedupTranscriptUnit],
    secret_hits: list[SecretHit],
    compliance_hits: list[ComplianceHit],
    keyword_hits: list[KeywordHit],
    regex_hits: list[RegexHit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    pipeline_run_id: str,
) -> EvidenceBundle:
    # 先构建索引结构，降低后续按 unit/source 关联成本。
    secrets_by_source: dict[str, list[SecretHit]] = defaultdict(list)
    for hit in secret_hits:
        secrets_by_source[hit.source_id].append(hit)

    compliance_by_source: dict[str, list[ComplianceHit]] = defaultdict(list)
    for hit in compliance_hits:
        compliance_by_source[hit.source_id].append(hit)

    keyword_by_unit: dict[str, list[KeywordHit]] = defaultdict(list)
    for hit in keyword_hits:
        keyword_by_unit[hit.unit_id].append(hit)

    regex_by_unit: dict[str, list[RegexHit]] = defaultdict(list)
    for hit in regex_hits:
        regex_by_unit[hit.unit_id].append(hit)

    privacy_by_unit = {item.unit_id: item for item in privacy_results}
    safety_by_unit = {item.unit_id: item for item in safety_results}

    evidence_units: list[TranscriptEvidence] = []
    for unit in units:
        # 将多路扫描结果汇总为单 unit 的完整证据视图。
        evidence_units.append(
            TranscriptEvidence(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                text=unit.text,
                speaker_id=unit.speaker_id,
                is_duplicate=unit.is_duplicate,
                secret_hits=secrets_by_source.get(unit.source_id, []),
                compliance_hits=compliance_by_source.get(unit.source_id, []),
                keyword_hits=keyword_by_unit.get(unit.unit_id, []),
                regex_hits=regex_by_unit.get(unit.unit_id, []),
                privacy=privacy_by_unit.get(unit.unit_id),
                safety=safety_by_unit.get(unit.unit_id),
            )
        )

    safety_counts = Counter(item.safety_level.value for item in safety_results)
    # summary 作为策略决策与审计报表的快速统计入口。
    return EvidenceBundle(
        pipeline_run_id=pipeline_run_id,
        transcript_units=evidence_units,
        summary={
            "total_units": len(units),
            "duplicate_units": sum(1 for unit in units if unit.is_duplicate),
            "distinct_sources": len({unit.source_id for unit in units}),
            "total_secret_hits": len(secret_hits),
            "total_compliance_hits": len(compliance_hits),
            "total_keyword_hits": len(keyword_hits),
            "total_regex_hits": len(regex_hits),
            "total_pii_entities": sum(item.pii_count for item in privacy_results),
            "unsafe_units": safety_counts.get("unsafe", 0),
            "controversial_units": safety_counts.get("controversial", 0),
            "safe_units": safety_counts.get("safe", 0),
        },
    )

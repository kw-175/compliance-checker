"""
Step D: transcript deduplication.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from audio.models.schemas import DedupMapEntry, DedupTranscriptUnit, TranscriptUnit


def _normalize_text(text: str) -> str:
    # 去重前统一大小写与空白，降低格式差异影响。
    return " ".join(text.strip().lower().split())


def _token_jaccard(left: str, right: str) -> float:
    # 基于词集合的 Jaccard 相似度，衡量近似重复。
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def run(units: list[TranscriptUnit], settings=None) -> tuple[list[DedupTranscriptUnit], list[DedupMapEntry]]:
    # exact_seen: 精确去重；normalized_by_source: 同源近似去重。
    exact_seen: dict[str, str] = {}
    normalized_by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    threshold = float(getattr(settings, "dedup_threshold", 0.8) or 0.8)

    deduped: list[DedupTranscriptUnit] = []
    mapping: list[DedupMapEntry] = []

    for unit in units:
        normalized = _normalize_text(unit.text)
        # 先做哈希精确匹配，再做阈值近似匹配。
        fingerprint = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        duplicate_of = exact_seen.get(fingerprint)
        similarity = 1.0 if duplicate_of else 0.0

        if duplicate_of is None and normalized:
            for candidate_id, candidate_text in normalized_by_source[unit.source_id]:
                similarity = _token_jaccard(normalized, candidate_text)
                if similarity >= threshold:
                    duplicate_of = candidate_id
                    break

        is_duplicate = duplicate_of is not None
        if not is_duplicate:
            exact_seen[fingerprint] = unit.unit_id
            if normalized:
                normalized_by_source[unit.source_id].append((unit.unit_id, normalized))
        else:
            # 记录重复映射，供审计与回溯使用。
            mapping.append(
                DedupMapEntry(
                    unit_id=unit.unit_id,
                    duplicate_of=duplicate_of,
                    jaccard_similarity=similarity,
                )
            )

        deduped.append(
            # 无论是否重复都保留记录，供后续步骤统一遍历。
            DedupTranscriptUnit(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                start_time=unit.start_time,
                end_time=unit.end_time,
                speaker_id=unit.speaker_id,
                text=unit.text,
                confidence=unit.confidence,
                engine_name=unit.engine_name,
                is_duplicate=is_duplicate,
                duplicate_of=duplicate_of,
            )
        )

    return deduped, mapping

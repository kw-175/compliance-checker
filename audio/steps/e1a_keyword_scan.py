"""
Step E1a: keyword scan.
"""

from __future__ import annotations

from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import DedupTranscriptUnit, KeywordHit


def _load_keywords(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _context(text: str, start: int, end: int) -> str:
    left = max(0, start - 40)
    right = min(len(text), end + 40)
    return text[left:right]


def run(units: list[DedupTranscriptUnit], settings: Settings | None = None) -> list[KeywordHit]:
    if settings is None:
        from audio.config.settings import get_settings
        settings = get_settings()
    keywords = _load_keywords(settings.keywords_file)
    hits: list[KeywordHit] = []
    for unit in units:
        if unit.is_duplicate:
            continue
        text_lower = unit.text.lower()
        for keyword in keywords:
            needle = keyword.lower()
            start = 0
            while True:
                index = text_lower.find(needle, start)
                if index == -1:
                    break
                end = index + len(needle)
                hits.append(KeywordHit(unit_id=unit.unit_id, keyword=keyword, start_pos=index, end_pos=end, context=_context(unit.text, index, end)))
                start = end
    return hits

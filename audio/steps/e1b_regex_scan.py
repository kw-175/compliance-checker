"""
Step E1b: regex scan.
"""

from __future__ import annotations

import re

from audio.config.settings import Settings
from audio.models.schemas import DedupTranscriptUnit, RegexHit
from audio.steps import load_yaml


def _context(text: str, start: int, end: int) -> str:
    left = max(0, start - 40)
    right = min(len(text), end + 40)
    return text[left:right]


def run(units: list[DedupTranscriptUnit], settings: Settings | None = None) -> list[RegexHit]:
    if settings is None:
        from audio.config.settings import get_settings
        settings = get_settings()
    patterns = load_yaml(settings.patterns_file)
    compiled = [(name, re.compile(pattern)) for name, pattern in patterns.items()]
    hits: list[RegexHit] = []
    for unit in units:
        if unit.is_duplicate:
            continue
        for name, pattern in compiled:
            for match in pattern.finditer(unit.text):
                hits.append(
                    RegexHit(
                        unit_id=unit.unit_id,
                        pattern_name=name,
                        pattern=pattern.pattern,
                        matched_text=match.group()[:200],
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=_context(unit.text, match.start(), match.end()),
                    )
                )
    return hits

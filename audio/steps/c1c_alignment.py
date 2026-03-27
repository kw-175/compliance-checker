"""
Step C1c: alignment.
"""

from __future__ import annotations

from audio.models.schemas import ASRSegment


def run(segments: list[ASRSegment]) -> list[ASRSegment]:
    return [segment.model_copy() for segment in segments]

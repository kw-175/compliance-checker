"""
Step C1c: alignment.
"""

from __future__ import annotations

from audio.models.schemas import ASRSegment


def run(segments: list[ASRSegment]) -> list[ASRSegment]:
    # 当前版本为直通复制；保留该步骤用于后续接入精细对齐算法。
    return [segment.model_copy() for segment in segments]

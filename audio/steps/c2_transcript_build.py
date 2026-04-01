"""
Step C2: transcript build.
"""

from __future__ import annotations

from collections import defaultdict

from audio.models.schemas import ASRSegment, SpeakerSegment, TranscriptUnit


def _pick_speaker(segment: ASRSegment, speakers: list[SpeakerSegment]) -> str:
    # 通过时间重叠最大原则为 ASR 片段选择说话人。
    best_speaker = "speaker_0"
    best_overlap = 0.0
    for speaker in speakers:
        overlap = min(segment.end_time, speaker.end_time) - max(segment.start_time, speaker.start_time)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker.speaker_id
    return best_speaker


def run(asr_segments: list[ASRSegment], speaker_segments: list[SpeakerSegment]) -> list[TranscriptUnit]:
    # 先按 source 分组说话人片段，便于后续快速匹配。
    speakers_by_source: dict[str, list[SpeakerSegment]] = defaultdict(list)
    for speaker in sorted(speaker_segments, key=lambda item: (item.source_id, item.start_time, item.end_time)):
        speakers_by_source[speaker.source_id].append(speaker)

    units: list[TranscriptUnit] = []
    for segment in sorted(asr_segments, key=lambda item: (item.source_id, item.start_time, item.end_time)):
        # 将 ASR 片段规范化为统一 transcript unit 结构。
        units.append(
            TranscriptUnit(
                source_id=segment.source_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                speaker_id=_pick_speaker(segment, speakers_by_source.get(segment.source_id, [])),
                text=segment.text,
                confidence=segment.confidence,
                engine_name=segment.engine_name,
                language=segment.language,
                metadata=segment.metadata,
            )
        )
    return units

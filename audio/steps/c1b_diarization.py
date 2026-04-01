"""
Step C1b: diarization.
"""

from __future__ import annotations

import logging

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, SpeakerSegment

logger = logging.getLogger(__name__)


def _fallback_segments(record: NormalizedAudioRecord) -> list[SpeakerSegment]:
    # 无法分离说话人时，整段归为默认说话人。
    return [
        SpeakerSegment(
            source_id=record.source_id,
            speaker_id="speaker_0",
            start_time=0.0,
            end_time=record.duration_seconds,
            confidence=0.0,
            engine_name="fallback",
        )
    ]


def _run_pyannote(record: NormalizedAudioRecord, settings: Settings) -> list[SpeakerSegment]:
    # 主路径：调用 pyannote 进行说话人分离。
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio unavailable") from exc
    pipeline = Pipeline.from_pretrained(settings.pyannote_model)
    diarization = pipeline(record.normalized_path)
    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 将 pyannote 片段统一转换为 SpeakerSegment。
        segments.append(
            SpeakerSegment(
                source_id=record.source_id,
                speaker_id=str(speaker),
                start_time=float(turn.start),
                end_time=float(turn.end),
                confidence=1.0,
                engine_name="pyannote",
            )
        )
    return segments


def run(records: list[NormalizedAudioRecord], settings: Settings) -> list[SpeakerSegment]:
    # 每条音频独立处理，失败后回退到单说话人占位。
    all_segments: list[SpeakerSegment] = []
    for record in records:
        segments: list[SpeakerSegment] = []
        if settings.pyannote_enabled:
            try:
                segments = _run_pyannote(record, settings)
            except Exception as exc:
                logger.warning("pyannote diarization failed for %s: %s", record.source_id, exc)
        if not segments:
            segments = _fallback_segments(record)
        all_segments.extend(segments)
    return all_segments

"""
Step K: audio redaction.
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, RedactedAudioRecord, RedactionSpan, RenderStrategy
from audio.steps import run_command

logger = logging.getLogger(__name__)


def _select_strategy(settings: Settings) -> RenderStrategy:
    raw = (settings.redaction_strategy or "silence").lower()
    if raw == RenderStrategy.BEEP.value:
        return RenderStrategy.BEEP
    if raw == RenderStrategy.COPY.value:
        return RenderStrategy.COPY
    return RenderStrategy.SILENCE


def _render_silence(record: NormalizedAudioRecord, spans: list[RedactionSpan], target_path: Path, settings: Settings) -> bool:
    filters = [f"volume=enable='between(t,{span.start_time},{span.end_time})':volume=0" for span in spans]
    result = run_command(
        [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            record.normalized_path,
            "-af",
            ",".join(filters),
            str(target_path),
        ],
        timeout=600,
    )
    return result is not None


def _render_beep(record: NormalizedAudioRecord, spans: list[RedactionSpan], target_path: Path, settings: Settings) -> bool:
    duration = max(record.duration_seconds, max((span.end_time for span in spans), default=0.0))
    expr = "+".join([f"between(t,{span.start_time},{span.end_time})" for span in spans])
    filter_complex = (
        f"[0:a]volume=enable='{expr}':volume=0[main];"
        f"sine=f={settings.beep_frequency}:sample_rate=16000:d={duration},volume={settings.beep_volume},"
        f"aselect='{expr}',asetpts=N/SR/TB[beep];"
        "[main][beep]amix=inputs=2:normalize=0[out]"
    )
    result = run_command(
        [
            settings.ffmpeg_bin,
            "-y",
            "-i",
            record.normalized_path,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            str(target_path),
        ],
        timeout=600,
    )
    return result is not None


def run(records: list[NormalizedAudioRecord], spans: list[RedactionSpan], settings: Settings, output_dir: Path) -> list[RedactedAudioRecord]:
    redaction_dir = output_dir / "redacted_audio"
    redaction_dir.mkdir(parents=True, exist_ok=True)
    spans_by_source: dict[str, list[RedactionSpan]] = defaultdict(list)
    for span in spans:
        spans_by_source[span.source_id].append(span)

    outputs: list[RedactedAudioRecord] = []
    requested_strategy = _select_strategy(settings)
    for record in records:
        target_path = redaction_dir / Path(record.normalized_path).name
        source_spans = sorted(spans_by_source.get(record.source_id, []), key=lambda item: item.start_time)
        if requested_strategy == RenderStrategy.COPY or not source_spans:
            shutil.copy2(record.normalized_path, target_path)
            strategy = RenderStrategy.COPY
        else:
            render_ok = False
            if requested_strategy == RenderStrategy.BEEP:
                render_ok = _render_beep(record, source_spans, target_path, settings)
                strategy = RenderStrategy.BEEP if render_ok else RenderStrategy.COPY
            else:
                render_ok = _render_silence(record, source_spans, target_path, settings)
                strategy = RenderStrategy.SILENCE if render_ok else RenderStrategy.COPY

            if not render_ok:
                logger.warning("Audio redaction render failed for %s, fallback to copy", record.source_id)
                shutil.copy2(record.normalized_path, target_path)
                strategy = RenderStrategy.COPY
        outputs.append(
            RedactedAudioRecord(
                source_id=record.source_id,
                original_audio_path=record.normalized_path,
                redacted_audio_path=str(target_path.resolve()),
                render_strategy=strategy,
                span_count=len(source_spans),
                duration_seconds=record.duration_seconds,
                metadata={
                    "engine_name": "ffmpeg" if strategy != RenderStrategy.COPY else "copy_fallback",
                    "requested_strategy": requested_strategy.value,
                },
            )
        )
    return outputs

"""
Step C1b: diarization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, SpeakerSegment
from audio.steps import load_jsonl

logger = logging.getLogger(__name__)


def _record_metadata(record: NormalizedAudioRecord) -> dict[str, Any]:
    return record.metadata if isinstance(record.metadata, dict) else {}


def _record_ids(record: NormalizedAudioRecord) -> set[str]:
    metadata = _record_metadata(record)
    return {
        str(value).strip()
        for value in (
            record.source_id,
            metadata.get("audio_id"),
            metadata.get("upstream_source_id"),
        )
        if str(value or "").strip()
    }


def _row_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(row.get(key)).strip()
        for key in ("audio_id", "source_id")
        if str(row.get(key, "")).strip()
    }


def _row_matches_record(row: dict[str, Any], record: NormalizedAudioRecord) -> bool:
    ids = _row_ids(row)
    if ids:
        return bool(ids & _record_ids(record))
    for key in ("clean_audio_path", "normalized_path", "audio_path"):
        raw_ref = str(row.get(key, "")).strip()
        if raw_ref and Path(raw_ref).name == Path(record.normalized_path).name:
            return True
    return False


def _speaker_sidecar_path(record: NormalizedAudioRecord) -> Path | None:
    metadata = _record_metadata(record)
    sidecar_paths = metadata.get("sidecar_paths")
    if isinstance(sidecar_paths, dict) and sidecar_paths.get("speaker_segments.jsonl"):
        return Path(str(sidecar_paths["speaker_segments.jsonl"]))
    if metadata.get("speaker_segments_path"):
        return Path(str(metadata["speaker_segments_path"]))
    return None


def _load_sidecar_segments(record: NormalizedAudioRecord) -> list[SpeakerSegment]:
    sidecar_path = _speaker_sidecar_path(record)
    if sidecar_path is None or not sidecar_path.exists():
        return []

    segments: list[SpeakerSegment] = []
    for row in load_jsonl(sidecar_path):
        if not _row_matches_record(row, record):
            continue
        try:
            segments.append(
                SpeakerSegment(
                    source_id=record.source_id,
                    speaker_id=str(row.get("speaker_id") or row.get("speaker") or "speaker_0"),
                    start_time=float(row.get("start_time", row.get("start", 0.0)) or 0.0),
                    end_time=float(row.get("end_time", row.get("end", record.duration_seconds)) or record.duration_seconds),
                    confidence=float(row.get("confidence", row.get("score", 0.0)) or 0.0),
                    engine_name=str(row.get("engine_name") or row.get("engine") or "sidecar"),
                )
            )
        except (TypeError, ValueError):
            logger.warning("Skipping malformed speaker sidecar row in %s", sidecar_path)
    return segments


def _fallback_segments(record: NormalizedAudioRecord) -> list[SpeakerSegment]:
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
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError("pyannote.audio unavailable") from exc
    pipeline = Pipeline.from_pretrained(settings.pyannote_model)
    diarization = pipeline(record.normalized_path)
    segments: list[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
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
    all_segments: list[SpeakerSegment] = []
    for record in records:
        segments = _load_sidecar_segments(record)
        if segments:
            all_segments.extend(segments)
            continue

        if settings.pyannote_enabled:
            try:
                segments = _run_pyannote(record, settings)
            except Exception as exc:
                logger.warning("pyannote diarization failed for %s: %s", record.source_id, exc)
        if not segments:
            segments = _fallback_segments(record)
        all_segments.extend(segments)
    return all_segments

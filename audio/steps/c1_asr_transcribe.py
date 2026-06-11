"""
Step C1: ASR transcription.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from audio.adapters import qwen_asr_adapter
from audio.config.settings import Settings
from audio.models.schemas import ASRSegment, NormalizedAudioRecord
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


def _row_matches_record(
    row: dict[str, Any],
    record: NormalizedAudioRecord,
    *,
    require_identifier: bool,
) -> bool:
    ids = _row_ids(row)
    if ids:
        return bool(ids & _record_ids(record))
    if require_identifier:
        return False

    for key in ("clean_audio_path", "normalized_path", "audio_path"):
        raw_ref = str(row.get(key, "")).strip()
        if raw_ref and Path(raw_ref).name == Path(record.normalized_path).name:
            return True
    return True


def _candidate_transcript_paths(record: NormalizedAudioRecord) -> list[tuple[Path, bool]]:
    metadata = _record_metadata(record)
    paths: list[tuple[Path, bool]] = []

    sidecar_paths = metadata.get("sidecar_paths")
    if isinstance(sidecar_paths, dict) and sidecar_paths.get("transcript_segments.jsonl"):
        paths.append((Path(str(sidecar_paths["transcript_segments.jsonl"])), True))
    if isinstance(sidecar_paths, dict) and sidecar_paths.get("transcript_segments.json"):
        paths.append((Path(str(sidecar_paths["transcript_segments.json"])), True))
    if metadata.get("transcript_segments_path"):
        paths.append((Path(str(metadata["transcript_segments_path"])), True))

    original = Path(record.original_path)
    paths.extend(
        [
            (original.with_suffix(".transcript.jsonl"), False),
            (original.with_suffix(".transcript.json"), False),
            (original.with_name(f"{original.stem}_transcript.jsonl"), False),
            (original.with_name(f"{original.stem}_transcript.json"), False),
            (original.with_name("sample_transcript.jsonl"), False),
            (original.with_name("sample_transcript.json"), False),
        ]
    )

    unique: list[tuple[Path, bool]] = []
    seen: set[str] = set()
    for path, require_match in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, require_match))
    return unique


def _segment_payload(row: dict[str, Any], record: NormalizedAudioRecord, sidecar_path: Path) -> dict[str, Any]:
    text = str(row.get("text") or row.get("transcript") or row.get("asr_text") or "")
    end_time = row.get("end_time", row.get("end", record.duration_seconds))
    payload: dict[str, Any] = {
        "source_id": record.source_id,
        "start_time": float(row.get("start_time", row.get("start", 0.0)) or 0.0),
        "end_time": float(end_time or record.duration_seconds),
        "text": text,
        "confidence": float(row.get("confidence", row.get("score", 0.0)) or 0.0),
        "engine_name": str(row.get("engine_name") or row.get("engine") or "sidecar"),
        "language": str(row.get("language", "")),
        "metadata": {
            "sidecar_path": str(sidecar_path),
            "sidecar_audio_id": str(row.get("audio_id", "")),
            "sidecar_source_id": str(row.get("source_id", "")),
            "sidecar_segment_id": str(row.get("segment_id", "")),
            "sidecar_record": row,
        },
    }
    if row.get("segment_id"):
        payload["segment_id"] = str(row.get("segment_id"))
    return payload


def _load_json_sidecar(path: Path, record: NormalizedAudioRecord) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Skipping malformed transcript JSON sidecar: %s", path)
        return []

    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    if isinstance(payload, dict):
        segments = payload.get("segments")
        if isinstance(segments, list):
            enriched_rows: list[dict[str, Any]] = []
            root_language = payload.get("language", "")
            root_engine = payload.get("engine_name") or payload.get("engine") or "whisper_json"
            for row in segments:
                if not isinstance(row, dict):
                    continue
                enriched_row = dict(row)
                enriched_row.setdefault("language", root_language)
                enriched_row.setdefault("engine_name", root_engine)
                enriched_rows.append(enriched_row)
            return enriched_rows

        text = str(payload.get("text") or payload.get("transcript") or "").strip()
        if text:
            return [
                {
                    "segment_id": payload.get("segment_id") or f"{record.source_id}_seg_0",
                    "source_id": payload.get("source_id") or record.source_id,
                    "audio_id": payload.get("audio_id") or record.source_id,
                    "start_time": payload.get("start_time", 0.0),
                    "end_time": payload.get("end_time", payload.get("duration", record.duration_seconds)),
                    "text": text,
                    "confidence": payload.get("confidence", payload.get("avg_logprob", 0.0)),
                    "engine_name": payload.get("engine_name") or payload.get("engine") or "whisper_json",
                    "language": payload.get("language", ""),
                }
            ]

    logger.warning("Unsupported transcript JSON sidecar format: %s", path)
    return []


def _sidecar_rows(candidate: Path, record: NormalizedAudioRecord) -> list[dict[str, Any]]:
    if candidate.suffix.lower() == ".json":
        return _load_json_sidecar(candidate, record)
    return load_jsonl(candidate)


def _load_sidecar_segments(record: NormalizedAudioRecord) -> list[ASRSegment]:
    for candidate, require_match in _candidate_transcript_paths(record):
        if not candidate.exists():
            continue
        segments: list[ASRSegment] = []
        for row in _sidecar_rows(candidate, record):
            if not _row_matches_record(row, record, require_identifier=require_match):
                continue
            payload = _segment_payload(row, record, candidate)
            if payload["text"].strip():
                segments.append(ASRSegment(**payload))
        if segments:
            return segments
    return []


def _run_qwen_asr(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    return qwen_asr_adapter.transcribe(record, settings)


def _run_faster_whisper(record: NormalizedAudioRecord, settings: Settings) -> list[ASRSegment]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper unavailable") from exc

    model = WhisperModel(settings.faster_whisper_model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(record.normalized_path, vad_filter=True)
    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=float(segment.start),
            end_time=float(segment.end),
            text=str(segment.text).strip(),
            confidence=float(getattr(segment, "avg_logprob", 0.0) or 0.0),
            engine_name="faster-whisper",
            language=str(getattr(info, "language", "")),
        )
        for segment in segments
        if str(segment.text).strip()
    ]


def _fallback_segments(record: NormalizedAudioRecord) -> list[ASRSegment]:
    sidecar = _load_sidecar_segments(record)
    if sidecar:
        return sidecar
    return [
        ASRSegment(
            source_id=record.source_id,
            start_time=0.0,
            end_time=record.duration_seconds,
            text="Audio transcript unavailable",
            confidence=0.0,
            engine_name="fallback",
        )
    ]


def _should_allow_unavailable_fallback(settings: Settings) -> bool:
    return bool(getattr(settings, "asr_unavailable_fallback_enabled", False)) or not bool(getattr(settings, "asr_required", True))


def run(records: list[NormalizedAudioRecord], settings: Settings) -> list[ASRSegment]:
    all_segments: list[ASRSegment] = []
    for record in records:
        segments = _load_sidecar_segments(record)
        if segments:
            all_segments.extend(segments)
            continue

        if settings.qwen_asr_enabled:
            try:
                segments = _run_qwen_asr(record, settings)
            except Exception as exc:
                logger.warning("Qwen ASR failed for %s: %s", record.source_id, exc)
        if not segments and settings.faster_whisper_enabled:
            try:
                segments = _run_faster_whisper(record, settings)
            except Exception as exc:
                logger.warning("faster-whisper failed for %s: %s", record.source_id, exc)
        if not segments:
            if not _should_allow_unavailable_fallback(settings):
                raise RuntimeError(
                    "ASR transcript unavailable for "
                    f"{record.source_id}; refusing to run text compliance on placeholder transcript."
                )
            segments = _fallback_segments(record)
        all_segments.extend(segments)
    return all_segments

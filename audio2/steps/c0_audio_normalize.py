"""
Step C0: audio normalization.
"""

from __future__ import annotations

import json
import logging
import shutil
import wave
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, SourceProfile, SourceType
from audio.steps import load_yaml, run_command

logger = logging.getLogger(__name__)


def _probe_with_ffprobe(path: Path, settings: Settings) -> dict[str, object]:
    result = run_command(
        [
            settings.ffprobe_bin,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(path),
        ]
    )
    if result is None:
        return {}
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
        fmt = payload.get("format", {})
        return {
            "sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
            "channels": int(audio_stream.get("channels", 0) or 0),
            "codec": str(audio_stream.get("codec_name", "")),
            "duration_seconds": float(fmt.get("duration", 0.0) or 0.0),
        }
    except Exception:
        logger.debug("ffprobe output parse failed for %s", path)
        return {}


def _probe_with_wave(path: Path) -> dict[str, object]:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return {
                "sample_rate": rate,
                "channels": handle.getnchannels(),
                "codec": "pcm_s16le",
                "duration_seconds": frames / rate if rate else 0.0,
            }
    except Exception:
        return {"sample_rate": 0, "channels": 0, "codec": "", "duration_seconds": 0.0}


def _nested_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    containers = (
        _nested_dict(metadata.get("manifest_record")),
        _nested_dict(metadata.get("quality_record")),
        metadata,
    )
    for container in containers:
        for key in keys:
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def _metadata_int(metadata: dict[str, Any], probe: dict[str, object], keys: tuple[str, ...]) -> int:
    value = _first_value(metadata, keys)
    if value in (None, ""):
        value = probe.get(keys[0], 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _metadata_float(metadata: dict[str, Any], probe: dict[str, object], keys: tuple[str, ...]) -> float:
    value = _first_value(metadata, keys)
    if value in (None, ""):
        value = probe.get(keys[0], 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metadata_str(metadata: dict[str, Any], probe: dict[str, object], keys: tuple[str, ...]) -> str:
    value = _first_value(metadata, keys)
    if value in (None, ""):
        value = probe.get(keys[0], "")
    return str(value or "")


def _probe_audio(path: Path, settings: Settings) -> dict[str, object]:
    metadata = _probe_with_ffprobe(path, settings)
    if not metadata:
        metadata = _probe_with_wave(path)
    return metadata


def _from_cleaned_package(profile: SourceProfile, settings: Settings) -> NormalizedAudioRecord:
    path = Path(profile.path)
    metadata = dict(profile.metadata or {})
    probe = _probe_audio(path, settings)
    metadata["normalization_skipped"] = True
    metadata["normalization_source"] = "cleaned_audio_package"

    return NormalizedAudioRecord(
        source_id=profile.source_id,
        original_path=profile.path,
        normalized_path=str(path.resolve()),
        sample_rate=_metadata_int(metadata, probe, ("sample_rate", "sampling_rate")),
        channels=_metadata_int(metadata, probe, ("channels", "channel_count")),
        codec=_metadata_str(metadata, probe, ("codec", "audio_codec")),
        duration_seconds=_metadata_float(metadata, probe, ("duration_seconds", "duration")),
        engine_name="cleaned_package",
        metadata=metadata,
    )


def run(profiles: list[SourceProfile], settings: Settings, output_dir: Path) -> list[NormalizedAudioRecord]:
    profiles_cfg = load_yaml(settings.ffmpeg_profiles_file)
    normalize_cfg = profiles_cfg.get("normalize", {})
    target_rate = int(normalize_cfg.get("sample_rate", 16000))
    target_channels = int(normalize_cfg.get("channels", 1))
    target_codec = str(normalize_cfg.get("codec", "pcm_s16le"))
    target_ext = str(normalize_cfg.get("extension", "wav"))

    records: list[NormalizedAudioRecord] = []
    normalized_dir = output_dir / "normalized_audio"
    normalized_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        if profile.source_type != SourceType.AUDIO:
            continue
        if profile.metadata.get("cleaned_audio_package"):
            records.append(_from_cleaned_package(profile, settings))
            continue

        source_path = Path(profile.path)
        output_path = normalized_dir / f"{profile.source_id}.{target_ext}"
        result = run_command(
            [
                settings.ffmpeg_bin,
                "-y",
                "-i",
                str(source_path),
                "-ar",
                str(target_rate),
                "-ac",
                str(target_channels),
                "-c:a",
                target_codec,
                str(output_path),
            ],
            timeout=600,
        )
        if result is None:
            shutil.copy2(source_path, output_path)
            engine_name = "copy_fallback"
        else:
            engine_name = "ffmpeg"

        metadata = _probe_audio(output_path, settings)
        records.append(
            NormalizedAudioRecord(
                source_id=profile.source_id,
                original_path=profile.path,
                normalized_path=str(output_path.resolve()),
                sample_rate=int(metadata.get("sample_rate", 0)),
                channels=int(metadata.get("channels", 0)),
                codec=str(metadata.get("codec", "")),
                duration_seconds=float(metadata.get("duration_seconds", 0.0)),
                engine_name=engine_name,
                metadata=profile.metadata,
            )
        )

    return records

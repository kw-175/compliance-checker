"""
Step C0: audio normalization.
"""

from __future__ import annotations

import json
import logging
import shutil
import wave
from pathlib import Path

from audio.config.settings import Settings
from audio.models.schemas import NormalizedAudioRecord, SourceProfile, SourceType
from audio.steps import load_yaml, run_command

logger = logging.getLogger(__name__)


def _probe_with_ffprobe(path: Path, settings: Settings) -> dict[str, object]:
    # 优先使用 ffprobe 获取精确音频参数。
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
        # 解析失败时返回空，由上层继续尝试 wave 探测。
        logger.debug("ffprobe output parse failed for %s", path)
        return {}


def _probe_with_wave(path: Path) -> dict[str, object]:
    # 纯 Python 兜底探测，仅对 WAV 等标准容器可用。
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


def run(profiles: list[SourceProfile], settings: Settings, output_dir: Path) -> list[NormalizedAudioRecord]:
    # 从 YAML 读取归一化目标参数，便于按项目策略调整。
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
            # 仅处理音频类型输入。
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
            # ffmpeg 失败时保底复制原文件，保持流程可继续。
            shutil.copy2(source_path, output_path)
            engine_name = "copy_fallback"
        else:
            engine_name = "ffmpeg"

        metadata = _probe_with_ffprobe(output_path, settings)
        if not metadata:
            metadata = _probe_with_wave(output_path)

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

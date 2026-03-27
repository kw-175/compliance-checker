"""
Global settings for the audio compliance pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline-wide configuration loaded from environment variables."""

    model_config = {
        "env_prefix": "COMPLIANCE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    work_dir: Path = Field(
        default=Path("./compliance_output_audio"),
        description="Root directory for intermediate and final pipeline artifacts.",
    )

    trufflehog_bin: str = Field(default="trufflehog")
    scancode_bin: str = Field(default="scancode")
    ffmpeg_bin: str = Field(default="ffmpeg")
    ffprobe_bin: str = Field(default="ffprobe")

    qwen_asr_enabled: bool = Field(default=True)
    qwen_asr_model: str = Field(default="Qwen/Qwen3-ASR")
    faster_whisper_enabled: bool = Field(default=True)
    faster_whisper_model: str = Field(default="base")

    pyannote_enabled: bool = Field(default=True)
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1")

    dedup_threshold: float = Field(default=0.8)
    dedup_num_perm: int = Field(default=128)

    keywords_file: Path = Field(
        default=Path(__file__).resolve().parent / "keywords.txt",
    )
    patterns_file: Path = Field(
        default=Path(__file__).resolve().parent / "patterns.yaml",
    )
    ffmpeg_profiles_file: Path = Field(
        default=Path(__file__).resolve().parent / "ffmpeg_profiles.yaml",
    )

    presidio_languages: list[str] = Field(default=["en"])
    pii_model_name: Optional[str] = Field(default="Meddies/meddies-pii")
    pii_score_threshold: float = Field(default=0.35)

    qwen_guard_enabled: bool = Field(default=True)
    qwen_guard_model: str = Field(default="Qwen/Qwen3-Guard-0.6B")
    qwen_guard_device: str = Field(default="auto")

    opa_url: str = Field(default="http://localhost:8181")
    opa_policy_path: str = Field(default="v1/data/compliance/decision")
    opa_enabled: bool = Field(default=True)

    openlineage_url: Optional[str] = Field(default=None)
    openlineage_namespace: str = Field(default="compliance-checker")

    redaction_strategy: str = Field(default="silence")
    beep_frequency: int = Field(default=1000)
    beep_volume: float = Field(default=0.4)

    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8001)
    max_workers: int = Field(default=4)


def get_settings() -> Settings:
    """Return a fresh settings instance."""

    return Settings()

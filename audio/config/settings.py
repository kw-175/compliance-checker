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

    # BaseSettings 统一从 COMPLIANCE_ 前缀环境变量读取配置。
    model_config = {
        "env_prefix": "COMPLIANCE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    work_dir: Path = Field(
        default=Path("./compliance_output_audio"),
        description="Root directory for intermediate and final pipeline artifacts.",
    )

    # 外部工具可执行文件位置。
    trufflehog_bin: str = Field(default="trufflehog")
    scancode_bin: str = Field(default="scancode")
    ffmpeg_bin: str = Field(default="ffmpeg")
    ffprobe_bin: str = Field(default="ffprobe")

    # ASR 引擎配置（Qwen + faster-whisper 双通道回退）。
    qwen_asr_enabled: bool = Field(default=True)
    qwen_asr_model: str = Field(default="Qwen/Qwen3-ASR")
    faster_whisper_enabled: bool = Field(default=True)
    faster_whisper_model: str = Field(default="base")

    # 说话人分离配置。
    pyannote_enabled: bool = Field(default=True)
    pyannote_model: str = Field(default="pyannote/speaker-diarization-3.1")

    # 去重阈值参数。
    dedup_threshold: float = Field(default=0.8)
    dedup_num_perm: int = Field(default=128)

    # 文本扫描与音频归一化配置文件。
    keywords_file: Path = Field(
        default=Path(__file__).resolve().parent / "keywords.txt",
    )
    patterns_file: Path = Field(
        default=Path(__file__).resolve().parent / "patterns.yaml",
    )
    ffmpeg_profiles_file: Path = Field(
        default=Path(__file__).resolve().parent / "ffmpeg_profiles.yaml",
    )

    # PII 识别策略。
    presidio_languages: list[str] = Field(default=["en"])
    pii_model_name: Optional[str] = Field(default="Meddies/meddies-pii")
    pii_score_threshold: float = Field(default=0.35)

    # 语义安全审核模型配置。
    qwen_guard_enabled: bool = Field(default=True)
    qwen_guard_model: str = Field(default="Qwen/Qwen3-Guard-0.6B")
    qwen_guard_device: str = Field(default="auto")

    # OPA 远程策略引擎配置。
    opa_url: str = Field(default="http://localhost:8181")
    opa_policy_path: str = Field(default="v1/data/compliance/decision")
    opa_enabled: bool = Field(default=True)

    # 审计链路上报配置。
    openlineage_url: Optional[str] = Field(default=None)
    openlineage_namespace: str = Field(default="compliance-checker")

    # 音频脱敏渲染策略配置。
    redaction_strategy: str = Field(default="silence")
    beep_frequency: int = Field(default=1000)
    beep_volume: float = Field(default=0.4)

    # 服务启动配置。
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8001)
    max_workers: int = Field(default=4)


def get_settings() -> Settings:
    """Return a fresh settings instance."""

    # 返回新实例，避免全局单例在测试中被污染。
    return Settings()

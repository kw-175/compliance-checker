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

    # External binaries.
    trufflehog_bin: str = Field(default="trufflehog")
    scancode_bin: str = Field(default="scancode")
    ffmpeg_bin: str = Field(default="/data/kw/compliance-checker/models/ffmpeg-7.0.2-amd64-static/ffmpeg")
    ffprobe_bin: str = Field(default="/data/kw/compliance-checker/models/ffmpeg-7.0.2-amd64-static/ffprobe")

    # ASR engines. Qwen is preferred, faster-whisper remains a local fallback.
    qwen_asr_enabled: bool = Field(default=True)
    qwen_asr_model: str = Field(default="/data/kw/compliance-checker/models/Qwen/Qwen3-ASR-0.6B")
    qwen_asr_device: str = Field(default="auto")
    qwen_asr_endpoint: str = Field(default="")
    qwen_asr_timeout_seconds: int = Field(default=180)
    asr_required: bool = Field(default=True)
    asr_unavailable_fallback_enabled: bool = Field(default=False)
    faster_whisper_enabled: bool = Field(default=True)
    faster_whisper_model: str = Field(default="base")

    # Speaker diarization.
    pyannote_enabled: bool = Field(default=False)
    pyannote_model: str = Field(default="/data/kw/compliance-checker/models/pyannote/speaker-diarization-3.1")

    # Legacy transcript dedup settings.
    dedup_threshold: float = Field(default=0.8)
    dedup_num_perm: int = Field(default=128)

    # Local rule assets.
    keywords_file: Path = Field(default=Path(__file__).resolve().parent / "keywords.txt")
    patterns_file: Path = Field(default=Path(__file__).resolve().parent / "patterns.yaml")
    ffmpeg_profiles_file: Path = Field(default=Path(__file__).resolve().parent / "ffmpeg_profiles.yaml")

    # Local PII engine: Presidio + Stanza + GLiNER with regex fallbacks.
    presidio_languages: list[str] = Field(default=["en", "zh"])
    pii_engine: str = Field(default="local_presidio_stanza_gliner")
    pii_model_root: Path = Field(default=Path("/data/kw/compliance-checker/models/compliance-pii"))
    pii_stanza_resources_dir: Path = Field(default=Path("/data/kw/compliance-checker/models/compliance-pii/stanza_resources"))
    pii_stanza_en_model: str = Field(default="en")
    pii_stanza_zh_model: str = Field(default="zh")
    pii_stanza_download_if_missing: bool = Field(default=False)
    pii_enable_presidio: bool = Field(default=True)
    pii_enable_gliner: bool = Field(default=True)
    pii_gliner_model: str = Field(default="/data/kw/compliance-checker/models/compliance-pii/gliner-pii-large-v1.0")
    pii_gliner_threshold: float = Field(default=0.5)
    pii_gliner_max_chars: int = Field(default=12000)
    pii_gliner_labels: str = Field(default="")
    pii_enable_regex_rules: bool = Field(default=True)
    pii_score_threshold: float = Field(default=0.45)
    pii_model_name: Optional[str] = Field(default=None)
    pii_endpoint: str = Field(default="")
    pii_timeout_seconds: int = Field(default=60)
    privacy_operator_ids: list[str] = Field(default_factory=list)
    privacy_target_types: list[str] = Field(default_factory=list)
    audio_privacy_operator_ids: list[str] = Field(default_factory=list)
    audio_privacy_target_types: list[str] = Field(default_factory=list)
    enable_audio_privacy_detection: Optional[bool] = Field(default=None)

    # Content safety moderation.
    qwen_guard_enabled: bool = Field(default=True)
    qwen_guard_model: str = Field(default="/data/kw/compliance-checker/models/Qwen/Qwen3Guard-Gen-0.6B")
    qwen_guard_device: str = Field(default="auto")
    qwen_guard_endpoint: str = Field(default="")
    qwen_guard_timeout_seconds: int = Field(default=90)
    content_safety_operator_ids: list[str] = Field(default_factory=list)
    content_safety_target_labels: list[str] = Field(default_factory=list)
    audio_content_safety_operator_ids: list[str] = Field(default_factory=list)
    audio_content_safety_target_labels: list[str] = Field(default_factory=list)
    enable_audio_content_detection: Optional[bool] = Field(default=None)
    disabled_operator_ids: list[str] = Field(default_factory=list)
    disabled_target_types: list[str] = Field(default_factory=list)
    preserved_training_targets: list[str] = Field(default_factory=list)

    # Hard-case adjudication for uncertain audio compliance records.
    enable_hard_case_adjudication: bool = Field(default=True)
    hard_case_model_name: str = Field(default="Qwen3.5-9B")
    hard_case_prompt_version: str = Field(default="audio-qwen-hard-case-v1")
    hard_case_local_model_path: str = Field(default="/data/kw/compliance-checker/models/Qwen/Qwen3.5-9B")
    hard_case_endpoint: str = Field(default="")
    hard_case_timeout_seconds: int = Field(default=60)
    hard_case_max_chars: int = Field(default=3500)
    hard_case_max_new_tokens: int = Field(default=512)
    hard_case_device: str = Field(default="auto")
    hard_case_asr_confidence_threshold: float = Field(default=0.65)
    hard_case_privacy_score_margin: float = Field(default=0.08)
    hard_case_safety_score_floor: float = Field(default=0.35)
    hard_case_safety_score_ceiling: float = Field(default=0.85)

    # OPA policy integration.
    opa_url: str = Field(default="http://localhost:8181")
    opa_policy_path: str = Field(default="v1/data/compliance/decision")
    opa_enabled: bool = Field(default=True)

    # Optional lineage publishing.
    openlineage_url: Optional[str] = Field(default=None)
    openlineage_namespace: str = Field(default="compliance-checker")

    # Audio redaction rendering settings. The bridge always writes a redaction
    # span plan; rendering modified audio is opt-in.
    audio_redaction_enabled: bool = Field(default=False)
    audio_redaction_padding_ms: int = Field(default=200)
    audio_redaction_min_confidence: float = Field(default=0.0)
    redaction_strategy: str = Field(default="silence")
    beep_frequency: int = Field(default=1000)
    beep_volume: float = Field(default=0.4)

    # API server settings.
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=19001)
    max_workers: int = Field(default=4)

    # Default route: Qwen3-ASR transcription + text API compliance.
    audio_execution_route: str = Field(default="api")
    text_api_base_url: str = Field(default="http://127.0.0.1:19002")
    text_api_submit_path: str = Field(default="/api/v1/check")
    text_api_status_path: str = Field(default="/api/v1/status")
    text_api_result_path: str = Field(default="/api/v1/result")
    text_api_key: str = Field(default="")
    text_api_submit_timeout_seconds: int = Field(default=120)
    text_api_poll_timeout_seconds: int = Field(default=30)
    text_api_task_timeout_seconds: int = Field(default=1800)
    text_api_poll_interval_millis: int = Field(default=3000)


def get_settings() -> Settings:
    """Return a fresh settings instance."""

    return Settings()

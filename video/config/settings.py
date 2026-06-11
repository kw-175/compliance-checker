"""Settings for the video compliance pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline-wide configuration loaded from environment variables."""

    model_config = {
        "env_prefix": "VIDEO_",
        "env_file": ".env",
        "extra": "ignore",
    }

    work_dir: Path = Field(
        default=Path("./compliance_output_video"),
        description="Root directory for intermediate and final video artifacts.",
    )
    storage_base_path: Path = Field(
        default=Path("./compliance_output_video/storage"),
        description="Storage path for persisted video artifacts.",
    )
    default_profile: str = Field(default="default_cn_enterprise")
    route_hint: str = Field(default="auto")
    frame_stride: int = Field(default=6)
    sample_fps: float = Field(default=4.0)
    max_frames: int = Field(default=0)
    default_frame_duration_ms: int = Field(default=250)
    max_workers: int = Field(default=4)
    skip_duplicate_frames: bool = Field(default=True)
    frame_difference_threshold: float = Field(default=0.02)
    scene_detection_enabled: bool = Field(default=True)
    scene_change_threshold: float = Field(default=0.18)
    scene_min_duration_ms: int = Field(default=1000)
    clip_window_ms: int = Field(default=5000)
    clip_window_overlap_ms: int = Field(default=1000)
    clip_moderation_enabled: bool = Field(default=True)
    clip_moderation_base_url: str = Field(default="http://127.0.0.1:8200")
    clip_moderation_endpoint: str = Field(default="/video/action-recognition")
    clip_moderation_timeout_seconds: int = Field(default=120)
    clip_moderation_max_frames: int = Field(default=8)
    clip_moderation_confidence_threshold: float = Field(default=0.55)
    clip_moderation_fail_on_error: bool = Field(default=False)
    track_gap_tolerance_ms: int = Field(default=1000)
    track_iou_threshold: float = Field(default=0.35)
    sam3_video_tracking_enabled: bool = Field(default=True)
    sam3_video_tracker_base_url: str = Field(default="http://127.0.0.1:8218")
    sam3_video_tracker_endpoint: str = Field(default="/v1/sam3/video-track")
    sam3_video_tracker_timeout_seconds: int = Field(default=300)
    sam3_video_tracker_fail_on_error: bool = Field(default=False)
    sam3_video_tracker_return_masks: bool = Field(default=True)
    render_preview: bool = Field(default=True)
    picture_api_base_url: str = Field(default="http://127.0.0.1:19012")
    picture_api_submit_path: str = Field(default="/v1/picture/jobs")
    picture_api_status_path: str = Field(default="/v1/picture/jobs/{job_id}")
    picture_api_report_path: str = Field(default="/v1/picture/jobs/{job_id}/report")
    picture_api_health_path: str = Field(default="/api/v1/health")
    picture_api_timeout_seconds: int = Field(default=30)
    picture_api_task_timeout_seconds: int = Field(default=1800)
    picture_api_poll_interval_seconds: float = Field(default=1.0)
    enable_audio_sidecar: bool = Field(default=True)
    extract_native_audio: bool = Field(default=True)
    fail_on_audio_error: bool = Field(default=False)
    sidecar_audio_extensions: list[str] = Field(default=[".wav", ".mp3"])
    ffmpeg_bin: str = Field(default="ffmpeg")
    ffprobe_bin: str = Field(default="ffprobe")

    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=19003)


def get_settings() -> Settings:
    """Return a fresh settings instance."""

    return Settings()

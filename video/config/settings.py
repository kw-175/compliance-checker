"""Settings for the video compliance pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Pipeline-wide configuration loaded from environment variables."""

    # 统一从 VIDEO_ 前缀环境变量读取配置。
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
    # 任务默认策略与采样参数。
    default_profile: str = Field(default="default_cn_enterprise")
    route_hint: str = Field(default="auto")
    frame_stride: int = Field(default=1)
    max_frames: int = Field(default=0)
    default_frame_duration_ms: int = Field(default=250)
    max_workers: int = Field(default=2)
    render_preview: bool = Field(default=True)
    # 音轨处理策略：可选 sidecar 与容器音轨抽取。
    enable_audio_sidecar: bool = Field(default=True)
    extract_native_audio: bool = Field(default=True)
    fail_on_audio_error: bool = Field(default=False)
    sidecar_audio_extensions: list[str] = Field(default=[".wav", ".mp3"])
    # 外部媒体工具路径。
    ffmpeg_bin: str = Field(default="ffmpeg")
    ffprobe_bin: str = Field(default="ffprobe")

    # HTTP 服务配置。
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8003)


def get_settings() -> Settings:
    """Return a fresh settings instance."""

    # 返回新实例，避免全局可变状态污染测试。
    return Settings()

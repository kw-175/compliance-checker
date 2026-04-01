"""
Configuration management for the picture compliance engine.

Loads settings from environment variables / .env file following
the same pattern as the audio module (pydantic-settings).
"""
# 中文说明：该文件负责 picture 模块的运行时配置管理。
# 通过 Pydantic Settings，可以同时支持环境变量、.env 文件和代码覆盖。
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class PictureSettings(BaseSettings):
    """Pipeline-wide configuration loaded from environment variables."""

    # 中文说明：统一使用 PICTURE_ 前缀读取环境变量，避免与其他模块冲突。
    model_config = {
        "env_prefix": "PICTURE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # 中文说明：通用配置，影响任务目录、默认 profile、并发度和失败策略。
    work_dir: Path = Field(
        default=Path("./compliance_output_picture"),
        description="Root directory for intermediate and final pipeline artifacts.",
    )
    default_profile: str = Field(default="default_cn_enterprise")
    max_workers: int = Field(default=4)
    fail_policy: str = Field(
        default="fail_closed",
        description="fail_open or fail_closed",
    )

    # 中文说明：内置 API 服务监听地址。
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8002)

    # 中文说明：产物存储位置，可切换本地目录或对象存储。
    storage_backend: str = Field(default="local", description="local or s3")
    storage_base_path: Path = Field(default=Path("./compliance_output_picture/storage"))
    s3_bucket: str = Field(default="")
    s3_prefix: str = Field(default="picture-compliance/")
    s3_endpoint_url: Optional[str] = Field(default=None)

    # 中文说明：OCR provider 相关配置。
    ocr_provider: str = Field(
        default="mock",
        description="mock | paddleocr | mineru | surya",
    )
    paddleocr_lang: str = Field(default="ch")
    paddleocr_use_gpu: bool = Field(default=False)

    # 中文说明：文字 PII 检测配置。
    pii_provider: str = Field(default="mock", description="mock | presidio")
    presidio_languages: list[str] = Field(default=["en", "zh"])

    # 中文说明：图像安全审核模型配置。
    safety_provider: str = Field(default="mock", description="mock | shieldgemma2")
    shieldgemma_model: str = Field(default="google/shieldgemma-2b-img")
    shieldgemma_device: str = Field(default="auto")

    # 中文说明：视觉检测模型配置。
    vision_provider: str = Field(
        default="mock",
        description="mock | yolo26 | grounding_dino",
    )
    yolo_model_path: str = Field(default="yolo26n.pt")
    yolo_confidence: float = Field(default=0.25)

    # 中文说明：分割模型配置。
    segmentation_provider: str = Field(default="mock", description="mock | sam2")
    sam2_model_id: str = Field(default="facebook/sam2-hiera-large")

    # 中文说明：不同类型目标的默认脱敏模式配置。
    redaction_mode_text: str = Field(default="black_box")
    redaction_mode_face: str = Field(default="gaussian_blur")
    redaction_mode_qr: str = Field(default="black_box")
    redaction_mode_signature: str = Field(default="solid_fill")
    redaction_mode_default: str = Field(default="black_box")

    # 中文说明：策略 profile 所在目录。
    policy_config_dir: Path = Field(default=_CONFIGS_DIR)

    def get_redaction_mode(self, category: str) -> str:
        """Return the configured redaction mode for a finding category."""
        # 中文说明：部分类别有专门模式，未命中时统一回退到默认模式。
        mode_map: dict[str, str] = {
            "face": self.redaction_mode_face,
            "qr_code": self.redaction_mode_qr,
            "barcode": self.redaction_mode_qr,
            "signature": self.redaction_mode_signature,
            "stamp": self.redaction_mode_signature,
        }
        return mode_map.get(category, self.redaction_mode_default)


@functools.lru_cache(maxsize=1)
def get_settings() -> PictureSettings:
    """Return a cached settings instance."""
    # 中文说明：多数场景下配置可视为只读全局单例，用缓存避免重复解析环境变量。
    return PictureSettings()


def get_fresh_settings(**overrides: Any) -> PictureSettings:
    """Return a new settings instance with optional overrides."""
    # 中文说明：测试场景常需要临时覆盖配置，因此额外保留“非缓存”版本。
    return PictureSettings(**overrides)

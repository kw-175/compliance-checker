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
    ordinary_dataset_enabled: bool = Field(default=True)
    restricted_dataset_enabled: bool = Field(default=False)
    max_redaction_area_ratio: float = Field(default=0.45)

    # 中文说明：内置 API 服务监听地址。
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=19012)

    # 中文说明：产物存储位置，可切换本地目录或对象存储。
    storage_backend: str = Field(default="local", description="local or s3")
    storage_base_path: Path = Field(default=Path("./compliance_output_picture/storage"))
    s3_bucket: str = Field(default="")
    s3_prefix: str = Field(default="picture-compliance/")
    s3_endpoint_url: Optional[str] = Field(default=None)

    # 中文说明：OCR provider 相关配置。
    ocr_provider: str = Field(
        default="paddleocr_vl_api",
        description="mock | paddleocr_vl_api | paddleocr | mineru | surya | openai_gpt52",
    )
    paddleocr_vl_api_url: str = Field(
        default="http://127.0.0.1:8217",
        description="PaddleX Serving endpoint for official PaddleOCR-VL /layout-parsing API.",
    )
    paddleocr_vl_api_timeout_seconds: float = Field(default=300.0)
    paddleocr_vl_api_file_type: int = Field(default=1, description="PaddleX serving fileType: 1=image.")
    paddleocr_vl_api_use_layout_detection: bool = Field(default=True)
    paddleocr_vl_api_use_chart_recognition: bool = Field(default=True)
    paddleocr_vl_api_use_seal_recognition: bool = Field(default=True)
    paddleocr_vl_api_prettify_markdown: bool = Field(default=True)
    paddleocr_vl_api_visualize: bool = Field(default=False)
    paddleocr_model_dir: Path = Field(
        default=Path("/data/kw/compliance-checker/models/paddleocr_vl/PaddleOCR-VL-1.5"),
        description="Local PaddleOCR-VL model directory.",
    )
    paddleocr_lang: str = Field(default="ch")
    paddleocr_use_gpu: bool = Field(default=True)
    paddleocr_device: str = Field(
        default="",
        description="PaddleOCR 3.x device string, e.g. cpu, gpu, gpu:0. Empty derives from paddleocr_use_gpu.",
    )
    paddleocr_vl_task: str = Field(default="spotting", description="PaddleOCR-VL task: spotting | ocr | seal")
    paddleocr_vl_backend: str = Field(
        default="transformers",
        description="PaddleOCR-VL backend: transformers | paddleocr_pipeline | auto",
    )
    paddleocr_vl_max_new_tokens: int = Field(default=768)
    paddleocr_vl_generation_timeout_seconds: float = Field(default=90.0)
    paddleocr_vl_qwen_fallback_enabled: bool = Field(default=True)
    qwen_ocr_timeout_seconds: float = Field(default=180.0)
    qwen_ocr_max_tokens: int = Field(default=4096)

    # 中文说明：文字 PII 检测配置。
    pii_provider: str = Field(
        default="text_compliance",
        description="mock | presidio | openai_gpt52 | text_compliance",
    )
    text_compliance_provider: str = Field(
        default="text_api",
        description="none | text_api | text_pipeline. Reuses the completed text compliance system for OCR text.",
    )
    text_api_base_url: str = Field(default="http://127.0.0.1:19002")
    text_api_timeout_seconds: float = Field(default=300.0)
    text_api_poll_interval_seconds: float = Field(default=2.0)
    presidio_languages: list[str] = Field(default=["en", "zh"])

    # 中文说明：图像安全审核模型配置。
    safety_provider: str = Field(
        default="qwen_sam3_safety_fusion",
        description="mock | shieldgemma2 | openai_gpt52 | qwen35_vl | qwen_sam3_safety_fusion",
    )
    shieldgemma_model: str = Field(default="")
    shieldgemma_device: str = Field(default="cuda")

    # 中文说明：视觉检测模型配置。
    vision_provider: str = Field(
        default="qwen_sam3_api_fusion",
        description="mock | yolo26 | grounding_dino | openai_gpt52 | sam3 | sam3_api | qwen_sam3_fusion | qwen_sam3_api_fusion",
    )
    yolo_model_path: str = Field(default="")
    yolo_confidence: float = Field(default=0.25)
    yolo_device: str = Field(default="cuda")

    # 中文说明：OpenAI 图片外部 API 路线配置。当前仅接入支持图片输入的 GPT-5.2。
    openai_api_key: str = Field(default="")
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    openai_model: str = Field(default="gpt-5.2")
    openai_timeout_seconds: float = Field(default=90.0)
    openai_image_detail: str = Field(default="high")
    qwen35_vl_max_tokens: int = Field(default=384)
    qwen35_vl_image_max_side: int = Field(default=1280)
    qwen35_vl_image_jpeg_quality: int = Field(default=85)
    qwen_sam3_semantic_threshold: float = Field(default=0.55)
    qwen_sam3_keep_without_qwen_threshold: float = Field(default=0.75)

    # 中文说明：分割模型配置。
    segmentation_provider: str = Field(default="sam3_api", description="mock | sam2 | sam3 | sam3_api")
    sam2_model_id: str = Field(default="")
    sam2_model_dir: Path = Field(default=Path(""))
    sam2_device: str = Field(default="cuda")
    sam3_model_dir: Path = Field(default=Path("/data/kw/compliance-checker/models/facebook/sam3"))
    sam3_device: str = Field(default="cuda")
    sam3_confidence: float = Field(default=0.35)
    sam3_api_url: str = Field(default="http://127.0.0.1:8218")
    sam3_api_timeout_seconds: float = Field(default=180.0)

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

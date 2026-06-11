"""
Configuration management for the picture compliance engine.

Loads settings from environment variables / .env file following
the same pattern as the audio module (pydantic-settings).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class PictureSettings(BaseSettings):
    """Pipeline-wide configuration loaded from environment variables."""

    model_config = {
        "env_prefix": "PICTURE_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # ── General ──────────────────────────────────────────────────────
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

    # ── Server ───────────────────────────────────────────────────────
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8002)

    # ── Storage ──────────────────────────────────────────────────────
    storage_backend: str = Field(default="local", description="local or s3")
    storage_base_path: Path = Field(default=Path("./compliance_output_picture/storage"))
    s3_bucket: str = Field(default="")
    s3_prefix: str = Field(default="picture-compliance/")
    s3_endpoint_url: Optional[str] = Field(default=None)

    # ── OCR ──────────────────────────────────────────────────────────
    ocr_provider: str = Field(default="mock", description="mock | paddleocr | mineru | surya")
    paddleocr_lang: str = Field(default="ch")
    paddleocr_use_gpu: bool = Field(default=False)

    # ── PII ──────────────────────────────────────────────────────────
    pii_provider: str = Field(default="mock", description="mock | presidio")
    presidio_languages: list[str] = Field(default=["en", "zh"])

    # ── Safety ───────────────────────────────────────────────────────
    safety_provider: str = Field(default="mock", description="mock | shieldgemma2")
    shieldgemma_model: str = Field(default="google/shieldgemma-2b-img")
    shieldgemma_device: str = Field(default="auto")

    # ── Vision detection ─────────────────────────────────────────────
    vision_provider: str = Field(default="mock", description="mock | yolo26 | grounding_dino")
    yolo_model_path: str = Field(default="yolo26n.pt")
    yolo_confidence: float = Field(default=0.25)

    # ── Segmentation ─────────────────────────────────────────────────
    segmentation_provider: str = Field(default="mock", description="mock | sam2")
    sam2_model_id: str = Field(default="facebook/sam2-hiera-large")

    # ── Redaction defaults ───────────────────────────────────────────
    redaction_mode_text: str = Field(default="black_box")
    redaction_mode_face: str = Field(default="gaussian_blur")
    redaction_mode_qr: str = Field(default="black_box")
    redaction_mode_signature: str = Field(default="solid_fill")
    redaction_mode_default: str = Field(default="black_box")

    # ── Policy ───────────────────────────────────────────────────────
    policy_config_dir: Path = Field(default=_CONFIGS_DIR)

    def get_redaction_mode(self, category: str) -> str:
        """Return the configured redaction mode for a finding category."""
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
    return PictureSettings()


def get_fresh_settings(**overrides: Any) -> PictureSettings:
    """Return a new settings instance with optional overrides."""
    return PictureSettings(**overrides)

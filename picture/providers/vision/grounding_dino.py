"""
Grounding DINO vision detection provider skeleton.

Requires: groundingdino, transformers, torch
"""
# 中文说明：Grounding DINO 这类开放词表检测器适合“类别集合会持续扩展”的场景。
# 当前这里仍是骨架实现，先把接口和依赖边界固定下来。
from __future__ import annotations

import logging
from typing import Any

from picture.domain.models import PictureFinding
from picture.providers.base import VisionDetector

logger = logging.getLogger(__name__)


class GroundingDINOVisionDetector(VisionDetector):
    """Grounding DINO based open-vocabulary object detection provider."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        text_prompt: str = "face . id card . badge . signature . stamp . qr code . barcode . license plate",
        confidence_threshold: float = 0.3,
        **kwargs: Any,
    ) -> None:
        self._model_id = model_id
        self._text_prompt = text_prompt
        self._conf_threshold = confidence_threshold
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "GroundingDINO"

    def detect(self, image_path: str) -> list[PictureFinding]:
        """Run Grounding DINO detection. Requires groundingdino package."""
        try:
            import groundingdino  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("Grounding DINO (groundingdino)")

        # 中文说明：这里暂未真正接入推理逻辑，只保留依赖检查与统一返回接口。
        # 后续落地时，需要把 text_prompt 检测结果映射成 PictureFinding 列表。
        logger.warning("Grounding DINO provider is not yet fully implemented")
        return []

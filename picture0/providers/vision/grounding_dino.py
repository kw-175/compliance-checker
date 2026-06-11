"""
Grounding DINO vision detection provider skeleton.

Requires: groundingdino, transformers, torch
"""

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

        # TODO: Implement Grounding DINO integration
        logger.warning("Grounding DINO provider is not yet fully implemented")
        return []

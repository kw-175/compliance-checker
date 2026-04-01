"""
Surya OCR provider skeleton.

Requires: surya-ocr
"""
# 中文说明：Surya 是另一个可选 OCR provider。
# 当前同样先固定抽象边界，后续需要时再补齐真实推理逻辑。
from __future__ import annotations

import logging

from picture.domain.models import OCRLayoutResult
from picture.providers.base import OCRLayoutProvider

logger = logging.getLogger(__name__)


class SuryaProvider(OCRLayoutProvider):
    """Surya-based OCR and layout analysis provider."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "Surya"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Run Surya OCR analysis. Requires surya-ocr package."""
        try:
            import surya  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("Surya (surya-ocr)")

        # 中文说明：当前还未完成真实集成，因此先返回空结果。
        logger.warning("Surya provider is not yet fully implemented")
        return OCRLayoutResult(engine_name=self.name)

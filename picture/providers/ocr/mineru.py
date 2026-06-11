"""
MinerU OCR/layout provider skeleton.

Requires: magic-pdf (MinerU)
"""
# 中文说明：MinerU 更偏文档解析场景，这里先保留 provider 骨架和依赖检查，
# 便于后续接真实的版面结构结果。
from __future__ import annotations

import logging

from picture.domain.models import OCRLayoutResult
from picture.providers.base import OCRLayoutProvider

logger = logging.getLogger(__name__)


class MinerUProvider(OCRLayoutProvider):
    """MinerU-based document layout analysis and OCR provider."""

    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "MinerU"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Run MinerU analysis. Requires magic-pdf package."""
        try:
            # 中文说明：延迟导入可以避免项目在未安装 MinerU 时导入阶段直接失败。
            import magic_pdf  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError

            raise ProviderNotAvailableError("MinerU (magic-pdf)")

        # 中文说明：当前还未完成真实集成，因此返回空结果对象并记录 warning。
        logger.warning("MinerU provider is not yet fully implemented")
        return OCRLayoutResult(engine_name=self.name)

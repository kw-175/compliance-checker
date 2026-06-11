"""
MinerU OCR/layout provider skeleton.

Requires: magic-pdf (MinerU)
"""

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
            # Lazy import to avoid hard dependency
            import magic_pdf  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            from picture.domain.exceptions import ProviderNotAvailableError
            raise ProviderNotAvailableError("MinerU (magic-pdf)")

        # TODO: Implement MinerU integration when available
        logger.warning("MinerU provider is not yet fully implemented")
        return OCRLayoutResult(engine_name=self.name)

"""
Mock OCR/layout analysis provider for testing and local development.

Returns deterministic OCR results with sample text blocks that include
common PII patterns for end-to-end pipeline testing.
"""

from __future__ import annotations

import logging

from picture.domain.models import BBox, LayoutRegion, OCRLayoutResult, OCRTextBlock
from picture.providers.base import OCRLayoutProvider

logger = logging.getLogger(__name__)


class MockOCRLayoutProvider(OCRLayoutProvider):
    """Mock OCR provider that returns pre-defined text blocks."""

    def __init__(self, return_pii: bool = True) -> None:
        self._return_pii = return_pii

    @property
    def name(self) -> str:
        return "MockOCR"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Return mock OCR results with optional PII content."""
        logger.info("[MockOCR] Analyzing image: %s", image_path)

        blocks: list[OCRTextBlock] = []

        if self._return_pii:
            blocks = [
                OCRTextBlock(
                    text="张三",
                    bbox=BBox(x=100, y=50, w=80, h=30),
                    confidence=0.95,
                    language="zh",
                ),
                OCRTextBlock(
                    text="手机号: 13812345678",
                    bbox=BBox(x=100, y=100, w=200, h=30),
                    confidence=0.92,
                    language="zh",
                ),
                OCRTextBlock(
                    text="邮箱: zhangsan@example.com",
                    bbox=BBox(x=100, y=150, w=250, h=30),
                    confidence=0.90,
                    language="zh",
                ),
                OCRTextBlock(
                    text="身份证号: 110101199001011234",
                    bbox=BBox(x=100, y=200, w=300, h=30),
                    confidence=0.88,
                    language="zh",
                ),
                OCRTextBlock(
                    text="地址: 北京市朝阳区建国路100号",
                    bbox=BBox(x=100, y=250, w=280, h=30),
                    confidence=0.85,
                    language="zh",
                ),
            ]
        else:
            blocks = [
                OCRTextBlock(
                    text="Company Annual Report 2025",
                    bbox=BBox(x=100, y=50, w=300, h=40),
                    confidence=0.97,
                    language="en",
                ),
                OCRTextBlock(
                    text="公开发布",
                    bbox=BBox(x=100, y=100, w=100, h=30),
                    confidence=0.95,
                    language="zh",
                ),
            ]

        full_text = "\n".join(b.text for b in blocks)
        regions = [
            LayoutRegion(
                region_type="text",
                bbox=BBox(x=80, y=30, w=400, h=280),
                text_blocks=blocks,
            )
        ]

        return OCRLayoutResult(
            full_text=full_text,
            text_blocks=blocks,
            layout_regions=regions,
            engine_name=self.name,
        )

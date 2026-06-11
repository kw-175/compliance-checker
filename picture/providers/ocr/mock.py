"""
Mock OCR/layout analysis provider for testing and local development.

Returns deterministic OCR results with sample text blocks that include
common PII patterns for end-to-end pipeline testing.
"""
# 中文说明：这是测试用 OCR 模拟实现，用固定文本块来驱动后续 PII 和脱敏链路。

from __future__ import annotations

import logging

from picture.domain.models import BBox, LayoutRegion, OCRLayoutResult, OCRTextBlock
from picture.providers.base import OCRLayoutProvider

logger = logging.getLogger(__name__)


class MockOCRLayoutProvider(OCRLayoutProvider):
    """Mock OCR provider that returns pre-defined text blocks."""

    def __init__(self, return_pii: bool = True) -> None:
        # 中文说明：return_pii 为 True 时返回包含敏感信息样例的 OCR 结果，
        # 方便驱动下游 PII 检测和脱敏链路。
        self._return_pii = return_pii

    @property
    def name(self) -> str:
        return "MockOCR"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Return mock OCR results with optional PII content."""
        logger.info("[MockOCR] Analyzing image: %s", image_path)

        blocks: list[OCRTextBlock] = []

        if self._return_pii:
            # 中文说明：这一组样例文本故意包含手机号、邮箱、证件号、地址等模式，
            # 用于验证文本 PII 检测是否能正确命中。
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
            # 中文说明：关闭 return_pii 时返回相对干净的文本内容，
            # 用来测试“有 OCR 但不触发 PII”的路径。
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

        # 中文说明：full_text 是把 block 文本拼成的一整段，便于 PII 检测器整体扫描。
        full_text = "\n".join(b.text for b in blocks)
        regions = [
            LayoutRegion(
                region_type="text",
                bbox=BBox(x=80, y=30, w=400, h=280),
                text_blocks=blocks,
            )
        ]

        # 中文说明：mock provider 只构造一个大文本区域即可，重点是保证数据结构完整。
        return OCRLayoutResult(
            full_text=full_text,
            text_blocks=blocks,
            layout_regions=regions,
            engine_name=self.name,
        )

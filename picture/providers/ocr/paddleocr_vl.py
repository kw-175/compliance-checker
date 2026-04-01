"""
PaddleOCR-VL provider skeleton.

Requires: paddleocr, paddlepaddle
Install: pip install paddleocr paddlepaddle
"""
# 中文说明：该 provider 把 PaddleOCR 的返回结果整理成统一的 OCRLayoutResult。
# 上层不需要关心 PaddleOCR 原始输出结构，只消费标准数据模型。
from __future__ import annotations

import logging
from typing import Any

from picture.domain.models import BBox, OCRLayoutResult, OCRTextBlock
from picture.providers.base import OCRLayoutProvider

logger = logging.getLogger(__name__)


class PaddleOCRVLProvider(OCRLayoutProvider):
    """PaddleOCR-VL based OCR + layout analysis provider."""

    def __init__(self, lang: str = "ch", use_gpu: bool = False, **kwargs: Any) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._kwargs = kwargs
        self._engine: Any = None

    def _get_engine(self) -> Any:
        """Lazy initialization of PaddleOCR engine."""
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]

                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    **self._kwargs,
                )
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError

                raise ProviderNotAvailableError("PaddleOCR")
        return self._engine

    @property
    def name(self) -> str:
        return "PaddleOCR-VL"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        """Run PaddleOCR on the image."""
        engine = self._get_engine()
        result = engine.ocr(image_path, cls=True)

        blocks: list[OCRTextBlock] = []
        if result and result[0]:
            for line in result[0]:
                box_pts, (text, conf) = line

                # 中文说明：PaddleOCR 返回的是四点框，这里转成系统统一使用的 axis-aligned bbox。
                x_min = min(p[0] for p in box_pts)
                y_min = min(p[1] for p in box_pts)
                x_max = max(p[0] for p in box_pts)
                y_max = max(p[1] for p in box_pts)
                blocks.append(
                    OCRTextBlock(
                        text=text,
                        bbox=BBox(x=x_min, y=y_min, w=x_max - x_min, h=y_max - y_min),
                        confidence=conf,
                    )
                )

        # 中文说明：full_text 由所有文本块拼接而成，方便后续直接送入 PII 检测器。
        full_text = "\n".join(b.text for b in blocks)
        return OCRLayoutResult(
            full_text=full_text,
            text_blocks=blocks,
            engine_name=self.name,
        )

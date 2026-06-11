"""
OpenAI GPT-5.2 OCR/layout provider.
"""
from __future__ import annotations

from picture.domain.models import OCRLayoutResult
from picture.providers.base import OCRLayoutProvider
from picture.providers.openai_shared import OpenAIPictureAnalyzer


class OpenAIGPT52OCRLayoutProvider(OCRLayoutProvider):
    def __init__(self, analyzer: OpenAIPictureAnalyzer) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "OpenAIGPT52OCR"

    def analyze(self, image_path: str) -> OCRLayoutResult:
        return self._analyzer.build_ocr_result(image_path)

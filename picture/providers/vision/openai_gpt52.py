"""
OpenAI GPT-5.2 vision detection provider.
"""
from __future__ import annotations

from picture.domain.models import PictureFinding
from picture.providers.base import VisionDetector
from picture.providers.openai_shared import OpenAIPictureAnalyzer


class OpenAIGPT52VisionDetector(VisionDetector):
    def __init__(self, analyzer: OpenAIPictureAnalyzer) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "OpenAIGPT52Vision"

    def detect(self, image_path: str) -> list[PictureFinding]:
        return self._analyzer.build_vision_findings(image_path)

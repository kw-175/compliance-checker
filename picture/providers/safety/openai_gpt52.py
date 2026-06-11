"""
OpenAI GPT-5.2 safety moderation provider.
"""
from __future__ import annotations

from picture.domain.models import PictureModerationResult
from picture.providers.base import SafetyModerator
from picture.providers.openai_shared import OpenAIPictureAnalyzer


class OpenAIGPT52SafetyModerator(SafetyModerator):
    def __init__(self, analyzer: OpenAIPictureAnalyzer) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "OpenAIGPT52Safety"

    def moderate(self, image_path: str) -> PictureModerationResult:
        return self._analyzer.build_safety_result(image_path)

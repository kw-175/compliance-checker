"""
OpenAI GPT-5.2 precomputed PII provider.

The actual image analysis is performed by the shared OCR provider and cached
inside OCR metadata. This detector remains as a compatibility layer for the
existing orchestrator wiring.
"""
from __future__ import annotations

import logging

from picture.domain.models import PictureFinding
from picture.providers.base import PIIDetector
from picture.providers.openai_shared import OpenAIPictureAnalyzer

logger = logging.getLogger(__name__)


class OpenAIGPT52PIIDetector(PIIDetector):
    def __init__(self, analyzer: OpenAIPictureAnalyzer) -> None:
        self._analyzer = analyzer

    @property
    def name(self) -> str:
        return "OpenAIGPT52PII"

    def detect(self, text: str, language: str = "zh") -> list[PictureFinding]:
        logger.info(
            "[OpenAIGPT52PII] returning empty direct detection result; OCR metadata carries precomputed findings"
        )
        return []

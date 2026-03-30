"""
Presidio PII detection provider skeleton.

Requires: presidio-analyzer, presidio-anonymizer
"""

from __future__ import annotations

import logging
from typing import Any

from picture.domain.enums import FindingType
from picture.domain.models import PictureFinding
from picture.providers.base import PIIDetector

logger = logging.getLogger(__name__)

# Mapping from Presidio entity types to our internal types
_PRESIDIO_MAP: dict[str, str] = {
    "PERSON": "person_name",
    "PHONE_NUMBER": "phone_number",
    "EMAIL_ADDRESS": "email",
    "CREDIT_CARD": "bank_card",
    "IBAN_CODE": "bank_card",
    "LOCATION": "address",
    "IP_ADDRESS": "other",
    "URL": "other",
}


class PresidioPIIDetector(PIIDetector):
    """Presidio-based PII detection provider."""

    def __init__(self, languages: list[str] | None = None, **kwargs: Any) -> None:
        self._languages = languages or ["en", "zh"]
        self._kwargs = kwargs
        self._analyzer: Any = None

    def _get_analyzer(self) -> Any:
        """Lazy initialization of Presidio analyzer."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]
                self._analyzer = AnalyzerEngine()
            except ImportError:
                from picture.domain.exceptions import ProviderNotAvailableError
                raise ProviderNotAvailableError("Presidio (presidio-analyzer)")
        return self._analyzer

    @property
    def name(self) -> str:
        return "Presidio"

    def detect(self, text: str, language: str = "en") -> list[PictureFinding]:
        """Detect PII using Presidio analyzer."""
        analyzer = self._get_analyzer()
        results = analyzer.analyze(text=text, language=language)

        findings: list[PictureFinding] = []
        for result in results:
            category = _PRESIDIO_MAP.get(result.entity_type, "other")
            findings.append(PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category=category,
                label=f"PII: {result.entity_type}",
                score=result.score,
                text_span=text[result.start:result.end],
                reason_code=f"PII_{result.entity_type}",
                provider=self.name,
                metadata={
                    "char_start": result.start,
                    "char_end": result.end,
                    "presidio_entity_type": result.entity_type,
                },
            ))

        return findings

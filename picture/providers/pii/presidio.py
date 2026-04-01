"""
Presidio PII detection provider skeleton.

Requires: presidio-analyzer, presidio-anonymizer
"""
# 中文说明：该 provider 把微软 Presidio 的分析结果转成 picture 模块自己的 finding 结构。
# 这样上层编排器就不需要感知 Presidio 的原生返回格式。
from __future__ import annotations

import logging
from typing import Any

from picture.domain.enums import FindingType
from picture.domain.models import PictureFinding
from picture.providers.base import PIIDetector

logger = logging.getLogger(__name__)

# 中文说明：Presidio 的实体命名和本项目内部 category 命名并不完全一致，
# 因此需要做一层映射，保持策略和脱敏配置的类别稳定。
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
        # 中文说明：languages 预留给多语言场景，_analyzer 采用懒加载避免导入即初始化。
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

        # 中文说明：Presidio 返回的是字符级 span，因此非常适合回填到 text_span 中。
        results = analyzer.analyze(text=text, language=language)

        findings: list[PictureFinding] = []
        for result in results:
            category = _PRESIDIO_MAP.get(result.entity_type, "other")
            findings.append(
                PictureFinding(
                    finding_type=FindingType.TEXT_PII,
                    category=category,
                    label=f"PII: {result.entity_type}",
                    score=result.score,
                    text_span=text[result.start : result.end],
                    reason_code=f"PII_{result.entity_type}",
                    provider=self.name,
                    metadata={
                        "char_start": result.start,
                        "char_end": result.end,
                        "presidio_entity_type": result.entity_type,
                    },
                )
            )

        return findings

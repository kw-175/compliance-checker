"""
Mock PII detector for testing and local development.

Uses regex patterns to detect common Chinese PII types:
phone numbers, email, ID card numbers, bank card numbers, license plates.
"""

from __future__ import annotations

import logging
import re

from picture.domain.enums import FindingType, PIIEntityType
from picture.domain.models import PictureFinding
from picture.providers.base import PIIDetector

logger = logging.getLogger(__name__)

# Regex patterns for common Chinese PII
_PATTERNS: list[tuple[PIIEntityType, str, re.Pattern[str]]] = [
    (PIIEntityType.PHONE_NUMBER, "PII_PHONE",
     re.compile(r"1[3-9]\d{9}")),
    (PIIEntityType.EMAIL, "PII_EMAIL",
     re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    (PIIEntityType.ID_CARD, "PII_ID_CARD",
     re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]")),
    (PIIEntityType.BANK_CARD, "PII_BANK_CARD",
     re.compile(r"\b[3-6]\d{15,18}\b")),
    (PIIEntityType.LICENSE_PLATE, "PII_LICENSE_PLATE",
     re.compile(r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁][A-Z][A-HJ-NP-Z0-9]{5,6}")),
    (PIIEntityType.PERSON_NAME, "PII_NAME",
     re.compile(r"姓名[：:]\s*[\u4e00-\u9fff]{2,4}")),
    (PIIEntityType.ADDRESS, "PII_ADDRESS",
     re.compile(r"地址[：:]\s*[\u4e00-\u9fff\d]+(?:路|街|巷|号|楼|室|区|市|省|镇|村)[\u4e00-\u9fff\d]*")),
]


class MockPIIDetector(PIIDetector):
    """Regex-based PII detector for testing."""

    @property
    def name(self) -> str:
        return "MockPII"

    def detect(self, text: str, language: str = "zh") -> list[PictureFinding]:
        """Detect PII entities using regex patterns."""
        logger.info("[MockPII] Scanning text of length %d", len(text))
        findings: list[PictureFinding] = []

        for entity_type, reason_code, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                findings.append(PictureFinding(
                    finding_type=FindingType.TEXT_PII,
                    category=entity_type.value,
                    label=f"PII: {entity_type.value}",
                    score=0.95,
                    text_span=match.group(),
                    reason_code=reason_code,
                    provider=self.name,
                    metadata={
                        "char_start": match.start(),
                        "char_end": match.end(),
                    },
                ))

        logger.info("[MockPII] Found %d PII entities", len(findings))
        return findings

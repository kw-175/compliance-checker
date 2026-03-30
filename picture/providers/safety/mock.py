"""
Mock safety moderator for testing and local development.

Provides deterministic safety results based on filename heuristics
for end-to-end pipeline testing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from picture.domain.enums import SafetyCategory
from picture.domain.models import PictureModerationResult
from picture.providers.base import SafetyModerator

logger = logging.getLogger(__name__)


class MockSafetyModerator(SafetyModerator):
    """
    Mock safety moderator.

    Returns unsafe results if the filename contains 'unsafe' or 'explicit',
    otherwise returns safe.
    """

    def __init__(self, default_safe: bool = True) -> None:
        self._default_safe = default_safe

    @property
    def name(self) -> str:
        return "MockSafety"

    def moderate(self, image_path: str) -> PictureModerationResult:
        """Return mock moderation result based on filename."""
        logger.info("[MockSafety] Moderating image: %s", image_path)

        filename = Path(image_path).stem.lower()

        if "explicit" in filename or "nsfw" in filename:
            return PictureModerationResult(
                is_safe=False,
                categories=[SafetyCategory.EXPLICIT],
                scores={"explicit": 0.95, "safe": 0.05},
                reason_codes=["SAFETY_EXPLICIT"],
                provider=self.name,
            )
        elif "violence" in filename or "gore" in filename:
            return PictureModerationResult(
                is_safe=False,
                categories=[SafetyCategory.GRAPHIC_VIOLENCE],
                scores={"graphic_violence": 0.88, "safe": 0.12},
                reason_codes=["SAFETY_GRAPHIC_VIOLENCE"],
                provider=self.name,
            )
        elif "unsafe" in filename:
            return PictureModerationResult(
                is_safe=False,
                categories=[SafetyCategory.DANGEROUS],
                scores={"dangerous": 0.78, "safe": 0.22},
                reason_codes=["SAFETY_DANGEROUS"],
                provider=self.name,
            )

        return PictureModerationResult(
            is_safe=True,
            categories=[SafetyCategory.SAFE],
            scores={"safe": 0.99},
            reason_codes=[],
            provider=self.name,
        )

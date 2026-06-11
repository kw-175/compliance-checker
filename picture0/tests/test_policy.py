"""
Tests for the policy engine.

Validates that the configurable policy engine produces correct decisions:
- clean → pass_raw
- PII → pass_redacted
- face → pass_redacted
- explicit → drop
"""

from __future__ import annotations

from pathlib import Path

import pytest

from picture.domain.enums import DecisionType, FindingType, SafetyCategory
from picture.domain.models import (
    BBox,
    PictureFinding,
    PictureModerationResult,
    RegionMask,
)
from picture.domain.policy import ConfigurablePolicyEngine

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def engine() -> ConfigurablePolicyEngine:
    return ConfigurablePolicyEngine(CONFIGS_DIR)


class TestPolicyDecisions:
    """Test policy evaluation decisions."""

    def test_clean_image_passes_raw(self, engine: ConfigurablePolicyEngine):
        """No findings → pass_raw."""
        result = engine.evaluate([], None, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_RAW
        assert len(result.reason_codes) == 0

    def test_pii_triggers_redaction(self, engine: ConfigurablePolicyEngine):
        """PII finding → pass_redacted."""
        findings = [
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="phone_number",
                label="Phone",
                score=0.95,
                reason_code="PII_PHONE",
            )
        ]
        result = engine.evaluate(findings, None, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_REDACTED
        assert "PII_PHONE" in result.reason_codes

    def test_face_triggers_redaction(self, engine: ConfigurablePolicyEngine):
        """Face finding → pass_redacted."""
        findings = [
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="face",
                label="Face",
                score=0.9,
                region=RegionMask(bbox=BBox(x=100, y=100, w=50, h=50), confidence=0.9),
                reason_code="VISION_FACE",
            )
        ]
        result = engine.evaluate(findings, None, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_REDACTED
        assert "VISION_FACE" in result.reason_codes

    def test_explicit_triggers_drop(self, engine: ConfigurablePolicyEngine):
        """Explicit safety moderation → drop."""
        moderation = PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.EXPLICIT],
            scores={"explicit": 0.95},
            reason_codes=["SAFETY_EXPLICIT"],
        )
        result = engine.evaluate([], moderation, "default_cn_enterprise")
        assert result.decision == DecisionType.DROP
        assert "SAFETY_EXPLICIT" in result.reason_codes

    def test_violence_triggers_drop(self, engine: ConfigurablePolicyEngine):
        """Graphic violence → drop."""
        moderation = PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.GRAPHIC_VIOLENCE],
            scores={"graphic_violence": 0.85},
            reason_codes=["SAFETY_GRAPHIC_VIOLENCE"],
        )
        result = engine.evaluate([], moderation, "default_cn_enterprise")
        assert result.decision == DecisionType.DROP

    def test_drop_overrides_redaction(self, engine: ConfigurablePolicyEngine):
        """DROP should override REDACTED when both are triggered."""
        findings = [
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="phone_number",
                score=0.9,
                reason_code="PII_PHONE",
            )
        ]
        moderation = PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.EXPLICIT],
            scores={"explicit": 0.95},
            reason_codes=["SAFETY_EXPLICIT"],
        )
        result = engine.evaluate(findings, moderation, "default_cn_enterprise")
        assert result.decision == DecisionType.DROP

    def test_low_score_finding_ignored(self, engine: ConfigurablePolicyEngine):
        """Finding with score below threshold should not trigger redaction."""
        findings = [
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="phone_number",
                score=0.1,  # Below threshold
                reason_code="PII_PHONE",
            )
        ]
        result = engine.evaluate(findings, None, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_RAW

    def test_low_safety_score_not_dropped(self, engine: ConfigurablePolicyEngine):
        """Safety score below threshold should not trigger drop."""
        moderation = PictureModerationResult(
            is_safe=False,
            categories=[SafetyCategory.EXPLICIT],
            scores={"explicit": 0.3},  # Below 0.7 threshold
            reason_codes=["SAFETY_EXPLICIT"],
        )
        result = engine.evaluate([], moderation, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_RAW

    def test_multiple_findings_combined(self, engine: ConfigurablePolicyEngine):
        """Multiple findings should produce correct combined decision."""
        findings = [
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="email",
                score=0.9,
                reason_code="PII_EMAIL",
            ),
            PictureFinding(
                finding_type=FindingType.VISION_OBJECT,
                category="qr_code",
                score=0.85,
                reason_code="VISION_QR_CODE",
            ),
        ]
        result = engine.evaluate(findings, None, "default_cn_enterprise")
        assert result.decision == DecisionType.PASS_REDACTED
        assert "PII_EMAIL" in result.reason_codes
        assert "VISION_QR_CODE" in result.reason_codes

    def test_profile_recorded(self, engine: ConfigurablePolicyEngine):
        """Result should record which profile was used."""
        result = engine.evaluate([], None, "default_cn_enterprise")
        assert result.profile == "default_cn_enterprise"

    def test_nonexistent_profile_raises(self, engine: ConfigurablePolicyEngine):
        """Nonexistent profile should raise ConfigurationError."""
        from picture.domain.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            engine.evaluate([], None, "nonexistent_profile")

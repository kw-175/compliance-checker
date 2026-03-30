"""
Configurable policy engine for picture compliance decisions.

Evaluates findings + moderation results against a named profile
loaded from YAML configuration files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from picture.domain.enums import DecisionType, FindingType
from picture.domain.exceptions import ConfigurationError
from picture.domain.models import (
    PictureFinding,
    PictureModerationResult,
    PicturePolicyResult,
)
from picture.providers.base import PolicyEngine as PolicyEngineBase

logger = logging.getLogger(__name__)


class ConfigurablePolicyEngine(PolicyEngineBase):
    """
    Policy engine that evaluates findings against YAML-configured profiles.

    Each profile defines:
    - safety_drop_categories: list of safety categories that cause a DROP
    - safety_drop_threshold: minimum score for a safety category to trigger DROP
    - pii_redact_categories: list of PII types that cause PASS_REDACTED
    - vision_redact_categories: list of vision object types that cause PASS_REDACTED
    - min_score_threshold: minimum confidence for a finding to be considered
    """

    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._profiles: dict[str, dict[str, Any]] = {}

    def _load_profile(self, profile_name: str) -> dict[str, Any]:
        """Load a policy profile from YAML."""
        if profile_name in self._profiles:
            return self._profiles[profile_name]

        config_path = self._config_dir / f"{profile_name}.yaml"
        if not config_path.exists():
            raise ConfigurationError(
                f"Policy profile '{profile_name}' not found at {config_path}"
            )

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        self._profiles[profile_name] = config
        logger.info("Loaded policy profile '%s' from %s", profile_name, config_path)
        return config

    def evaluate(
        self,
        findings: list[PictureFinding],
        moderation: PictureModerationResult | None,
        profile: str = "default_cn_enterprise",
    ) -> PicturePolicyResult:
        """
        Evaluate findings against the named policy profile.

        Decision priority: DROP > PASS_REDACTED > PASS_RAW
        """
        config = self._load_profile(profile)
        policy = config.get("policy", {})

        reason_codes: list[str] = []
        needs_redaction = False
        should_drop = False

        # ── 1) Safety moderation check ──────────────────────────────
        safety_drop_categories = set(policy.get("safety_drop_categories", [
            "explicit", "graphic_violence",
        ]))
        safety_drop_threshold = float(policy.get("safety_drop_threshold", 0.7))

        if moderation and not moderation.is_safe:
            for cat in moderation.categories:
                cat_value = cat.value if hasattr(cat, "value") else str(cat)
                score = moderation.scores.get(cat_value, 0.0)
                if cat_value in safety_drop_categories and score >= safety_drop_threshold:
                    should_drop = True
                    reason_codes.append(f"SAFETY_{cat_value.upper()}")
                    logger.info(
                        "Policy DROP triggered by safety category '%s' (score=%.2f)",
                        cat_value, score,
                    )

        # ── 2) Text PII check ──────────────────────────────────────
        pii_redact_categories = set(policy.get("pii_redact_categories", [
            "person_name", "phone_number", "email", "id_card",
            "bank_card", "address", "license_plate",
        ]))
        min_score = float(policy.get("min_score_threshold", 0.5))

        for finding in findings:
            if finding.finding_type == FindingType.TEXT_PII:
                if finding.category in pii_redact_categories and finding.score >= min_score:
                    needs_redaction = True
                    reason_codes.append(finding.reason_code or f"PII_{finding.category.upper()}")

        # ── 3) Vision object check ─────────────────────────────────
        vision_redact_categories = set(policy.get("vision_redact_categories", [
            "face", "id_card", "badge", "qr_code", "barcode",
            "license_plate", "signature", "stamp",
        ]))

        for finding in findings:
            if finding.finding_type == FindingType.VISION_OBJECT:
                if finding.category in vision_redact_categories and finding.score >= min_score:
                    needs_redaction = True
                    reason_codes.append(finding.reason_code or f"VISION_{finding.category.upper()}")

        # ── 4) Determine final decision ────────────────────────────
        # Deduplicate reason codes while preserving order
        seen: set[str] = set()
        unique_reasons: list[str] = []
        for rc in reason_codes:
            if rc not in seen:
                seen.add(rc)
                unique_reasons.append(rc)

        if should_drop:
            decision = DecisionType.DROP
        elif needs_redaction:
            decision = DecisionType.PASS_REDACTED
        else:
            decision = DecisionType.PASS_RAW

        logger.info(
            "Policy decision: %s (reasons=%s, profile=%s)",
            decision.value, unique_reasons, profile,
        )

        return PicturePolicyResult(
            decision=decision,
            reason_codes=unique_reasons,
            profile=profile,
        )

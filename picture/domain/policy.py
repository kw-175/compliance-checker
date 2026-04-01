"""
Configurable policy engine for picture compliance decisions.

Evaluates findings + moderation results against a named profile
loaded from YAML configuration files.
"""
# 中文说明：策略层是把“模型发现了什么”翻译成“业务上该怎么处理”的关键模块。
# 这里把策略规则从代码中抽离到 YAML profile，便于按租户或场景配置。
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
        # 中文说明：config_dir 指向策略配置目录，_profiles 是已加载 profile 的缓存。
        self._config_dir = Path(config_dir)
        self._profiles: dict[str, dict[str, Any]] = {}

    def _load_profile(self, profile_name: str) -> dict[str, Any]:
        """Load a policy profile from YAML."""
        # 中文说明：已加载过的 profile 直接复用，避免重复读文件。
        if profile_name in self._profiles:
            return self._profiles[profile_name]

        config_path = self._config_dir / f"{profile_name}.yaml"
        if not config_path.exists():
            raise ConfigurationError(
                f"Policy profile '{profile_name}' not found at {config_path}"
            )

        # 中文说明：如果 YAML 是空文件，则 safe_load 可能返回 None，
        # 这里统一退回空 dict，后续通过默认值补齐。
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

        # 中文说明：第一阶段先看安全审核结果。
        # 如果命中了配置中的高风险类别且分数超过阈值，可以直接触发 DROP。
        safety_drop_categories = set(
            policy.get("safety_drop_categories", ["explicit", "graphic_violence"])
        )
        safety_drop_threshold = float(policy.get("safety_drop_threshold", 0.7))

        if moderation and not moderation.is_safe:
            for cat in moderation.categories:
                cat_value = cat.value if hasattr(cat, "value") else str(cat)
                score = moderation.scores.get(cat_value, 0.0)
                if (
                    cat_value in safety_drop_categories
                    and score >= safety_drop_threshold
                ):
                    should_drop = True
                    reason_codes.append(f"SAFETY_{cat_value.upper()}")
                    logger.info(
                        "Policy DROP triggered by safety category '%s' (score=%.2f)",
                        cat_value,
                        score,
                    )

        # 中文说明：第二阶段处理文本 PII。
        # 文本敏感信息通常不会直接 DROP，而是触发 PASS_REDACTED。
        pii_redact_categories = set(
            policy.get(
                "pii_redact_categories",
                [
                    "person_name",
                    "phone_number",
                    "email",
                    "id_card",
                    "bank_card",
                    "address",
                    "license_plate",
                ],
            )
        )
        min_score = float(policy.get("min_score_threshold", 0.5))

        for finding in findings:
            if finding.finding_type == FindingType.TEXT_PII:
                if (
                    finding.category in pii_redact_categories
                    and finding.score >= min_score
                ):
                    needs_redaction = True
                    reason_codes.append(
                        finding.reason_code or f"PII_{finding.category.upper()}"
                    )

        # 中文说明：第三阶段处理视觉类敏感对象。
        # 例如人脸、工牌、二维码、签名、印章等。
        vision_redact_categories = set(
            policy.get(
                "vision_redact_categories",
                [
                    "face",
                    "id_card",
                    "badge",
                    "qr_code",
                    "barcode",
                    "license_plate",
                    "signature",
                    "stamp",
                ],
            )
        )

        for finding in findings:
            if finding.finding_type == FindingType.VISION_OBJECT:
                if (
                    finding.category in vision_redact_categories
                    and finding.score >= min_score
                ):
                    needs_redaction = True
                    reason_codes.append(
                        finding.reason_code or f"VISION_{finding.category.upper()}"
                    )

        # 中文说明：去重 reason code，但保留原始出现顺序，
        # 这样报告更稳定，也不至于同一原因重复出现。
        seen: set[str] = set()
        unique_reasons: list[str] = []
        for rc in reason_codes:
            if rc not in seen:
                seen.add(rc)
                unique_reasons.append(rc)

        # 中文说明：最终决策优先级是 DROP > PASS_REDACTED > PASS_RAW。
        if should_drop:
            decision = DecisionType.DROP
        elif needs_redaction:
            decision = DecisionType.PASS_REDACTED
        else:
            decision = DecisionType.PASS_RAW

        logger.info(
            "Policy decision: %s (reasons=%s, profile=%s)",
            decision.value,
            unique_reasons,
            profile,
        )

        return PicturePolicyResult(
            decision=decision,
            reason_codes=unique_reasons,
            profile=profile,
        )

"""
Configurable policy engine for picture compliance decisions.
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
    def __init__(self, config_dir: str | Path) -> None:
        self._config_dir = Path(config_dir)
        self._profiles: dict[str, dict[str, Any]] = {}

    def _load_profile(self, profile_name: str) -> dict[str, Any]:
        if profile_name in self._profiles:
            return self._profiles[profile_name]

        config_path = self._config_dir / f"{profile_name}.yaml"
        if not config_path.exists():
            raise ConfigurationError(
                f"Policy profile '{profile_name}' not found at {config_path}"
            )

        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        self._profiles[profile_name] = config
        return config

    def evaluate(
        self,
        findings: list[PictureFinding],
        moderation: PictureModerationResult | None,
        profile: str = "default_cn_enterprise",
        context: dict[str, Any] | None = None,
    ) -> PicturePolicyResult:
        config = self._load_profile(profile)
        policy = config.get("policy", {})
        context = context or {}

        min_score = float(policy.get("min_score_threshold", 0.5))
        safety_drop_categories = set(
            policy.get("safety_drop_categories", ["explicit", "graphic_violence", "self_harm", "other_nsfw"])
        )
        safety_drop_threshold = float(policy.get("safety_drop_threshold", 0.7))
        pii_redact_categories = set(
            policy.get(
                "pii_redact_categories",
                [
                    "person_name",
                    "phone_number",
                    "email",
                    "id_card",
                    "bank_card",
                    "bank_account",
                    "address",
                    "license_plate",
                    "student_id",
                    "date_time",
                    "pii_entity",
                ],
            )
        )
        vision_redact_categories = set(
            policy.get(
                "vision_redact_categories",
                ["face", "id_card", "badge", "qr_code", "barcode", "license_plate", "signature", "stamp"],
            )
        )

        ordinary_enabled = bool(context.get("ordinary_dataset_enabled", True))
        restricted_enabled = bool(context.get("restricted_dataset_enabled", False))
        restricted_use_case = str(context.get("restricted_use_case", "") or "").strip()
        authorized_sensitive_use = bool(context.get("authorized_sensitive_use", False))
        education_value_preserved = bool(context.get("education_value_preserved", True))
        executed_steps = list(context.get("executed_steps", []))
        skipped_steps = list(context.get("skipped_steps", []))

        should_drop = False
        needs_redaction = False
        review_required = False
        reason_codes: list[str] = []

        if moderation and not moderation.is_safe:
            categories = list(moderation.categories) or ["other_nsfw"]
            for category in categories:
                cat_value = getattr(category, "value", str(category))
                score = float(moderation.scores.get(cat_value, 1.0))
                if _moderation_category_is_redact_only(moderation, cat_value):
                    needs_redaction = True
                    reason_codes.append(f"SAFETY_{cat_value.upper()}")
                    continue
                if cat_value in safety_drop_categories and score >= safety_drop_threshold:
                    should_drop = True
                    reason_codes.append(f"SAFETY_{cat_value.upper()}")

        for finding in findings:
            if finding.score < min_score:
                continue
            if finding.finding_type == FindingType.TEXT_PII:
                if finding.category in pii_redact_categories:
                    needs_redaction = True
                    reason_codes.append(finding.reason_code or f"PII_{finding.category.upper()}")
            elif finding.finding_type == FindingType.VISION_OBJECT:
                if finding.category in vision_redact_categories:
                    if finding.region is None and bool((finding.metadata or {}).get("localization_required", False)):
                        review_required = True
                        should_drop = True
                        reason_codes.append(finding.reason_code or f"VISION_UNLOCALIZED_{finding.category.upper()}")
                        continue
                    needs_redaction = True
                    reason_codes.append(finding.reason_code or f"VISION_{finding.category.upper()}")
                    if finding.category in {"face", "badge", "signature", "stamp", "qr_code", "barcode"}:
                        review_required = True
            elif finding.finding_type == FindingType.TEXT_CONTENT:
                review_required = True
                reason_codes.append(finding.reason_code or f"TEXT_CONTENT_{finding.category.upper()}")
                if finding.category in {
                    "violence",
                    "sexual_content",
                    "self_harm",
                    "hate_speech",
                    "illegal_instruction",
                    "ocr_extraction_failed",
                }:
                    should_drop = True
            elif finding.finding_type == FindingType.SAFETY:
                redact_only = _finding_is_redact_only(finding)
                review_required = review_required or not redact_only
                reason_codes.append(finding.reason_code or f"SAFETY_{finding.category.upper()}")
                if redact_only:
                    needs_redaction = True
                    continue
                if finding.category in safety_drop_categories and finding.score >= safety_drop_threshold:
                    should_drop = True

        unique_reason_codes: list[str] = []
        seen: set[str] = set()
        for code in reason_codes:
            if code not in seen:
                seen.add(code)
                unique_reason_codes.append(code)

        preserved_learning_content = self._preserved_learning_content(context)
        redacted_identity_content = self._redacted_identity_content(findings)

        decision = DecisionType.PASS_RAW
        dataset_action = "deliver_raw"
        compliance_decision = "pass_raw"
        requires_restricted_dataset = False
        authorization_required = False
        access_control_required = False
        audit_required = False
        restricted_reason = ""
        annotation_guidance = "图片未发现需要治理的身份信息或不安全内容，可进入普通教育数据集。"

        if should_drop:
            decision = DecisionType.DROP
            dataset_action = "do_not_deliver"
            compliance_decision = "drop"
            annotation_guidance = "图片包含明显不适合进入普通教育数据集的内容，应阻断或进入人工复核。"
        elif restricted_use_case:
            requires_restricted_dataset = True
            authorization_required = True
            access_control_required = True
            audit_required = True
            if restricted_enabled and authorized_sensitive_use:
                decision = DecisionType.PASS_RAW
                dataset_action = "deliver_raw_restricted"
                compliance_decision = "restricted_raw_allowed"
                restricted_reason = f"当前任务需要使用敏感目标：{restricted_use_case}"
                annotation_guidance = "原图仅可进入授权受控敏感数据集，不得进入普通教育数据集。"
            else:
                decision = DecisionType.DROP
                dataset_action = "do_not_deliver"
                compliance_decision = "drop"
                restricted_reason = f"敏感目标任务 {restricted_use_case} 缺少授权或受控数据集策略。"
                annotation_guidance = "图片涉及受控敏感目标任务，但授权或访问控制条件不足，不得投递。"
        elif not education_value_preserved:
            decision = DecisionType.DROP
            dataset_action = "do_not_deliver"
            compliance_decision = "drop"
            review_required = True
            annotation_guidance = "自动脱敏会破坏题目、作答、板书或场景主体内容，需人工复核或不投递。"
        elif needs_redaction:
            decision = DecisionType.PASS_REDACTED
            dataset_action = (
                "deliver_redacted_with_constraints"
                if review_required or any(item in {"人脸", "胸牌工牌", "签名", "印章", "二维码", "条形码"} for item in redacted_identity_content)
                else "deliver_redacted"
            )
            compliance_decision = "pass_redacted"
            annotation_guidance = self._annotation_guidance(
                dataset_action,
                redacted_identity_content,
                preserved_learning_content,
            )
        elif not ordinary_enabled and restricted_enabled:
            decision = DecisionType.PASS_RAW
            dataset_action = "deliver_raw_restricted"
            compliance_decision = "restricted_raw_allowed"
            requires_restricted_dataset = True
            authorization_required = True
            access_control_required = True
            audit_required = True
            restricted_reason = "当前任务仅允许受控敏感数据集输出。"
            annotation_guidance = "原图不进入普通教育数据集，仅可进入受控数据集。"

        logger.info(
            "Policy decision: %s / %s (reasons=%s, profile=%s)",
            decision.value,
            dataset_action,
            unique_reason_codes,
            profile,
        )

        return PicturePolicyResult(
            decision=decision,
            dataset_action=dataset_action,
            compliance_decision=compliance_decision,
            review_required=review_required,
            requires_restricted_dataset=requires_restricted_dataset,
            authorization_required=authorization_required,
            access_control_required=access_control_required,
            audit_required=audit_required,
            restricted_reason=restricted_reason,
            redaction_strategy="minimal_identity_redaction",
            education_value_preserved=education_value_preserved,
            annotation_guidance_zh=annotation_guidance,
            preserved_learning_content=preserved_learning_content,
            redacted_identity_content=redacted_identity_content,
            executed_steps=executed_steps,
            skipped_steps=skipped_steps,
            reason_codes=unique_reason_codes,
            profile=profile,
            metadata={
                "ordinary_dataset_enabled": ordinary_enabled,
                "restricted_dataset_enabled": restricted_enabled,
                "restricted_use_case": restricted_use_case,
            },
        )

    def _preserved_learning_content(self, context: dict[str, Any]) -> list[str]:
        if context.get("ocr_executed", False):
            return ["题目", "学生作答", "板书", "课堂场景"]
        return ["课堂场景", "人体动作", "实验器材"]

    def _redacted_identity_content(self, findings: list[PictureFinding]) -> list[str]:
        mapping = {
            "person_name": "姓名",
            "phone_number": "手机号",
            "email": "邮箱",
            "id_card": "身份证号",
            "bank_card": "银行卡号",
            "bank_account": "金融账户信息",
            "address": "地址",
            "license_plate": "车牌",
            "student_id": "学号",
            "date_time": "日期时间",
            "pii_entity": "个人信息片段",
            "combined_identity": "组合可识别风险",
            "face": "人脸",
            "badge": "胸牌工牌",
            "signature": "签名",
            "stamp": "印章",
            "qr_code": "二维码",
            "barcode": "条形码",
        }
        seen: set[str] = set()
        items: list[str] = []
        for finding in findings:
            label = mapping.get(finding.category, finding.category)
            if label not in seen:
                seen.add(label)
                items.append(label)
        return items

    def _annotation_guidance(
        self,
        dataset_action: str,
        redacted_identity_content: list[str],
        preserved_learning_content: list[str],
    ) -> str:
        identity = "、".join(redacted_identity_content) if redacted_identity_content else "无"
        preserved = "、".join(preserved_learning_content) if preserved_learning_content else "教育主体内容"
        if dataset_action == "deliver_redacted_with_constraints":
            return (
                f"已对{identity}进行最小破坏脱敏，保留{preserved}，"
                "可进入普通教育数据集，但不得用于身份识别、证件识别或二维码识别任务。"
            )
        if dataset_action == "deliver_redacted":
            return f"已对{identity}进行最小破坏脱敏，保留{preserved}，可进入普通教育数据集。"
        return "图片可按原图进入普通教育数据集。"


def _finding_is_redact_only(finding: PictureFinding) -> bool:
    metadata = finding.metadata or {}
    if str(metadata.get("decision_hint") or "").strip().lower() == "redact_only":
        return True
    for item in metadata.get("evidence_regions") or []:
        if isinstance(item, dict) and str(item.get("decision_hint") or "").strip().lower() == "redact_only":
            return True
    return False


def _moderation_category_is_redact_only(moderation: PictureModerationResult, category: str) -> bool:
    metadata = moderation.metadata or {}
    violations = [
        item for item in metadata.get("violations") or []
        if isinstance(item, dict) and str(item.get("category") or "").lower() == category.lower()
    ]
    evidence = [
        item for item in metadata.get("evidence_regions") or []
        if isinstance(item, dict) and str(item.get("category") or "").lower() == category.lower()
    ]
    items = violations or evidence
    return bool(items) and all(str(item.get("decision_hint") or "").strip().lower() == "redact_only" for item in items)

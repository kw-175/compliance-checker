"""Operator-selection-driven video governance policy."""

from __future__ import annotations

from video.domain.enums import VideoGovernanceDecision
from video.domain.models import (
    ComplianceOperatorSelection,
    PreservedTrainingTarget,
    TaskContext,
    VideoGovernancePolicyResult,
    VideoRiskAnnotation,
)


def evaluate_policy(
    risks: list[VideoRiskAnnotation],
    task_context: TaskContext,
    profile: str = "default_cn_enterprise",
    operator_selection: ComplianceOperatorSelection | None = None,
    preserved_targets: list[PreservedTrainingTarget] | None = None,
) -> VideoGovernancePolicyResult:
    """Evaluate enabled operator findings; disabled operators are training targets."""
    preserved_targets = preserved_targets or []
    decision = VideoGovernanceDecision.ALLOW
    reason_codes: list[str] = []
    matched_rules: list[str] = []
    requires_review = False
    requires_transformation = False
    allow_original = True

    if preserved_targets:
        reason_codes.append("TRAINING_TARGETS_PRESERVED_BY_OPERATOR_SELECTION")
        matched_rules.append("preserve_disabled_operators")

    if risks:
        decision = VideoGovernanceDecision.ALLOW_WITH_RISK_LABELS
        reason_codes.append("ENABLED_COMPLIANCE_OPERATORS_MATCHED")
        matched_rules.append("enabled_operator_risks")

    for risk in risks:
        if risk.excluded_by_operator_selection:
            continue
        category = risk.category
        if category == "content.sexual_minor":
            decision = VideoGovernanceDecision.REJECT
            requires_review = True
            requires_transformation = False
            allow_original = False
            reason_codes.append("REJECT_ENABLED_OPERATOR_CONTENT_SEXUAL_MINOR")
            matched_rules.append("reject_enabled_critical_content")
            continue
        if risk.severity == "critical" and category.startswith("content."):
            decision = _raise_decision(decision, VideoGovernanceDecision.REVIEW_REQUIRED)
            requires_review = True
            reason_codes.append(f"REVIEW_ENABLED_OPERATOR_{_code(category)}")
            matched_rules.append("review_enabled_critical_content")
            continue
        if category.startswith("content."):
            decision = _raise_decision(decision, VideoGovernanceDecision.REVIEW_REQUIRED)
            requires_review = True
            reason_codes.append(f"REVIEW_ENABLED_OPERATOR_{_code(category)}")
            matched_rules.append("review_enabled_content")
            continue
        if category.startswith("privacy.") and risk.eligible_for_redaction:
            decision = _raise_decision(decision, VideoGovernanceDecision.TRANSFORM_REQUIRED)
            requires_transformation = True
            reason_codes.append(f"PLAN_REDACTION_FOR_ENABLED_OPERATOR_{_code(category)}")
            matched_rules.append("redact_enabled_privacy_operator")

    return VideoGovernancePolicyResult(
        decision=decision,
        requires_review=requires_review,
        requires_transformation=requires_transformation,
        requires_restricted_dataset=False,
        allow_original_for_annotation=allow_original,
        reason_codes=_dedupe(reason_codes),
        matched_rules=_dedupe(matched_rules),
        profile=profile,
        metadata={
            "task_context": task_context.model_dump(mode="json"),
            "operator_selection": operator_selection.model_dump(mode="json") if operator_selection else {},
            "risk_count": len(risks),
            "preserved_training_target_count": len(preserved_targets),
        },
    )


_DECISION_ORDER = {
    VideoGovernanceDecision.ALLOW: 0,
    VideoGovernanceDecision.ALLOW_WITH_RISK_LABELS: 1,
    VideoGovernanceDecision.TRANSFORM_REQUIRED: 2,
    VideoGovernanceDecision.REVIEW_REQUIRED: 3,
    VideoGovernanceDecision.RESTRICTED: 4,
    VideoGovernanceDecision.REJECT: 5,
}


def _raise_decision(left: VideoGovernanceDecision, right: VideoGovernanceDecision) -> VideoGovernanceDecision:
    return right if _DECISION_ORDER[right] > _DECISION_ORDER[left] else left


def _code(category: str) -> str:
    return category.upper().replace(".", "_").replace("-", "_")


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result

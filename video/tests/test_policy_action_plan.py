from __future__ import annotations

from video.application.action_planner import build_action_plan
from video.application.operator_selection import resolve_operator_selection
from video.application.policy import evaluate_policy
from video.domain.enums import VideoGovernanceDecision
from video.domain.models import TaskContext, TimeSpan, VideoRiskAnnotation


def _risk(category: str, source_modality: str = "visual") -> VideoRiskAnnotation:
    return VideoRiskAnnotation(
        source_modality=source_modality,
        category=category,
        severity="medium",
        confidence=0.9,
        span=TimeSpan(start_ms=1000, end_ms=2000),
        frame_ids=["frame_001"],
        regions=[{"frame_id": "frame_001", "bbox": {"x": 10, "y": 20, "w": 80, "h": 90}}],
        metadata={
            "tracking": {
                "redaction_ready": True,
                "redaction_scope": "sampled_frame",
                "redaction_series": [
                    {
                        "frame_id": "frame_001",
                        "pts_ms": 1000,
                        "bbox": {"x": 10, "y": 20, "w": 80, "h": 90},
                        "confidence": 0.9,
                    }
                ],
            }
        },
    )


def test_public_face_risk_requires_transform_and_renders_by_default():
    context = TaskContext(task_type="action_recognition", release_scope="public", needs_face=False)
    risks = [_risk("privacy.face")]
    policy = evaluate_policy(risks, context)
    plan = build_action_plan(risks, policy, context, options={})

    assert policy.decision == VideoGovernanceDecision.TRANSFORM_REQUIRED
    assert policy.requires_transformation is True
    assert plan.operations[0].operation == "gaussian_blur"
    assert plan.render_redacted_asset is True


def test_rendering_can_still_be_disabled_explicitly():
    context = TaskContext(task_type="action_recognition", release_scope="public", needs_face=False)
    risks = [_risk("privacy.face")]
    policy = evaluate_policy(risks, context)
    plan = build_action_plan(risks, policy, context, options={"render_redacted_asset": False})

    assert plan.operations
    assert plan.render_redacted_asset is False


def test_face_training_task_preserves_disabled_face_operator():
    context = TaskContext(task_type="face_expression")
    resolved = resolve_operator_selection({"disabled_operator_ids": ["VPI_001"], "disabled_target_types": ["face"]})
    risks: list[VideoRiskAnnotation] = []
    preserved = resolved.preserved_targets(task_type=context.task_type)
    policy = evaluate_policy(risks, context, operator_selection=resolved.selection, preserved_targets=preserved)
    plan = build_action_plan(risks, policy, context, options={"render_redacted_asset": True})

    assert policy.decision == VideoGovernanceDecision.ALLOW
    assert any(target.target_type == "face" for target in preserved)
    assert plan.operations == []
    assert plan.render_redacted_asset is False


def test_id_card_ocr_task_preserves_disabled_id_card_operator():
    context = TaskContext(task_type="id_card_ocr")
    resolved = resolve_operator_selection({"disabled_operator_ids": ["PII_003"], "disabled_target_types": ["id_card"]})
    risks: list[VideoRiskAnnotation] = []
    preserved = resolved.preserved_targets(task_type=context.task_type)
    policy = evaluate_policy(risks, context, operator_selection=resolved.selection, preserved_targets=preserved)
    plan = build_action_plan(risks, policy, context, options={"render_redacted_asset": True})

    assert policy.decision == VideoGovernanceDecision.ALLOW
    assert any(target.target_type == "id_card" for target in preserved)
    assert plan.operations == []

"""Build redaction/action plans without executing them."""

from __future__ import annotations

from video.domain.enums import VideoGovernanceDecision
from video.domain.models import (
    TaskContext,
    VideoActionPlan,
    VideoGovernancePolicyResult,
    VideoRedactionOperation,
    VideoRiskAnnotation,
)


def build_action_plan(
    risks: list[VideoRiskAnnotation],
    policy: VideoGovernancePolicyResult,
    task_context: TaskContext,
    options: dict[str, object] | None = None,
) -> VideoActionPlan:
    """Translate governance decisions into a concrete but optional action plan."""
    options = dict(options or {})
    render_requested = bool(options.get("render_redacted_asset", True))
    operations: list[VideoRedactionOperation] = []
    for risk in risks:
        operation = _operation_for_risk(risk, task_context)
        if not operation:
            continue
        tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
        if risk.source_modality not in {"audio"} and not tracking.get("redaction_ready", False):
            continue
        span = risk.display_span or risk.span
        operations.append(
            VideoRedactionOperation(
                risk_id=risk.risk_id,
                modality=risk.source_modality,
                operation=operation,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                track_id=risk.track_id,
                regions=risk.regions,
                task_impact=_task_impact(risk, task_context),
                metadata={
                    "category": risk.category,
                    "reason_codes": risk.reason_codes,
                    "display_span": span.model_dump(mode="json"),
                    "temporal_precision": risk.temporal_precision,
                    "spatial_precision": risk.spatial_precision,
                    "localization_status": risk.localization_status,
                    "redaction_scope": tracking.get("redaction_scope", ""),
                    "redaction_ready": bool(tracking.get("redaction_ready", False)),
                    "redaction_series": tracking.get("redaction_series", []),
                    "mask_keyframes": tracking.get("mask_keyframes", []),
                    "quality_flags": tracking.get("quality_flags", []),
                    "tracking_backend": tracking.get("tracking_backend", ""),
                },
            )
        )

    render_redacted_asset = (
        render_requested
        and policy.requires_transformation
        and bool(operations)
        and policy.decision != VideoGovernanceDecision.REJECT
    )
    default_action = policy.decision.value
    original_access_level = "restricted" if policy.requires_restricted_dataset else task_context.release_scope
    return VideoActionPlan(
        default_action=default_action,
        preserve_original=True,
        render_redacted_asset=render_redacted_asset,
        original_access_level=original_access_level,
        operations=operations,
        metadata={
            "render_requested": render_requested,
            "requires_transformation": policy.requires_transformation,
            "automatic_operation_count": len(operations),
            "manual_review_redaction_count": sum(1 for risk in risks if _needs_manual_redaction_review(risk)),
            "task_context": task_context.model_dump(mode="json"),
        },
    )


def _operation_for_risk(risk: VideoRiskAnnotation, task_context: TaskContext) -> str:
    if risk.excluded_by_operator_selection or not risk.eligible_for_redaction:
        return ""
    category = risk.category
    if category.startswith("content."):
        return ""
    if category == "privacy.face":
        return "gaussian_blur"
    if category in {"privacy.qr_code", "privacy.barcode", "privacy.id_card", "privacy.phone", "privacy.address", "privacy.screen_sensitive"}:
        return "black_box"
    if risk.source_modality == "audio" and category.startswith("privacy."):
        return "mute"
    if category.startswith("privacy."):
        return "black_box"
    return ""


def _task_impact(risk: VideoRiskAnnotation, task_context: TaskContext) -> str:
    if risk.excluded_by_operator_selection:
        return "preserved_training_target"
    return "redact_candidate" if risk.eligible_for_redaction else "label_only"


def _needs_manual_redaction_review(risk: VideoRiskAnnotation) -> bool:
    if risk.excluded_by_operator_selection or not risk.eligible_for_redaction:
        return False
    if not risk.category.startswith("privacy."):
        return False
    if risk.source_modality == "audio":
        return False
    tracking = risk.metadata.get("tracking") if isinstance(risk.metadata.get("tracking"), dict) else {}
    return not bool(tracking.get("redaction_ready", False))

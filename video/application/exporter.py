"""Export video compliance artifacts for platform and annotation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from video.application.services import write_json, write_jsonl
from video.application.temporal_aggregation import risk_summary
from video.domain.enums import VideoGovernanceDecision
from video.domain.models import (
    ComplianceOperatorSelection,
    PreservedTrainingTarget,
    TaskContext,
    VideoActionPlan,
    VideoGovernancePolicyResult,
    VideoRiskAnnotation,
)
from video.domain.operators import catalog_snapshot


def export_governance_artifacts(
    output_dir: Path,
    run_id: str,
    risks: list[VideoRiskAnnotation],
    policy: VideoGovernancePolicyResult,
    action_plan: VideoActionPlan,
    task_context: TaskContext,
    operator_selection: ComplianceOperatorSelection | None = None,
    preserved_targets: list[PreservedTrainingTarget] | None = None,
    display_risks: list[VideoRiskAnnotation] | None = None,
    extra_artifacts: dict[str, str] | None = None,
) -> ComplianceOutput:
    """Write governance artifacts and return the cross-modal compliance output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preserved_targets = preserved_targets or []
    display_risks = display_risks if display_risks is not None else risks
    artifact_paths = {
        "risk_annotations": output_dir / "risk_annotations.jsonl",
        "display_risks": output_dir / "display_risks.jsonl",
        "preserved_training_targets": output_dir / "preserved_training_targets.jsonl",
        "operator_selection_snapshot": output_dir / "operator_selection_snapshot.json",
        "operator_catalog_snapshot": output_dir / "operator_catalog_snapshot.json",
        "temporal_tracks": output_dir / "temporal_tracks.jsonl",
        "policy_decision": output_dir / "policy_decision.json",
        "action_plan": output_dir / "action_plan.json",
        "annotation_overlay": output_dir / "annotation_overlay.json",
        "audit_package": output_dir / "audit_package.jsonl",
        "compliance_output": output_dir / "compliance_output.json",
    }
    write_jsonl(risks, artifact_paths["risk_annotations"])
    write_jsonl(display_risks, artifact_paths["display_risks"])
    write_jsonl(preserved_targets, artifact_paths["preserved_training_targets"])
    write_json(operator_selection or ComplianceOperatorSelection(), artifact_paths["operator_selection_snapshot"])
    write_json(catalog_snapshot(), artifact_paths["operator_catalog_snapshot"])
    write_jsonl(_track_records(risks), artifact_paths["temporal_tracks"])
    write_json(policy, artifact_paths["policy_decision"])
    write_json(action_plan, artifact_paths["action_plan"])
    overlay = build_annotation_overlay(risks, policy, task_context, preserved_targets)
    write_json(overlay, artifact_paths["annotation_overlay"])
    write_jsonl(_audit_records(risks, policy, action_plan, preserved_targets), artifact_paths["audit_package"])

    output = ComplianceOutput(
        pipeline_run_id=run_id,
        modality=Modality.VIDEO,
        decision=_to_unified_decision(policy.decision),
        trust_level=TrustLevel.FULL,
        annotation_package_uri=str(artifact_paths["annotation_overlay"]),
        audit_package_uri=str(artifact_paths["audit_package"]),
        review_suggestions=_review_suggestions(risks, policy),
        explanation_summary=_explanation_summary(risks, policy),
        legacy_decision={
            "governance_decision": policy.model_dump(mode="json"),
            "action_plan": action_plan.model_dump(mode="json"),
            "risk_summary": risk_summary(risks),
            "display_risk_summary": risk_summary(display_risks),
            "preserved_training_targets": [item.model_dump(mode="json") for item in preserved_targets],
        },
        metadata={
            "artifact_paths": {
                **{key: str(value) for key, value in artifact_paths.items()},
                **dict(extra_artifacts or {}),
            },
            "task_context": task_context.model_dump(mode="json"),
            "operator_selection": (operator_selection or ComplianceOperatorSelection()).model_dump(mode="json"),
            "operator_catalog_uri": str(artifact_paths["operator_catalog_snapshot"]),
            "raw_risk_count": len(risks),
            "display_risk_count": len(display_risks),
        },
    )
    write_json(output, artifact_paths["compliance_output"])
    return output


def build_annotation_overlay(
    risks: list[VideoRiskAnnotation],
    policy: VideoGovernancePolicyResult,
    task_context: TaskContext,
    preserved_targets: list[PreservedTrainingTarget] | None = None,
) -> dict[str, Any]:
    preserved_targets = preserved_targets or []
    risk_overlays = [
        {
            "risk_id": risk.risk_id,
            "category": risk.category,
            "operator_id": risk.operator_id,
            "source_operator_id": risk.source_operator_id,
            "target_type": risk.target_type,
            "severity": risk.severity,
            "confidence": risk.confidence,
            "start_ms": risk.span.start_ms,
            "end_ms": risk.span.end_ms,
            "track_id": risk.track_id,
            "representative_frame_uri": risk.metadata.get("representative_frame_uri", ""),
            "recommended_actions": risk.recommended_actions,
        }
        for risk in risks
    ]
    return {
        "policy": policy.model_dump(mode="json"),
        "task_context": task_context.model_dump(mode="json"),
        "risk_overlays": risk_overlays,
        "timeline": risk_overlays,
        "preserved_target_overlays": [
            {
                "target_id": item.target_id,
                "target_type": item.target_type,
                "source_operator_id": item.source_operator_id,
                "video_operator_id": item.video_operator_id,
                "reason": item.reason,
                "preserved_for_task": item.preserved_for_task,
                "frame_ids": item.frame_ids,
                "regions": item.regions,
            }
            for item in preserved_targets
        ],
        "frame_overlays": [
            {
                "risk_id": risk.risk_id,
                "frame_ids": risk.frame_ids,
                "regions": risk.regions,
                "tracking": risk.metadata.get("tracking", {}),
                "category": risk.category,
                "severity": risk.severity,
            }
            for risk in risks
            if risk.regions
        ],
        "audio_overlays": [
            {
                "risk_id": risk.risk_id,
                "category": risk.category,
                "severity": risk.severity,
                "start_ms": risk.span.start_ms,
                "end_ms": risk.span.end_ms,
            }
            for risk in risks
            if risk.source_modality == "audio"
        ],
    }


def _track_records(risks: list[VideoRiskAnnotation]) -> list[dict[str, Any]]:
    return [
        {
            "track_id": risk.track_id,
            "risk_id": risk.risk_id,
            "category": risk.category,
            "source_modality": risk.source_modality,
            "start_ms": risk.span.start_ms,
            "end_ms": risk.span.end_ms,
            "frame_ids": risk.frame_ids,
            "region_count": len(risk.regions),
        }
        for risk in risks
        if risk.track_id
    ]


def _audit_records(
    risks: list[VideoRiskAnnotation],
    policy: VideoGovernancePolicyResult,
    action_plan: VideoActionPlan,
    preserved_targets: list[PreservedTrainingTarget] | None = None,
) -> list[dict[str, Any]]:
    preserved_targets = preserved_targets or []
    return [
        {"event": "risk_detected", **risk.model_dump(mode="json")}
        for risk in risks
    ] + [
        *[
            {"event": "training_target_preserved", **target.model_dump(mode="json")}
            for target in preserved_targets
        ],
        {"event": "policy_decision", **policy.model_dump(mode="json")},
        {"event": "action_plan", **action_plan.model_dump(mode="json")},
    ]


def _to_unified_decision(decision: VideoGovernanceDecision) -> UnifiedDecision:
    if decision in {VideoGovernanceDecision.ALLOW, VideoGovernanceDecision.ALLOW_WITH_RISK_LABELS}:
        return UnifiedDecision.ALLOW
    if decision == VideoGovernanceDecision.REJECT:
        return UnifiedDecision.REJECT
    if decision == VideoGovernanceDecision.TRANSFORM_REQUIRED:
        return UnifiedDecision.QUARANTINE
    return UnifiedDecision.REVIEW


def _review_suggestions(risks: list[VideoRiskAnnotation], policy: VideoGovernancePolicyResult) -> list[str]:
    suggestions = [
        f"{risk.risk_id}: {risk.category} / {risk.severity} / {risk.span.start_ms}-{risk.span.end_ms}ms"
        for risk in risks
        if risk.severity in {"medium", "high", "critical"}
    ]
    if policy.requires_review:
        suggestions.insert(0, f"Policy requires review: {', '.join(policy.reason_codes)}")
    return suggestions[:20]


def _explanation_summary(risks: list[VideoRiskAnnotation], policy: VideoGovernancePolicyResult) -> str:
    summary = risk_summary(risks)
    return (
        f"Processed video compliance risks: {summary['risk_count']} annotation(s). "
        f"Governance decision: {policy.decision.value}."
    )

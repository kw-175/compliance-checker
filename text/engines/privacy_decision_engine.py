from __future__ import annotations

from typing import Any

from text.engines.privacy_rule_engine import SENSITIVITY_RANK, TRAINING_RANK
from text.models.schemas import DetectionFinding

ACTION_RANK = {
    "retain": 0,
    "mask": 1,
    "generalize": 2,
    "restricted_review": 3,
    "drop_or_manual_review": 4,
}

ACTION_TO_DATASET_ROUTE = {
    "retain": "general_training",
    "mask": "training_after_redaction",
    "generalize": "training_after_generalization",
    "restricted_review": "restricted_privacy_review_pool",
    "drop_or_manual_review": "exclude_from_training",
}

ACTION_TO_ANNOTATION = {
    "retain": "allow_raw",
    "mask": "allow_after_redaction",
    "generalize": "allow_after_redaction",
    "restricted_review": "restricted_only",
    "drop_or_manual_review": "training_forbidden",
}


def decide_privacy_finding(
    finding: DetectionFinding,
    rule_hit: dict[str, Any],
    policy_hits: list[dict[str, Any]],
    context: dict[str, Any],
    training_context: dict[str, Any],
    custom_policy: str,
    custom_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = {
        "sensitivity_level": rule_hit.get("sensitivity_level") or "S2",
        "training_admissibility": rule_hit.get("training_admissibility") or "T1",
        "annotation_admissibility": ACTION_TO_ANNOTATION.get(str(rule_hit.get("action")), "allow_after_redaction"),
        "action": rule_hit.get("action") or "mask",
        "dataset_route": ACTION_TO_DATASET_ROUTE.get(str(rule_hit.get("action")), "training_after_redaction"),
        "allow_downstream_annotation": True,
        "requires_manual_review": bool(finding.needs_adjudication),
        "redaction_required": str(rule_hit.get("action")) in {"mask", "generalize", "restricted_review"},
    }
    decision_path = [
        {
            "stage": "api_privacy_recall",
            "outcome": finding.risk_type,
            "confidence": finding.confidence,
        },
        {
            "stage": "privacy_rule_engine",
            "outcome": rule_hit.get("rule_id", ""),
            "sensitivity_level": decision["sensitivity_level"],
            "training_admissibility": decision["training_admissibility"],
            "reason_zh": rule_hit.get("reason_zh", ""),
        },
    ]

    if policy_hits:
        top = policy_hits[0]
        _apply_policy(decision, top)
        decision_path.append(
            {
                "stage": "privacy_policy_engine",
                "outcome": top.get("policy_id", ""),
                "reason_zh": top.get("reason_zh", ""),
            }
        )
    else:
        decision_path.append({"stage": "privacy_policy_engine", "outcome": "default_rule_decision"})

    structured_override, matched_key = _structured_custom_policy_override(
        rule_hit.get("entity_type") or finding.risk_type,
        custom_policy_config,
    )
    if structured_override:
        _apply_stricter(decision, structured_override)
        decision_path.append(
            {
                "stage": "structured_custom_policy",
                "outcome": structured_override.get("action", ""),
                "matched_entity_type": matched_key,
            }
        )

    if custom_policy.strip():
        decision_path.append({"stage": "custom_policy_note", "outcome": "attached_for_audit"})

    if _training_use(training_context) and decision["training_admissibility"] in {"T2", "T3"}:
        decision["allow_downstream_annotation"] = False
        decision_path.append({"stage": "training_route", "outcome": "blocked_or_restricted_for_training"})

    return {
        **decision,
        "policy_hits": policy_hits,
        "rule_hits": [rule_hit],
        "decision_path": decision_path,
        "decision_engine_version": "privacy-decision-v1",
        "reason_zh": _reason(decision, rule_hit, policy_hits),
    }


def aggregate_privacy_document(doc_id: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return {
            "doc_id": doc_id,
            "sensitivity_level": "S1",
            "training_admissibility": "T0",
            "privacy_action": "retain",
            "dataset_route": "general_training",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
            "summary_zh": "未发现需要治理的隐私信息。",
        }
    sensitivity = max((item["sensitivity_level"] for item in decisions), key=lambda code: SENSITIVITY_RANK.get(code, 0))
    training = max((item["training_admissibility"] for item in decisions), key=lambda code: TRAINING_RANK.get(code, 0))
    action = max((item["action"] for item in decisions), key=lambda code: ACTION_RANK.get(code, 0))
    allow_annotation = all(bool(item.get("allow_downstream_annotation", True)) for item in decisions)
    manual_review = any(bool(item.get("requires_manual_review", False)) for item in decisions)
    return {
        "doc_id": doc_id,
        "sensitivity_level": sensitivity,
        "training_admissibility": training,
        "privacy_action": action,
        "dataset_route": ACTION_TO_DATASET_ROUTE.get(action, "restricted_privacy_review_pool"),
        "allow_downstream_annotation": allow_annotation,
        "requires_manual_review": manual_review,
        "finding_count": len(decisions),
        "summary_zh": (
            f"文档命中{len(decisions)}条隐私风险，最高敏感等级为{sensitivity}，"
            f"训练准入为{training}，建议处置为{_action_zh(action)}。"
        ),
    }


def _apply_policy(decision: dict[str, Any], policy_hit: dict[str, Any]) -> None:
    for key in ("sensitivity_level", "training_admissibility", "annotation_admissibility", "action", "dataset_route"):
        if policy_hit.get(key):
            decision[key] = policy_hit[key]
    if policy_hit.get("allow_downstream_annotation") is not None:
        decision["allow_downstream_annotation"] = bool(policy_hit.get("allow_downstream_annotation"))
    if policy_hit.get("requires_manual_review") is not None:
        decision["requires_manual_review"] = bool(policy_hit.get("requires_manual_review"))
    decision["redaction_required"] = decision["action"] in {"mask", "generalize", "restricted_review"}


def _apply_stricter(decision: dict[str, Any], override: dict[str, Any]) -> None:
    if SENSITIVITY_RANK.get(str(override.get("sensitivity_level")), -1) > SENSITIVITY_RANK.get(
        str(decision.get("sensitivity_level")),
        -1,
    ):
        decision["sensitivity_level"] = override["sensitivity_level"]
    if TRAINING_RANK.get(str(override.get("training_admissibility")), -1) > TRAINING_RANK.get(
        str(decision.get("training_admissibility")),
        -1,
    ):
        decision["training_admissibility"] = override["training_admissibility"]
    if ACTION_RANK.get(str(override.get("action")), -1) > ACTION_RANK.get(str(decision.get("action")), -1):
        decision["action"] = override["action"]
        decision["dataset_route"] = override.get("dataset_route") or ACTION_TO_DATASET_ROUTE.get(override["action"], "")
        decision["annotation_admissibility"] = override.get("annotation_admissibility") or ACTION_TO_ANNOTATION.get(
            override["action"],
            "",
        )
    if override.get("allow_downstream_annotation") is False:
        decision["allow_downstream_annotation"] = False
    if override.get("requires_manual_review") is not None:
        decision["requires_manual_review"] = bool(override.get("requires_manual_review")) or bool(
            decision.get("requires_manual_review")
        )
    decision["redaction_required"] = decision["action"] in {"mask", "generalize", "restricted_review"}


def _structured_custom_policy_override(entity_type: str, config: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    if not isinstance(config, dict) or not config.get("enabled"):
        return {}, ""
    entity_actions = config.get("entity_actions")
    if not isinstance(entity_actions, dict):
        return {}, ""
    action = str(entity_actions.get(entity_type) or "").strip()
    if not action:
        return {}, ""
    return _action_governance(action), entity_type


def _action_governance(action: str) -> dict[str, Any]:
    action = action.strip()
    mapping = {
        "retain": ("S1", "T0", "allow_raw", True, False),
        "mask": ("S2", "T1", "allow_after_redaction", True, False),
        "generalize": ("S3", "T1", "allow_after_redaction", True, False),
        "restricted_review": ("S3", "T2", "restricted_only", False, True),
        "drop_or_manual_review": ("S4", "T3", "training_forbidden", False, True),
    }
    sensitivity, training, annotation, allow_annotation, review = mapping.get(
        action,
        mapping["restricted_review"],
    )
    return {
        "sensitivity_level": sensitivity,
        "training_admissibility": training,
        "annotation_admissibility": annotation,
        "action": action,
        "dataset_route": ACTION_TO_DATASET_ROUTE.get(action, "restricted_privacy_review_pool"),
        "allow_downstream_annotation": allow_annotation,
        "requires_manual_review": review,
    }


def _training_use(training_context: dict[str, Any]) -> bool:
    value = str(training_context.get("downstream_use") or training_context.get("purpose") or "").lower()
    return value in {"training", "training_candidate", "model_training", "general_training", "pretraining"}


def _reason(decision: dict[str, Any], rule_hit: dict[str, Any], policy_hits: list[dict[str, Any]]) -> str:
    if policy_hits:
        return str(policy_hits[0].get("reason_zh") or rule_hit.get("reason_zh") or "")
    return str(rule_hit.get("reason_zh") or "")


def _action_zh(action: str) -> str:
    return {
        "retain": "保留",
        "mask": "脱敏遮蔽",
        "generalize": "泛化处理",
        "restricted_review": "受限复核",
        "drop_or_manual_review": "删除或人工复核",
    }.get(action, action)

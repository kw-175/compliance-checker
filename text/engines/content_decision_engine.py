from __future__ import annotations

from typing import Any

from text.engines.content_rule_engine import ContentRuleHit
from text.models.schemas import DetectionFinding, Severity

RISK_RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
ACTION_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
TRAINING_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}


def fallback_governance(severity: Severity, needs_adjudication: bool) -> dict[str, Any]:
    if severity in {Severity.CRITICAL, Severity.HIGH}:
        return {
            "risk_level_code": "C3",
            "action": "P4",
            "training_eligibility": "T3",
            "dataset_route": "exclude_from_training",
            "allow_downstream_annotation": False,
            "requires_manual_review": False,
        }
    if needs_adjudication or severity == Severity.MEDIUM:
        return {
            "risk_level_code": "C2",
            "action": "P3",
            "training_eligibility": "T2",
            "dataset_route": "safety_review_or_eval_only",
            "allow_downstream_annotation": False,
            "requires_manual_review": True,
        }
    return {
        "risk_level_code": "C1",
        "action": "P2",
        "training_eligibility": "T1",
        "dataset_route": "restricted_training_after_review",
        "allow_downstream_annotation": True,
        "requires_manual_review": False,
    }


def _matching_rule_hits(
    finding: DetectionFinding,
    rule_hits: list[ContentRuleHit],
) -> list[dict[str, Any]]:
    labels = {
        finding.policy_tag.lower(),
        finding.risk_type.lower(),
        str(finding.attributes.get("content_safety", {}).get("matched_label") or "").lower(),
    }
    labels.update(
        str(item).lower()
        for item in finding.attributes.get("content_safety", {}).get("label_hierarchy", [])
        if item
    )
    matched: list[dict[str, Any]] = []
    for hit in rule_hits:
        hit_labels = {hit.policy_tag.lower(), hit.risk_type.lower()}
        if _label_sets_match(labels, hit_labels):
            matched.append(hit.as_dict())
    return matched


def _label_sets_match(left: set[str], right: set[str]) -> bool:
    for item in left:
        for candidate in right:
            if not item or not candidate:
                continue
            if item == candidate:
                return True
            if item.startswith(candidate + ".") or candidate.startswith(item + "."):
                return True
    return False


def _api_recommendation(finding: DetectionFinding) -> dict[str, Any]:
    attrs = finding.attributes.get("content_safety", {})
    allowed = {
        "risk_level_code": {"C0", "C1", "C2", "C3"},
        "action": {"P0", "P1", "P2", "P3", "P4", "P5"},
        "training_eligibility": {"T0", "T1", "T2", "T3"},
    }
    mapping = {
        "risk_level_code": "api_recommended_risk_level",
        "action": "api_recommended_action",
        "training_eligibility": "api_recommended_training_eligibility",
        "dataset_route": "api_recommended_dataset_route",
        "allow_downstream_annotation": "api_allow_downstream_annotation",
    }
    result: dict[str, Any] = {}
    for target, source in mapping.items():
        value = attrs.get(source)
        if value in ("", None):
            continue
        if target in allowed and str(value) not in allowed[target]:
            continue
        result[target] = value
    return result


def _apply_more_strict(decision: dict[str, Any], candidate: dict[str, Any]) -> None:
    _max_code(decision, candidate, "risk_level_code", RISK_RANK)
    _max_code(decision, candidate, "action", ACTION_RANK)
    _max_code(decision, candidate, "training_eligibility", TRAINING_RANK)
    if candidate.get("dataset_route") and ACTION_RANK.get(str(candidate.get("action")), -1) >= ACTION_RANK.get(
        str(decision.get("action")),
        -1,
    ):
        decision["dataset_route"] = candidate["dataset_route"]
    if candidate.get("allow_downstream_annotation") is False:
        decision["allow_downstream_annotation"] = False
    if candidate.get("requires_manual_review") is not None:
        decision["requires_manual_review"] = bool(candidate.get("requires_manual_review")) or bool(
            decision.get("requires_manual_review")
        )


def _max_code(decision: dict[str, Any], candidate: dict[str, Any], key: str, rank: dict[str, int]) -> None:
    value = str(candidate.get(key) or "")
    if value and rank.get(value, -1) > rank.get(str(decision.get(key)), -1):
        decision[key] = value


def _semantic_adjudication(finding: DetectionFinding) -> dict[str, Any]:
    attrs = finding.attributes.get("content_safety", {})
    semantic = attrs.get("semantic_adjudication")
    if not isinstance(semantic, dict):
        semantic = {}
    return {
        "risk_level_code": attrs.get("semantic_risk_level") or semantic.get("final_risk_level") or "",
        "action": attrs.get("semantic_action") or semantic.get("final_action") or "",
        "training_eligibility": attrs.get("semantic_training_eligibility") or semantic.get("final_training_eligibility") or "",
        "dataset_route": attrs.get("semantic_dataset_route") or semantic.get("final_dataset_route") or "",
        "allow_downstream_annotation": attrs.get("semantic_allow_downstream_annotation", semantic.get("allow_downstream_annotation")),
        "requires_manual_review": attrs.get("semantic_requires_manual_review", semantic.get("requires_manual_review")),
        "downgrade_allowed": bool(attrs.get("semantic_downgrade_allowed", semantic.get("downgrade_allowed", False))),
        "upgrade_required": bool(attrs.get("semantic_upgrade_required", semantic.get("upgrade_required", False))),
        "semantic_decision": attrs.get("semantic_decision") or semantic.get("semantic_decision") or "",
        "context_type": attrs.get("semantic_context_type") or semantic.get("context_type") or "",
        "reasoning_summary": attrs.get("semantic_reasoning_summary") or semantic.get("reasoning_summary") or "",
    }


def _can_semantic_downgrade(
    finding: DetectionFinding,
    context: dict[str, Any],
    semantic: dict[str, Any],
) -> bool:
    if not semantic.get("downgrade_allowed"):
        return False
    audience = str(context.get("audience") or context.get("target_audience") or "").lower()
    if audience in {"minor", "minors", "student", "students", "child", "children"}:
        return False
    haystack = " ".join([finding.policy_tag, finding.risk_type, finding.explanation]).lower()
    strict_labels = {"minor_harmful", "self_harm", "sexual exploitation", "child"}
    if any(label in haystack for label in strict_labels):
        return False
    return True


def _apply_semantic_adjudication(
    decision: dict[str, Any],
    finding: DetectionFinding,
    context: dict[str, Any],
    decision_path: list[dict[str, Any]],
) -> None:
    semantic = _semantic_adjudication(finding)
    if not semantic.get("semantic_decision"):
        return

    decision_path.append(
        {
            "stage": "semantic_adjudication",
            "outcome": semantic["semantic_decision"],
            "context_type": semantic.get("context_type", ""),
            "reason": semantic.get("reasoning_summary", ""),
        }
    )
    candidate = {
        "risk_level_code": semantic.get("risk_level_code"),
        "action": semantic.get("action"),
        "training_eligibility": semantic.get("training_eligibility"),
        "dataset_route": semantic.get("dataset_route"),
        "allow_downstream_annotation": semantic.get("allow_downstream_annotation"),
        "requires_manual_review": semantic.get("requires_manual_review"),
    }
    if semantic.get("upgrade_required"):
        _apply_more_strict(decision, candidate)
        return
    if not _can_semantic_downgrade(finding, context, semantic):
        return

    # Context can move a first-pass unsafe recall to review, never directly to allow.
    decision["risk_level_code"] = "C2" if candidate.get("risk_level_code") in {"", "C0", "C1", "C2"} else str(candidate["risk_level_code"])
    decision["action"] = "P3"
    decision["training_eligibility"] = "T2"
    decision["dataset_route"] = candidate.get("dataset_route") or "safety_review_or_eval_only"
    decision["allow_downstream_annotation"] = False
    decision["requires_manual_review"] = True


def _custom_policy_intensification(finding: DetectionFinding, custom_policy: str) -> dict[str, Any]:
    policy = custom_policy.strip().lower()
    if not policy:
        return {}
    haystack = " ".join(
        [
            finding.policy_tag,
            finding.risk_type,
            finding.explanation,
            str(finding.attributes.get("content_safety", {}).get("matched_label") or ""),
        ]
    ).lower()
    strict_words = {"exclude", "block", "ban", "reject", "禁止", "阻止", "剔除", "排除"}
    if not any(word in policy for word in strict_words):
        return {}
    if "all" in policy or "全部" in policy:
        matched_scope = True
    elif ("jailbreak" in policy or "prompt injection" in policy or "prompt-injection" in policy) and (
        "jailbreak" in haystack or "prompt" in haystack
    ):
        matched_scope = True
    elif any(token in policy and token in haystack for token in ["violence", "violent", "hate", "porn", "self_harm"]):
        matched_scope = True
    else:
        matched_scope = False
    if not matched_scope:
        return {}
    return {
        "risk_level_code": "C3",
        "action": "P4",
        "training_eligibility": "T3",
        "dataset_route": "exclude_from_training",
        "allow_downstream_annotation": False,
        "requires_manual_review": False,
    }


def _structured_action_governance(action: str) -> dict[str, Any]:
    action = str(action or "").upper()
    if action in {"P4", "P5"}:
        return {
            "risk_level_code": "C3",
            "action": action,
            "training_eligibility": "T3",
            "dataset_route": "exclude_from_training",
            "allow_downstream_annotation": False,
            "requires_manual_review": False,
        }
    if action == "P3":
        return {
            "risk_level_code": "C2",
            "action": "P3",
            "training_eligibility": "T2",
            "dataset_route": "safety_review_or_eval_only",
            "allow_downstream_annotation": False,
            "requires_manual_review": True,
        }
    if action == "P2":
        return {
            "risk_level_code": "C1",
            "action": "P2",
            "training_eligibility": "T1",
            "dataset_route": "restricted_training_after_review",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
        }
    if action == "P1":
        return {
            "risk_level_code": "C1",
            "action": "P1",
            "training_eligibility": "T0",
            "dataset_route": "general_training",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
        }
    return {
        "risk_level_code": "C0",
        "action": "P0",
        "training_eligibility": "T0",
        "dataset_route": "general_training",
        "allow_downstream_annotation": True,
        "requires_manual_review": False,
    }


def _structured_custom_policy_override(
    finding: DetectionFinding,
    custom_policy_config: dict[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(custom_policy_config, dict) or not custom_policy_config.get("enabled"):
        return {}, ""
    risk_actions = custom_policy_config.get("risk_actions")
    if not isinstance(risk_actions, dict):
        return {}, ""
    content_attrs = finding.attributes.get("content_safety", {})
    labels = {
        finding.policy_tag.lower(),
        finding.risk_type.lower(),
        str(content_attrs.get("matched_label") or "").lower(),
    }
    labels.update(
        str(item).lower()
        for item in content_attrs.get("label_hierarchy", [])
        if item
    )
    best_action = ""
    best_label = ""
    for label, action in risk_actions.items():
        action = str(action or "").upper()
        if action not in ACTION_RANK:
            continue
        if _label_sets_match(labels, {str(label).lower()}):
            if ACTION_RANK[action] > ACTION_RANK.get(best_action, -1):
                best_action = action
                best_label = str(label)
    if not best_action:
        return {}, ""
    return _structured_action_governance(best_action), best_label


def decide_content_finding(
    finding: DetectionFinding,
    rule_hits: list[ContentRuleHit],
    policy_hits: list[dict[str, Any]],
    context: dict[str, Any],
    training_context: dict[str, Any],
    custom_policy: str,
    custom_policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = fallback_governance(finding.severity, finding.needs_adjudication)
    decision_path = [
        {
            "stage": "base_api_recall",
            "outcome": finding.policy_tag,
            "severity": finding.severity.value,
            "confidence": finding.confidence,
        }
    ]

    matched_rules = _matching_rule_hits(finding, rule_hits)
    if matched_rules:
        decision_path.append(
            {
                "stage": "rule_recall",
                "outcome": "matched",
                "rule_ids": [item["rule_id"] for item in matched_rules],
            }
        )

    if policy_hits:
        top = policy_hits[0]
        for key in (
            "risk_level_code",
            "action",
            "training_eligibility",
            "dataset_route",
            "allow_downstream_annotation",
            "requires_manual_review",
        ):
            if top.get(key) not in ("", None):
                decision[key] = top[key]
        decision_path.append(
            {
                "stage": "policy_engine",
                "outcome": top["policy_id"],
                "reason": top.get("reason", ""),
            }
        )
    else:
        api_recommendation = _api_recommendation(finding)
        if api_recommendation:
            decision.update(api_recommendation)
            decision_path.append(
                {
                    "stage": "semantic_api_recommendation",
                    "outcome": "applied_as_fallback",
                }
            )
        decision_path.append(
            {
                "stage": "policy_engine",
                "outcome": "fallback_governance",
            }
        )

    _apply_semantic_adjudication(decision, finding, context, decision_path)

    audience = str(context.get("audience") or context.get("target_audience") or "").lower()
    downstream_use = str(training_context.get("downstream_use") or training_context.get("purpose") or "").lower()
    if audience in {"minor", "minors", "student", "students", "child", "children"} and finding.severity in {
        Severity.HIGH,
        Severity.CRITICAL,
    }:
        decision.update(
            {
                "risk_level_code": "C3",
                "action": "P4",
                "training_eligibility": "T3",
                "dataset_route": "exclude_from_training",
                "allow_downstream_annotation": False,
                "requires_manual_review": False,
            }
        )
        decision_path.append({"stage": "scene_context", "outcome": "minor_audience_strict_route"})

    if custom_policy.strip():
        override = _custom_policy_intensification(finding, custom_policy)
        if override:
            _apply_more_strict(decision, override)
            decision_path.append({"stage": "custom_policy", "outcome": "strict_override"})
        else:
            decision_path.append({"stage": "custom_policy", "outcome": "attached_for_audit"})

    structured_override, structured_label = _structured_custom_policy_override(finding, custom_policy_config)
    if structured_override:
        _apply_more_strict(decision, structured_override)
        decision_path.append(
            {
                "stage": "structured_custom_policy",
                "outcome": structured_override.get("action", ""),
                "matched_label": structured_label,
            }
        )

    if downstream_use in {"training_candidate", "model_training", "pretraining"} and decision["action"] in {"P3", "P4"}:
        decision["allow_downstream_annotation"] = False
        decision_path.append({"stage": "training_route", "outcome": "annotation_blocked_for_training_risk"})

    return {
        **decision,
        "policy_hits": policy_hits,
        "rule_hits": matched_rules,
        "decision_path": decision_path,
        "decision_engine_version": "content-decision-v2",
    }

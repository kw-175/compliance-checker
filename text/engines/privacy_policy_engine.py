from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_privacy_policies(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Privacy policy file could not be loaded: %s", exc)
        return []
    policies = data.get("policies", [])
    if not isinstance(policies, list):
        return []
    return [item for item in policies if isinstance(item, dict) and item.get("enabled", True)]


def match_privacy_policy_hits(
    entity_type: str,
    policies: list[dict[str, Any]],
    context: dict[str, Any],
    training_context: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = {entity_type.lower()}
    hits: list[dict[str, Any]] = []
    for policy in policies:
        if not _matches_list(labels, policy.get("entity_types")):
            continue
        if not _context_allowed(policy, context, training_context):
            continue
        decision = policy.get("decision") if isinstance(policy.get("decision"), dict) else {}
        hits.append(
            {
                "policy_id": str(policy.get("policy_id") or "privacy_policy"),
                "policy_name_zh": str(policy.get("name_zh") or policy.get("policy_id") or "隐私治理策略"),
                "hit": True,
                "priority": int(policy.get("priority", 0)),
                "reason_zh": str(decision.get("reason_zh") or "命中隐私治理策略。"),
                "sensitivity_level": str(decision.get("sensitivity_level") or ""),
                "training_admissibility": str(decision.get("training_admissibility") or ""),
                "annotation_admissibility": str(decision.get("annotation_admissibility") or ""),
                "action": str(decision.get("action") or ""),
                "dataset_route": str(decision.get("dataset_route") or ""),
                "allow_downstream_annotation": decision.get("allow_downstream_annotation"),
                "requires_manual_review": decision.get("requires_manual_review"),
            }
        )
    return sorted(hits, key=lambda item: item["priority"], reverse=True)


def _matches_list(actual: set[str], values: Any) -> bool:
    if not values:
        return True
    allowed = {str(item).lower() for item in values if item}
    return bool(actual.intersection(allowed))


def _context_allowed(policy: dict[str, Any], context: dict[str, Any], training_context: dict[str, Any]) -> bool:
    source_values = policy.get("sources")
    if source_values and not _value_in(source_values, context.get("source") or context.get("source_type")):
        return False
    purpose_values = policy.get("purposes")
    if purpose_values:
        purpose = (
            context.get("purpose")
            or training_context.get("purpose")
            or training_context.get("downstream_use")
            or context.get("scene")
        )
        if not _value_in(purpose_values, purpose):
            return False
    subject_values = policy.get("subject_types")
    if subject_values:
        subject = context.get("subject_type") or context.get("audience") or "unknown"
        if not _value_in(subject_values, subject):
            return False
    return True


def _value_in(values: Any, actual: Any) -> bool:
    allowed = {str(item).lower() for item in values if item}
    return str(actual or "").lower() in allowed

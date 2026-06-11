from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from text.models.schemas import DetectionFinding, Severity

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


def load_content_safety_policies(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Content safety policy file could not be loaded: %s", exc)
        return []
    policies = data.get("policies", [])
    if not isinstance(policies, list):
        return []
    return [item for item in policies if isinstance(item, dict) and item.get("enabled", True)]


def _policy_labels(policy: dict[str, Any]) -> set[str]:
    labels = {str(item).lower() for item in policy.get("labels", []) if item}
    labels.update(str(item).lower() for item in policy.get("risk_types", []) if item)
    return labels


def _severity_allowed(finding: DetectionFinding, policy: dict[str, Any]) -> bool:
    severities = policy.get("severity_any_of") or policy.get("severities")
    if not severities:
        minimum = str(policy.get("min_severity") or "").lower()
        if not minimum:
            return True
        try:
            return SEVERITY_ORDER[finding.severity] >= SEVERITY_ORDER[Severity(minimum)]
        except ValueError:
            return True
    allowed = {str(item).lower() for item in severities if item}
    return finding.severity.value in allowed


def _context_allowed(policy: dict[str, Any], context: dict[str, Any]) -> bool:
    context_any_of = policy.get("context_type_any_of") or policy.get("contexts")
    if not context_any_of:
        return True
    allowed = {str(item).lower() for item in context_any_of if item}
    actual = {
        str(context.get("context_type") or "").lower(),
        str(context.get("scene") or "").lower(),
        str(context.get("source_type") or "").lower(),
        str(context.get("domain") or "").lower(),
    }
    return bool(allowed.intersection(actual))


def _training_allowed(policy: dict[str, Any], context: dict[str, Any]) -> bool:
    training_any_of = policy.get("downstream_use_any_of")
    if not training_any_of:
        return True
    allowed = {str(item).lower() for item in training_any_of if item}
    training_context = context.get("training_context")
    if not isinstance(training_context, dict):
        training_context = {}
    actual = {
        str(training_context.get("downstream_use") or "").lower(),
        str(training_context.get("purpose") or "").lower(),
        str(context.get("downstream_use") or "").lower(),
    }
    return bool(allowed.intersection(actual))


def match_policy_hits(
    finding: DetectionFinding,
    policies: list[dict[str, Any]],
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = {
        finding.policy_tag.lower(),
        finding.risk_type.lower(),
    }
    content_attrs = finding.attributes.get("content_safety", {})
    labels.add(str(content_attrs.get("matched_label") or "").lower())
    labels.update(str(item).lower() for item in content_attrs.get("label_hierarchy", []) if item)

    hits: list[dict[str, Any]] = []
    for policy in policies:
        policy_labels = _policy_labels(policy)
        if policy_labels and not _labels_intersect(labels, policy_labels):
            continue
        if not _severity_allowed(finding, policy):
            continue
        if not _context_allowed(policy, context):
            continue
        if not _training_allowed(policy, context):
            continue
        audience = str(policy.get("audience") or "")
        if audience and audience != str(context.get("audience") or ""):
            continue
        hits.append(
            {
                "policy_id": str(policy.get("policy_id") or "content_safety_policy"),
                "hit": True,
                "confidence": float(policy.get("confidence", finding.confidence)),
                "reason": str(policy.get("reason") or "Matched content-safety governance policy."),
                "evidence": [finding.span.text] if finding.span else [],
                "risk_level_code": str(policy.get("risk_level_code") or policy.get("risk_level") or ""),
                "action": str(policy.get("action") or ""),
                "training_eligibility": str(policy.get("training_eligibility") or ""),
                "dataset_route": str(policy.get("dataset_route") or ""),
                "allow_downstream_annotation": policy.get("allow_downstream_annotation"),
                "requires_manual_review": policy.get("requires_manual_review"),
                "priority": int(policy.get("priority", 0)),
            }
        )
    return sorted(hits, key=lambda item: item["priority"], reverse=True)


def _labels_intersect(labels: set[str], policy_labels: set[str]) -> bool:
    for label in labels:
        for policy_label in policy_labels:
            if not label or not policy_label:
                continue
            if label == policy_label:
                return True
            if label.startswith(policy_label + "."):
                return True
    return False

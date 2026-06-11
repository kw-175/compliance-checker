from __future__ import annotations

from typing import Any


def snippet_window(text: str, start: int, end: int, size: int = 240) -> str:
    left = max(0, start - size)
    right = min(len(text), end + size)
    return text[left:right]


def compact_privacy_finding_payload(finding: Any) -> dict[str, Any]:
    span = getattr(finding, "span", None)
    privacy_context = dict(getattr(finding, "attributes", {}) or {}).get("privacy_context", {}) or {}
    return {
        "finding_id": getattr(finding, "finding_id", ""),
        "risk_type": getattr(finding, "risk_type", ""),
        "policy_tag": getattr(finding, "policy_tag", ""),
        "severity": getattr(getattr(finding, "severity", None), "value", str(getattr(finding, "severity", ""))),
        "confidence": getattr(finding, "confidence", 0.0),
        "explanation": getattr(finding, "explanation", ""),
        "needs_adjudication": getattr(finding, "needs_adjudication", False),
        "hard_case_reason": getattr(finding, "hard_case_reason", ""),
        "privacy_context": privacy_context,
        "span": (
            {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "context_before": span.context_before,
                "context_after": span.context_after,
            }
            if span
            else None
        ),
    }


def compact_content_finding_payload(finding: Any) -> dict[str, Any]:
    attrs = dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}
    span = getattr(finding, "span", None)
    return {
        "finding_id": getattr(finding, "finding_id", ""),
        "risk_type": getattr(finding, "risk_type", ""),
        "policy_tag": getattr(finding, "policy_tag", ""),
        "severity": getattr(getattr(finding, "severity", None), "value", str(getattr(finding, "severity", ""))),
        "confidence": getattr(finding, "confidence", 0.0),
        "explanation": getattr(finding, "explanation", ""),
        "needs_adjudication": getattr(finding, "needs_adjudication", False),
        "hard_case_reason": getattr(finding, "hard_case_reason", ""),
        "context_type": attrs.get("semantic_context_type") or attrs.get("api_context_type") or "",
        "context_rationale": attrs.get("semantic_reasoning_summary") or attrs.get("api_context_rationale") or "",
        "span": (
            {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "context_before": span.context_before,
                "context_after": span.context_after,
            }
            if span
            else None
        ),
    }


def compact_fragment_adjudication_payload(record: Any) -> dict[str, Any]:
    data = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
    keys = [
        "finding_id",
        "risk_type",
        "fragment_truth",
        "governance_action",
        "requires_manual_review",
        "training_impact",
        "annotation_impact",
        "explanation",
        "confidence",
        "semantic_role",
        "operationality",
        "audience_risk",
        "protective_context",
        "recommended_action",
        "training_eligibility",
        "allow_downstream_annotation",
    ]
    return {key: data.get(key) for key in keys if key in data}


def summarize_privacy_findings(findings: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    return [compact_privacy_finding_payload(item) for item in findings[:limit]]


def summarize_content_findings(findings: list[Any], limit: int = 12) -> list[dict[str, Any]]:
    return [compact_content_finding_payload(item) for item in findings[:limit]]


def summarize_fragment_adjudications(records: list[Any], limit: int = 16) -> list[dict[str, Any]]:
    return [compact_fragment_adjudication_payload(item) for item in records[:limit]]

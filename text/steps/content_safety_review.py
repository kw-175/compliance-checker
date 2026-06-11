from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

RISK_RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
ACTION_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
TRAINING_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}
ROUTE_BY_TRAINING = {
    "T0": "general_training",
    "T1": "restricted_training_after_review",
    "T2": "safety_review_or_eval_only",
    "T3": "exclude_from_training",
}


def build_review_tasks(
    safety_results: list,
    document_context_by_doc: dict[str, dict[str, Any]] | None = None,
    candidate_windows_by_doc: dict[str, list[dict[str, Any]]] | None = None,
    localized_fragment_by_finding: dict[str, dict[str, Any]] | None = None,
    fragment_adjudication_by_finding: dict[str, dict[str, Any]] | None = None,
    document_assessment_by_doc: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    document_context_by_doc = document_context_by_doc or {}
    candidate_windows_by_doc = candidate_windows_by_doc or {}
    localized_fragment_by_finding = localized_fragment_by_finding or {}
    fragment_adjudication_by_finding = fragment_adjudication_by_finding or {}
    document_assessment_by_doc = document_assessment_by_doc or {}
    tasks: list[dict[str, Any]] = []
    for result in safety_results:
        for finding in list(getattr(result, "findings", []) or []):
            attrs = dict(getattr(finding, "attributes", {}) or {}).get("content_safety", {}) or {}
            if not _requires_review(finding, attrs):
                continue
            span = getattr(finding, "span", None)
            doc_id = getattr(result, "doc_id", "")
            finding_id = getattr(finding, "finding_id", "")
            localized_fragment = localized_fragment_by_finding.get(finding_id, {})
            window_id = str(localized_fragment.get("window_id") or attrs.get("candidate_window_id") or "")
            candidate_window = next(
                (
                    item
                    for item in candidate_windows_by_doc.get(doc_id, [])
                    if str(item.get("window_id") or "") == window_id
                ),
                {},
            )
            tasks.append(
                {
                    "review_task_id": _task_id(getattr(result, "run_id", ""), doc_id, finding_id),
                    "run_id": getattr(result, "run_id", ""),
                    "doc_id": doc_id,
                    "finding_id": finding_id,
                    "status": "pending",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "risk_type": getattr(finding, "risk_type", ""),
                    "policy_tag": getattr(finding, "policy_tag", ""),
                    "matched_label": attrs.get("matched_label", ""),
                    "label_hierarchy": attrs.get("label_hierarchy", []),
                    "risk_subcategory": attrs.get("risk_subcategory", ""),
                    "severity": getattr(getattr(finding, "severity", ""), "value", str(getattr(finding, "severity", ""))),
                    "confidence": getattr(finding, "confidence", 0.0),
                    "evidence": {
                        "text": getattr(span, "text", "") if span else "",
                        "start": getattr(span, "start", None) if span else None,
                        "end": getattr(span, "end", None) if span else None,
                        "explanation": getattr(finding, "explanation", ""),
                    },
                    "current_decision": {
                        "risk_level_code": attrs.get("risk_level_code", ""),
                        "action": attrs.get("action", ""),
                        "training_eligibility": attrs.get("training_eligibility", ""),
                        "dataset_route": attrs.get("dataset_route", ""),
                        "allow_downstream_annotation": attrs.get("allow_downstream_annotation", False),
                        "requires_manual_review": attrs.get("requires_manual_review", True),
                    },
                    "review_options": [
                        "confirm_violation",
                        "downgrade_to_review",
                        "allow_with_restriction",
                        "false_positive",
                    ],
                    "document_context": document_context_by_doc.get(doc_id, {}),
                    "candidate_window": candidate_window,
                    "localized_fragment": localized_fragment,
                    "fragment_adjudication": fragment_adjudication_by_finding.get(finding_id, {}),
                    "document_assessment": document_assessment_by_doc.get(doc_id, {}),
                    "semantic_adjudication": attrs.get("semantic_adjudication", {}),
                    "policy_hits": attrs.get("policy_hits", []),
                    "rule_hits": attrs.get("rule_hits", []),
                    "decision_path": attrs.get("decision_path", []),
                    "audit": {
                        "decision_engine_version": attrs.get("decision_engine_version", ""),
                        "policy_version": attrs.get("policy_version", ""),
                        "source_tool": getattr(finding, "source_tool", ""),
                    },
                }
            )
    return tasks


def merge_review_results(tasks: list[dict[str, Any]], submissions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("review_task_id") or ""): item for item in tasks}
    merged: list[dict[str, Any]] = []
    for submission in submissions:
        task_id = str(submission.get("review_task_id") or "")
        base = dict(by_id.get(task_id, {}))
        merged.append(
            {
                **base,
                "review_task_id": task_id,
                "status": "reviewed",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewer_id": str(submission.get("reviewer_id") or ""),
                "review_decision": str(submission.get("review_decision") or ""),
                "review_reason": str(submission.get("review_reason") or ""),
                "final_risk_level": str(submission.get("final_risk_level") or base.get("current_decision", {}).get("risk_level_code", "")),
                "final_action": str(submission.get("final_action") or base.get("current_decision", {}).get("action", "")),
                "final_training_eligibility": str(
                    submission.get("final_training_eligibility")
                    or base.get("current_decision", {}).get("training_eligibility", "")
                ),
                "final_dataset_route": str(submission.get("final_dataset_route") or base.get("current_decision", {}).get("dataset_route", "")),
                "notes": str(submission.get("notes") or ""),
            }
        )
    return merged


def build_final_decisions(
    decision_records: list[dict[str, Any]],
    review_tasks: list[dict[str, Any]],
    review_results: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    review_results = review_results or []
    tasks_by_doc: dict[str, list[dict[str, Any]]] = {}
    for task in review_tasks:
        tasks_by_doc.setdefault(str(task.get("doc_id") or ""), []).append(task)

    reviewed_by_finding = {
        str(result.get("finding_id") or ""): result
        for result in review_results
        if str(result.get("finding_id") or "")
    }

    final_records: list[dict[str, Any]] = []
    for record in decision_records:
        final_record = dict(record)
        doc_tasks = tasks_by_doc.get(str(record.get("doc_id") or ""), [])
        doc_reviews = [
            reviewed_by_finding.get(str(task.get("finding_id") or ""))
            for task in doc_tasks
            if reviewed_by_finding.get(str(task.get("finding_id") or ""))
        ]
        if not doc_tasks:
            review_status = "not_required"
        elif len(doc_reviews) >= len(doc_tasks):
            review_status = "reviewed"
        else:
            review_status = "pending"

        if doc_reviews:
            _apply_reviews_to_decision(final_record, doc_reviews)
        final_record["review_status"] = review_status
        final_record["review_required"] = bool(doc_tasks)
        final_record["review_task_count"] = len(doc_tasks)
        final_record["reviewed_task_count"] = len(doc_reviews)
        final_record["review_results"] = doc_reviews
        final_record["final_decision_source"] = "human_review" if doc_reviews else "initial_policy_decision"
        final_record.setdefault("metadata", {})
        final_record["metadata"] = {
            **dict(final_record.get("metadata") or {}),
            "final_decision_artifact_version": "content-safety-final-v1",
        }
        final_records.append(final_record)
    return final_records


def _requires_review(finding: Any, attrs: dict[str, Any]) -> bool:
    if bool(attrs.get("requires_manual_review")):
        return True
    if bool(getattr(finding, "needs_adjudication", False)):
        return True
    return str(attrs.get("action") or "") == "P3"


def _task_id(run_id: str, doc_id: str, finding_id: str) -> str:
    return hashlib.sha256(f"{run_id}|{doc_id}|{finding_id}".encode("utf-8")).hexdigest()[:16]


def _apply_reviews_to_decision(record: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    risk_level = str(record.get("risk_level") or "C0")
    action = str(record.get("decision") or "P0")
    training = str(record.get("training_eligibility") or "T0")
    dataset_route = str(record.get("dataset_route") or ROUTE_BY_TRAINING.get(training, "general_training"))
    allow_annotation = bool(record.get("allow_downstream_annotation", True))

    for review in reviews:
        if review.get("final_risk_level"):
            risk_level = str(review["final_risk_level"])
        else:
            risk_level = _max_code(risk_level, str(review.get("final_risk_level") or ""), RISK_RANK)
        if review.get("final_action"):
            action = str(review["final_action"])
        else:
            action = _max_code(action, str(review.get("final_action") or ""), ACTION_RANK)
        if review.get("final_training_eligibility"):
            training = str(review["final_training_eligibility"])
        else:
            training = _max_code(training, str(review.get("final_training_eligibility") or ""), TRAINING_RANK)
        if review.get("final_dataset_route"):
            dataset_route = str(review["final_dataset_route"])
        if str(review.get("review_decision") or "") in {"false_positive", "allow_with_restriction"}:
            allow_annotation = ACTION_RANK.get(action, 0) < ACTION_RANK["P3"]

    record["risk_level"] = risk_level
    record["decision"] = action
    record["training_eligibility"] = training
    record["dataset_route"] = dataset_route or ROUTE_BY_TRAINING.get(training, "general_training")
    record["allow_downstream_annotation"] = allow_annotation and ACTION_RANK.get(action, 0) < ACTION_RANK["P3"]
    record["needs_manual_review"] = False
    record["summary"] = f"{record.get('summary', '')} Human review applied.".strip()


def _max_code(left: str, right: str, rank: dict[str, int]) -> str:
    if not right:
        return left
    return right if rank.get(right, -1) > rank.get(left, -1) else left

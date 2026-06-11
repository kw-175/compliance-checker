from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

TRAINING_RANK = {"allowed": 0, "restricted": 1, "blocked": 2}


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
            "final_decision_artifact_version": "privacy-final-v1",
        }
        final_records.append(final_record)
    return final_records


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
                "final_privacy_action": str(submission.get("final_privacy_action") or base.get("privacy_action", "")),
                "final_training_suitability": str(
                    submission.get("final_training_suitability")
                    or base.get("document_assessment", {}).get("training_suitability", "")
                ),
                "final_annotation_suitability": str(
                    submission.get("final_annotation_suitability")
                    or base.get("document_assessment", {}).get("annotation_suitability", "")
                ),
                "notes": str(submission.get("notes") or ""),
            }
        )
    return merged


def _apply_reviews_to_decision(record: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    action = str(record.get("privacy_action") or record.get("action") or "retain")
    training = str(record.get("training_admissibility") or record.get("training_suitability") or "allowed")
    annotation = str(record.get("annotation_admissibility") or record.get("annotation_suitability") or "allowed")
    explanation = str(record.get("summary_zh") or "")

    for review in reviews:
        if review.get("final_privacy_action"):
            action = str(review["final_privacy_action"])
        if review.get("final_training_suitability"):
            training = str(review["final_training_suitability"])
        if review.get("final_annotation_suitability"):
            annotation = str(review["final_annotation_suitability"])
        if review.get("review_reason"):
            explanation = f"{explanation} 人工复核说明：{review['review_reason']}".strip()

    record["privacy_action"] = action
    record["training_suitability"] = training
    record["annotation_suitability"] = annotation
    record["summary_zh"] = explanation.strip()


def _max_training(left: str, right: str) -> str:
    return right if TRAINING_RANK.get(right, -1) > TRAINING_RANK.get(left, -1) else left

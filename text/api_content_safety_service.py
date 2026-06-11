from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient
from text.api_steps.api_safety_moderation import _load_label_catalog, _resolve_label, _safe_float, _severity
from text.config.settings import Settings, get_settings
from text.content_safety_registry import resolve_selected_sub_operators
from text.models.schemas import (
    ContentSafetyBatchCheckRequest,
    ContentSafetyBatchCheckResponse,
    ContentSafetyCheckRecord,
    ContentSafetyDecisionRecord,
    ContentSafetyEvidenceItem,
    ContentSafetyPolicyHit,
)
from text.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

RISK_PRIORITY = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
DECISION_PRIORITY = {"P0": 0, "P1": 1, "P2": 2, "P3": 3, "P4": 4, "P5": 5}
TRAINING_PRIORITY = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

DEFAULT_POLICY_BY_LABEL = {
    "content.political": "content_political_sensitive_default",
    "content.pornographic": "content_pornographic_default",
    "content.violent": "content_violent_default",
    "content.hate": "content_hate_default",
    "content.harassment": "content_harassment_default",
    "content.self_harm": "content_self_harm_default",
    "content.illegal_instruction": "content_illegal_instruction_default",
    "content.minor_harmful": "content_minor_harmful_default",
    "content.misleading": "content_misleading_default",
    "content.values_violation": "content_values_violation_default",
    "content.jailbreak": "content_jailbreak_default",
}

WHITELIST_CONTEXTS = {
    "teaching_reference",
    "historical_analysis",
    "legal_warning",
    "safety_education",
}

HIGH_RISK_LABELS = {
    "content.pornographic",
    "content.violent",
    "content.hate",
    "content.self_harm",
    "content.illegal_instruction",
    "content.minor_harmful",
    "content.jailbreak",
}


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fallback_doc_id(record: ContentSafetyCheckRecord, index: int) -> str:
    if record.doc_id.strip():
        return record.doc_id.strip()
    sample_id = record.upstream.sample_id.strip()
    if sample_id:
        return sample_id
    return f"doc_{index:04d}"


def _selected_target_labels(
    request: ContentSafetyBatchCheckRequest,
    settings: Settings,
    catalog: dict[str, dict[str, Any]],
) -> list[str]:
    requested = [item.strip() for item in request.target_labels if str(item).strip()]
    if requested:
        return list(dict.fromkeys(requested))

    configured = [item.strip() for item in settings.content_safety_target_labels if str(item).strip()]
    if configured:
        return list(dict.fromkeys(configured))

    if settings.content_safety_operator_ids:
        selected_ops = resolve_selected_sub_operators(settings, label_catalog=catalog)
        labels: list[str] = []
        for operator in selected_ops:
            labels.extend(operator.target_labels)
        if labels:
            return list(dict.fromkeys(labels))

    return list(catalog.keys())


def _resolve_span(text: str, payload: dict[str, Any] | None) -> tuple[int | None, int | None, str]:
    if not isinstance(payload, dict):
        return None, None, ""

    span_text = str(payload.get("text") or "")
    try:
        start = int(payload.get("start", -1))
        end = int(payload.get("end", -1))
    except (TypeError, ValueError):
        start = -1
        end = -1

    if start >= 0 and end > start and end <= len(text):
        actual = text[start:end]
        if not span_text or actual == span_text:
            return start, end, actual

    if span_text:
        first = text.find(span_text)
        last = text.rfind(span_text)
        if first >= 0 and first == last:
            return first, first + len(span_text), span_text

    return None, None, span_text


def _build_base_payload(
    record: ContentSafetyCheckRecord,
    doc_id: str,
    target_labels: list[str],
    request: ContentSafetyBatchCheckRequest,
    settings: Settings,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metadata = record.metadata.model_dump(mode="json")
    metadata.update(metadata.pop("extras", {}))
    training_context = record.training_context.model_dump(mode="json")
    training_context.update(training_context.pop("extras", {}))
    return {
        "doc_id": doc_id,
        "text": record.text[: settings.api_compliance_max_chars],
        "upstream": record.upstream.model_dump(mode="json"),
        "metadata": metadata,
        "training_context": training_context,
        "target_labels": target_labels,
        "label_catalog": [
            {
                "label": label_id,
                "name": spec.get("name", label_id),
                "risk_type": spec.get("risk_type", label_id.split(".")[-1]),
                "policy_tag": spec.get("default_policy_tag", label_id),
                "default_severity": spec.get("default_severity", "medium"),
                "description": spec.get("description", ""),
            }
            for label_id, spec in catalog.items()
            if label_id in target_labels
        ],
        "target_policies": request.target_policies,
        "custom_policy": request.custom_policy,
        "custom_policy_config": request.custom_policy_config,
    }


def _normalize_status(value: Any, has_findings: bool) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"flagged", "unsafe", "reject"}:
        return "flagged"
    if normalized in {"hard_case", "borderline", "controversial", "review"}:
        return "hard_case"
    return "flagged" if has_findings else "clear"


def _normalize_findings(
    record: ContentSafetyCheckRecord,
    doc_id: str,
    payload: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_findings = payload.get("findings")
    if not isinstance(raw_findings, list):
        return []

    findings: list[dict[str, Any]] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        label_id, label_spec = _resolve_label(item, catalog)
        label = label_id or str(item.get("label") or item.get("policy_tag") or "").strip()
        if not label:
            continue
        start, end, span_text = _resolve_span(record.text, item.get("span"))
        severity = _severity(item.get("severity")).value
        confidence = max(0.0, min(_safe_float(item.get("confidence"), 0.7), 1.0))
        findings.append(
            {
                "label": label,
                "risk_type": str(item.get("risk_type") or label_spec.get("risk_type") or label.split(".")[-1]),
                "policy_tag": str(item.get("policy_tag") or label_spec.get("default_policy_tag") or label),
                "severity": severity,
                "confidence": confidence,
                "explanation": str(item.get("explanation") or "Content safety finding."),
                "needs_adjudication": bool(item.get("needs_adjudication", False)) or start is None,
                "hard_case_reason": str(item.get("hard_case_reason") or ("span_unresolved" if start is None else "")),
                "start": start,
                "end": end,
                "text": span_text,
                "source": "api_content_safety_unified_v2",
                "doc_id": doc_id,
            }
        )
    return findings


def _custom_policy_config_enabled(config: dict[str, Any] | None) -> bool:
    return isinstance(config, dict) and bool(config.get("enabled"))


def _custom_policy_action_governance(action: str) -> dict[str, Any]:
    action = str(action or "").upper()
    if action in {"P4", "P5"}:
        return {
            "final_risk_level": "C3",
            "final_decision": action,
            "final_training_eligibility": "T3",
            "final_dataset_route": "exclude_from_training",
            "allow_downstream_annotation": False,
            "requires_manual_review": False,
        }
    if action == "P3":
        return {
            "final_risk_level": "C2",
            "final_decision": "P3",
            "final_training_eligibility": "T2",
            "final_dataset_route": "safety_review_or_eval_only",
            "allow_downstream_annotation": False,
            "requires_manual_review": True,
        }
    if action == "P2":
        return {
            "final_risk_level": "C1",
            "final_decision": "P2",
            "final_training_eligibility": "T1",
            "final_dataset_route": "restricted_training",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
        }
    if action == "P1":
        return {
            "final_risk_level": "C1",
            "final_decision": "P1",
            "final_training_eligibility": "T0",
            "final_dataset_route": "general_training",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
        }
    return {
        "final_risk_level": "C0",
        "final_decision": "P0",
        "final_training_eligibility": "T0",
        "final_dataset_route": "general_training",
        "allow_downstream_annotation": True,
        "requires_manual_review": False,
    }


def _labels_match(left: str, right: str) -> bool:
    left = left.lower().strip()
    right = right.lower().strip()
    return bool(left and right and (left == right or left.startswith(right + ".") or right.startswith(left + ".")))


def _matching_custom_policy_action(
    findings: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[str, str]:
    risk_actions = config.get("risk_actions")
    if not isinstance(risk_actions, dict):
        return "", ""
    best_action = ""
    best_label = ""
    for finding in findings:
        finding_labels = [
            str(finding.get("label") or ""),
            str(finding.get("policy_tag") or ""),
            str(finding.get("risk_type") or ""),
        ]
        for label, action in risk_actions.items():
            action = str(action or "").upper()
            if action not in DECISION_PRIORITY:
                continue
            if any(_labels_match(str(label), item) for item in finding_labels):
                if DECISION_PRIORITY[action] > DECISION_PRIORITY.get(best_action, -1):
                    best_action = action
                    best_label = str(label)
    return best_action, best_label


def _apply_structured_custom_policy(
    policy: dict[str, Any],
    findings: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> None:
    if not _custom_policy_config_enabled(config):
        return
    action, matched_label = _matching_custom_policy_action(findings, config or {})
    if not action:
        return
    current_action = str(policy.get("final_decision") or "P0")
    if DECISION_PRIORITY.get(action, -1) < DECISION_PRIORITY.get(current_action, -1):
        return
    override = _custom_policy_action_governance(action)
    policy.update(override)
    policy.setdefault("policy_hits", []).append(
        {
            "policy_id": "custom_structured_policy",
            "hit": True,
            "confidence": 1.0,
            "reason": f"Structured custom policy matched {matched_label} and required {action}.",
            "evidence": [matched_label] if matched_label else [],
        }
    )
    policy["summary"] = (str(policy.get("summary") or "") + " Structured custom policy applied.").strip()


def _needs_policy_pass(
    request: ContentSafetyBatchCheckRequest,
    base_status: str,
    findings: list[dict[str, Any]],
) -> bool:
    return bool(
        findings
        or request.custom_policy.strip()
        or _custom_policy_config_enabled(request.custom_policy_config)
        or request.target_policies
        or base_status == "hard_case"
    )


def _default_policy_hits(findings: list[dict[str, Any]], context_type: str) -> list[ContentSafetyPolicyHit]:
    hits: list[ContentSafetyPolicyHit] = []
    seen: set[str] = set()
    for finding in findings:
        policy_id = DEFAULT_POLICY_BY_LABEL.get(finding["label"], f"{finding['label'].replace('.', '_')}_default")
        if policy_id in seen:
            continue
        seen.add(policy_id)
        evidence = [finding["text"]] if finding["text"] else []
        hits.append(
            ContentSafetyPolicyHit(
                policy_id=policy_id,
                hit=True,
                confidence=round(float(finding["confidence"]), 4),
                reason=finding["explanation"],
                evidence=evidence,
            )
        )
    if context_type in WHITELIST_CONTEXTS and "education_context_whitelist_default" not in seen:
        hits.append(
            ContentSafetyPolicyHit(
                policy_id="education_context_whitelist_default",
                hit=True,
                confidence=0.75,
                reason="Educational whitelist context applies but still requires training-route judgement.",
                evidence=[],
            )
        )
    return hits


def _heuristic_policy_decision(
    record: ContentSafetyCheckRecord,
    base_status: str,
    context_type: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not findings:
        return {
            "context_type": context_type or "plain_safe",
            "policy_hits": [],
            "uncertainty": False,
            "uncertainty_reason": "",
            "recommended_labels": [],
            "final_risk_level": "C0",
            "final_decision": "P0",
            "final_training_eligibility": "T0",
            "final_dataset_route": "general_training",
            "allow_downstream_annotation": True,
            "requires_manual_review": False,
            "summary": "No content safety risk detected.",
        }

    metadata = record.metadata.model_dump(mode="json")
    audience = str(metadata.get("audience") or "").lower()
    visibility = str(metadata.get("visibility") or "").lower()
    is_minor = audience in {"minor", "mixed"}
    is_public = visibility in {"public", "group"}
    labels = {finding["label"] for finding in findings}
    highest_severity = max((finding["severity"] for finding in findings), key=lambda value: ["low", "medium", "high", "critical"].index(value))
    ambiguous = base_status == "hard_case" or any(finding["needs_adjudication"] for finding in findings)
    dangerous_label = bool(labels.intersection(HIGH_RISK_LABELS))
    already_circulated = bool(metadata.get("already_released")) or str(metadata.get("circulation_status") or "").lower() in {
        "released",
        "published",
        "circulated",
    }

    if dangerous_label and highest_severity in {"high", "critical"} and not context_type in WHITELIST_CONTEXTS:
        decision = "P4"
        risk_level = "C3"
        training = "T3"
        dataset_route = "exclude_from_training"
        review = False
        allow_annotation = False
    elif is_minor and dangerous_label:
        decision = "P4"
        risk_level = "C3"
        training = "T3"
        dataset_route = "exclude_from_training"
        review = False
        allow_annotation = False
    elif ambiguous or context_type in WHITELIST_CONTEXTS or is_public:
        decision = "P3"
        risk_level = "C2"
        training = "T2"
        dataset_route = "safety_review_or_eval_only"
        review = True
        allow_annotation = False
    else:
        decision = "P2"
        risk_level = "C1"
        training = "T1"
        dataset_route = "restricted_training"
        review = False
        allow_annotation = True

    if already_circulated and decision == "P4":
        decision = "P5"

    return {
        "context_type": context_type or "other",
        "policy_hits": [item.model_dump(mode="json") for item in _default_policy_hits(findings, context_type or "other")],
        "uncertainty": review,
        "uncertainty_reason": "Context-sensitive educational sample requires manual review." if review else "",
        "recommended_labels": sorted(labels),
        "final_risk_level": risk_level,
        "final_decision": decision,
        "final_training_eligibility": training,
        "final_dataset_route": dataset_route,
        "allow_downstream_annotation": allow_annotation,
        "requires_manual_review": review,
        "summary": "Heuristic policy fallback applied.",
    }


def _normalize_policy_payload(
    payload: dict[str, Any] | None,
    record: ContentSafetyCheckRecord,
    base_status: str,
    context_type: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback = _heuristic_policy_decision(record, base_status, context_type, findings)
    if not isinstance(payload, dict):
        return fallback

    policy_hits: list[ContentSafetyPolicyHit] = []
    raw_hits = payload.get("policy_hits")
    if isinstance(raw_hits, list):
        for item in raw_hits:
            if not isinstance(item, dict):
                continue
            policy_hits.append(
                ContentSafetyPolicyHit(
                    policy_id=str(item.get("policy_id") or "content_safety_policy"),
                    hit=bool(item.get("hit", True)),
                    confidence=max(0.0, min(_safe_float(item.get("confidence"), 0.7), 1.0)),
                    reason=str(item.get("reason") or ""),
                    evidence=[str(value) for value in item.get("evidence", []) if str(value).strip()],
                )
            )
    if not policy_hits and findings:
        policy_hits = _default_policy_hits(findings, str(payload.get("context_type") or context_type or "other"))

    result = {
        "context_type": str(payload.get("context_type") or fallback["context_type"]),
        "policy_hits": [item.model_dump(mode="json") for item in policy_hits],
        "uncertainty": bool(payload.get("uncertainty", fallback["uncertainty"])),
        "uncertainty_reason": str(payload.get("uncertainty_reason") or fallback["uncertainty_reason"]),
        "recommended_labels": [str(item) for item in payload.get("recommended_labels", fallback["recommended_labels"])],
        "final_risk_level": str(payload.get("final_risk_level") or fallback["final_risk_level"]),
        "final_decision": str(payload.get("final_decision") or fallback["final_decision"]),
        "final_training_eligibility": str(payload.get("final_training_eligibility") or fallback["final_training_eligibility"]),
        "final_dataset_route": str(payload.get("final_dataset_route") or fallback["final_dataset_route"]),
        "allow_downstream_annotation": bool(payload.get("allow_downstream_annotation", fallback["allow_downstream_annotation"])),
        "requires_manual_review": bool(payload.get("requires_manual_review", fallback["requires_manual_review"])),
        "summary": str(payload.get("summary") or fallback["summary"]),
    }

    for key, allowed in (
        ("final_risk_level", RISK_PRIORITY),
        ("final_decision", DECISION_PRIORITY),
        ("final_training_eligibility", TRAINING_PRIORITY),
    ):
        if result[key] not in allowed:
            result[key] = fallback[key]

    return result


def _build_policy_payload(
    record: ContentSafetyCheckRecord,
    doc_id: str,
    request: ContentSafetyBatchCheckRequest,
    target_labels: list[str],
    base_status: str,
    base_summary: str,
    context_type: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = record.metadata.model_dump(mode="json")
    metadata.update(metadata.pop("extras", {}))
    training_context = record.training_context.model_dump(mode="json")
    training_context.update(training_context.pop("extras", {}))
    return {
        "doc_id": doc_id,
        "text": record.text,
        "upstream": record.upstream.model_dump(mode="json"),
        "metadata": metadata,
        "training_context": training_context,
        "target_labels": target_labels,
        "target_policies": request.target_policies,
        "custom_policy": request.custom_policy,
        "custom_policy_config": request.custom_policy_config,
        "base_status": base_status,
        "base_summary": base_summary,
        "context_type": context_type,
        "base_findings": findings,
    }


def _decision_record(
    record: ContentSafetyCheckRecord,
    doc_id: str,
    base_status: str,
    base_summary: str,
    findings: list[dict[str, Any]],
    policy: dict[str, Any],
) -> ContentSafetyDecisionRecord:
    labels = sorted(set(policy.get("recommended_labels") or [finding["label"] for finding in findings]))
    evidence = [
        ContentSafetyEvidenceItem(
            label=finding["label"],
            risk_type=finding["risk_type"],
            policy_tag=finding["policy_tag"],
            severity=finding["severity"],
            confidence=round(float(finding["confidence"]), 4),
            text=finding["text"],
            start=finding["start"],
            end=finding["end"],
            explanation=finding["explanation"],
            source=finding["source"],
        )
        for finding in findings
    ]
    policy_hits = [
        ContentSafetyPolicyHit.model_validate(item)
        for item in policy.get("policy_hits", [])
        if isinstance(item, dict)
    ]
    confidence = 0.0 if not findings else round(
        sum(float(item["confidence"]) for item in findings) / len(findings),
        4,
    )
    metadata = {
        "upstream": record.upstream.model_dump(mode="json"),
        "scene": record.metadata.model_dump(mode="json"),
        "training_context": record.training_context.model_dump(mode="json"),
        "base_status": base_status,
    }
    return ContentSafetyDecisionRecord(
        doc_id=doc_id,
        text_hash=_content_hash(record.text),
        labels=labels,
        policy_hits=policy_hits,
        context_type=str(policy.get("context_type") or ""),
        risk_level=str(policy.get("final_risk_level") or "C0"),
        decision=str(policy.get("final_decision") or "P0"),
        training_eligibility=str(policy.get("final_training_eligibility") or "T0"),
        dataset_route=str(policy.get("final_dataset_route") or "general_training"),
        allow_downstream_annotation=bool(policy.get("allow_downstream_annotation", True)),
        needs_manual_review=bool(policy.get("requires_manual_review", False)),
        confidence=confidence,
        summary=str(policy.get("summary") or base_summary),
        evidence=evidence,
        explanation={
            "base_summary": base_summary,
            "policy_summary": str(policy.get("summary") or ""),
            "uncertainty": bool(policy.get("uncertainty", False)),
            "uncertainty_reason": str(policy.get("uncertainty_reason") or ""),
        },
        metadata=metadata,
    )


def check_content_safety(
    request: ContentSafetyBatchCheckRequest,
    settings: Settings | None = None,
) -> ContentSafetyBatchCheckResponse:
    settings = settings or get_settings()
    catalog = _load_label_catalog(str(settings.content_safety_labels_path))
    target_labels = _selected_target_labels(request, settings, catalog)
    base_prompt = load_prompt(str(settings.api_content_safety_unified_prompt_path))
    policy_prompt = load_prompt(str(settings.api_content_safety_policy_prompt_path))
    client = OpenAICompatibleComplianceClient(settings)

    results: list[ContentSafetyDecisionRecord] = []
    for index, record in enumerate(request.records, start=1):
        doc_id = _fallback_doc_id(record, index)
        base_payload = _build_base_payload(record, doc_id, target_labels, request, settings, catalog)

        base_response: dict[str, Any] | None = None
        base_error = ""
        try:
            base_response = client.complete_json(
                task_name="content_safety_unified_v2",
                system_prompt=base_prompt,
                payload=base_payload,
            )
        except (OpenAICompatibleAPIError, Exception) as exc:
            base_error = str(exc)
            logger.warning("Unified content safety API failed for %s: %s", doc_id, exc)

        findings = _normalize_findings(record, doc_id, base_response or {}, catalog)
        base_status = _normalize_status((base_response or {}).get("status"), bool(findings))
        base_summary = str((base_response or {}).get("summary") or ("Unified content safety API failed." if base_error else "No content safety risk detected."))
        context_type = str((base_response or {}).get("context_type") or ("other" if findings else "plain_safe"))

        policy_response: dict[str, Any] | None = None
        if _needs_policy_pass(request, base_status, findings) and not base_error:
            try:
                policy_response = client.complete_json(
                    task_name="content_safety_policy_v2",
                    system_prompt=policy_prompt,
                    payload=_build_policy_payload(
                        record,
                        doc_id,
                        request,
                        target_labels,
                        base_status,
                        base_summary,
                        context_type,
                        findings,
                    ),
                )
            except (OpenAICompatibleAPIError, Exception) as exc:
                logger.warning("Content safety policy API failed for %s: %s", doc_id, exc)

        policy = _normalize_policy_payload(policy_response, record, base_status, context_type, findings)
        _apply_structured_custom_policy(policy, findings, request.custom_policy_config)
        if base_error:
            if findings:
                policy = _normalize_policy_payload(None, record, base_status, context_type, findings)
                _apply_structured_custom_policy(policy, findings, request.custom_policy_config)
                policy["summary"] = "Fallback policy decision used because unified content safety API was unavailable."
            else:
                policy = {
                    "context_type": "other",
                    "policy_hits": [],
                    "uncertainty": True,
                    "uncertainty_reason": "Unified content safety API was unavailable.",
                    "recommended_labels": [],
                    "final_risk_level": "C2",
                    "final_decision": "P3",
                    "final_training_eligibility": "T2",
                    "final_dataset_route": "safety_review_or_eval_only",
                    "allow_downstream_annotation": False,
                    "requires_manual_review": True,
                    "summary": "Unified content safety API was unavailable; routed to manual review.",
                }

        results.append(_decision_record(record, doc_id, base_status, base_summary, findings, policy))

    overall_decision = max((item.decision for item in results), key=lambda value: DECISION_PRIORITY.get(value, -1), default="P0")
    overall_risk_level = max((item.risk_level for item in results), key=lambda value: RISK_PRIORITY.get(value, -1), default="C0")
    overall_training = max(
        (item.training_eligibility for item in results),
        key=lambda value: TRAINING_PRIORITY.get(value, -1),
        default="T0",
    )
    route_by_training = {
        "T0": "general_training",
        "T1": "restricted_training",
        "T2": "safety_review_or_eval_only",
        "T3": "exclude_from_training",
    }
    counts_by_decision: dict[str, int] = {}
    counts_by_risk: dict[str, int] = {}
    counts_by_training: dict[str, int] = {}
    for item in results:
        counts_by_decision[item.decision] = counts_by_decision.get(item.decision, 0) + 1
        counts_by_risk[item.risk_level] = counts_by_risk.get(item.risk_level, 0) + 1
        counts_by_training[item.training_eligibility] = counts_by_training.get(item.training_eligibility, 0) + 1

    review_suggestions = [
        f"{item.doc_id}: {item.decision} / {item.summary}"
        for item in results
        if item.needs_manual_review or item.decision in {"P4", "P5"}
    ]

    return ContentSafetyBatchCheckResponse(
        request_id=uuid.uuid4().hex,
        overall_decision=overall_decision,
        overall_risk_level=overall_risk_level,
        overall_training_eligibility=overall_training,
        overall_dataset_route=route_by_training.get(overall_training, "general_training"),
        review_suggestions=review_suggestions[:20],
        summary={
            "total_documents": len(results),
            "counts_by_decision": counts_by_decision,
            "counts_by_risk_level": counts_by_risk,
            "counts_by_training_eligibility": counts_by_training,
            "manual_review_count": sum(1 for item in results if item.needs_manual_review),
            "target_labels": target_labels,
            "target_policies": request.target_policies,
            "custom_policy_applied": bool(request.custom_policy.strip()) or _custom_policy_config_enabled(request.custom_policy_config),
        },
        results=results,
        versions={
            "api_model": settings.api_compliance_model,
            "base_prompt": settings.api_content_safety_unified_prompt_path.name,
            "policy_prompt": settings.api_content_safety_policy_prompt_path.name,
            "policy_bundle_version": settings.policy_version,
        },
    )

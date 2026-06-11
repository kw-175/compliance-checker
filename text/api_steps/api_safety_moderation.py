from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient, resolve_provider_config
from text.api_steps import api_content_semantic_adjudication
from text.config.settings import Settings, get_settings
from text.content_safety_registry import (
    ContentSafetySubOperator,
    resolve_selected_sub_operators,
)
from text.engines.content_decision_engine import decide_content_finding
from text.engines.content_policy_engine import load_content_safety_policies, match_policy_hits
from text.engines.content_rule_engine import ContentRuleHit, recall_content_rules, summarize_rule_hits
from text.models.schemas import (
    ContentSafetyResult,
    DetectionFinding,
    DetectionStatus,
    DocumentContextRecord,
    IngestUnit,
    Severity,
    TextSpan,
)
from text.prompt_loader import load_prompt
from text.steps import g_safety_moderation as local_safety_moderation

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    Severity.LOW: 0.30,
    Severity.MEDIUM: 0.55,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 1.00,
}


@lru_cache(maxsize=8)
def _load_label_catalog(path: str) -> dict[str, dict[str, Any]]:
    catalog_path = Path(path)
    if not catalog_path.exists():
        logger.warning("Content safety label catalog not found: %s", catalog_path)
        return {}
    data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    raw_labels = data.get("labels", data)
    if not isinstance(raw_labels, dict):
        logger.warning("Content safety label catalog has invalid shape: %s", catalog_path)
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for label_id, spec in raw_labels.items():
        if isinstance(spec, dict):
            labels[str(label_id)] = spec
    return labels


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dict_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _string_items(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _alias_to_label(catalog: dict[str, dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for label_id, spec in catalog.items():
        candidates = {
            label_id,
            str(spec.get("risk_type") or ""),
            str(spec.get("default_policy_tag") or ""),
            str(spec.get("policy_tag") or ""),
        }
        candidates.update(str(item) for item in spec.get("aliases", []) if item)
        for candidate in candidates:
            if candidate:
                aliases[candidate.lower()] = label_id
    return aliases


def _compact_catalog(catalog: dict[str, dict[str, Any]], label_ids: list[str]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for label_id in label_ids:
        spec = catalog.get(label_id, {})
        compact.append(
            {
                "label": label_id,
                "name": spec.get("name", label_id),
                "risk_type": spec.get("risk_type", label_id.split(".")[-1]),
                "policy_tag": spec.get("default_policy_tag", label_id),
                "default_severity": spec.get("default_severity", "medium"),
                "description": spec.get("description", ""),
            }
        )
    return compact


def _recall_label_terms(catalog: dict[str, dict[str, Any]], label_ids: list[str]) -> list[str]:
    terms: dict[str, None] = {}
    for label_id in label_ids:
        spec = catalog.get(label_id, {})
        for value in (
            label_id,
            spec.get("risk_type"),
            spec.get("default_policy_tag"),
            spec.get("policy_tag"),
        ):
            if value:
                terms[str(value)] = None
        for alias in spec.get("aliases", []):
            if alias:
                terms[str(alias)] = None
    return list(terms)


def _canonicalize_rule_hits(
    rule_hits: list[ContentRuleHit],
    catalog: dict[str, dict[str, Any]],
) -> list[ContentRuleHit]:
    aliases = _alias_to_label(catalog)
    canonical: list[ContentRuleHit] = []
    for hit in rule_hits:
        label_id = aliases.get(hit.policy_tag.lower()) or aliases.get(hit.risk_type.lower())
        if not label_id:
            canonical.append(hit)
            continue
        label_spec = catalog.get(label_id, {})
        canonical.append(
            ContentRuleHit(
                rule_id=hit.rule_id,
                policy_tag=label_id,
                risk_type=str(label_spec.get("risk_type") or hit.risk_type),
                severity=hit.severity,
                score=hit.score,
                evidence=hit.evidence,
                start=hit.start,
                end=hit.end,
                reason=hit.reason,
            )
        )
    return canonical


def _resolve_label(
    payload: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    aliases = _alias_to_label(catalog)
    candidates = [
        str(payload.get("policy_tag") or ""),
        str(payload.get("risk_type") or ""),
        str(payload.get("category") or ""),
        str(payload.get("label") or ""),
    ]
    candidates.extend(_string_items(payload.get("candidate_labels", [])))
    candidates.extend(_string_items(payload.get("labels", [])))
    for candidate in candidates:
        label_id = aliases.get(candidate.strip().lower())
        if label_id:
            return label_id, catalog[label_id]
    return "", {}


def _should_keep_finding(
    payload: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    selected_labels: list[str],
) -> bool:
    label_id, _spec = _resolve_label(payload, catalog)
    if label_id in selected_labels:
        return True
    policy_tag = str(payload.get("policy_tag") or "").lower()
    return any(policy_tag == item.lower() or policy_tag.startswith(item.lower() + ".") for item in selected_labels)


def _governance_hint(severity: Severity, needs_adjudication: bool) -> dict[str, str]:
    if severity in {Severity.CRITICAL, Severity.HIGH}:
        return {
            "risk_level_code": "C3",
            "action": "P4",
            "training_eligibility": "T3",
            "dataset_route": "exclude_from_training",
        }
    if needs_adjudication or severity == Severity.MEDIUM:
        return {
            "risk_level_code": "C2",
            "action": "P3",
            "training_eligibility": "T2",
            "dataset_route": "safety_review_or_eval_only",
        }
    return {
        "risk_level_code": "C1",
        "action": "P2",
        "training_eligibility": "T1",
        "dataset_route": "restricted_training_after_review",
    }


def _label_hierarchy(label: str) -> list[str]:
    parts = [part for part in str(label or "").split(".") if part]
    if len(parts) < 2:
        return [label] if label else []
    hierarchy: list[str] = []
    for index in range(2, len(parts) + 1):
        hierarchy.append(".".join(parts[:index]))
    return hierarchy


def _risk_subcategory(label: str, payload: dict[str, Any]) -> str:
    explicit = str(payload.get("risk_subcategory") or payload.get("subcategory") or payload.get("sub_label") or "").strip()
    if explicit:
        return explicit
    parts = [part for part in str(label or "").split(".") if part]
    return ".".join(parts[2:]) if len(parts) > 2 else ""


def _is_privacy_like_content_finding(finding: DetectionFinding) -> bool:
    haystack = " ".join(
        [
            finding.risk_type,
            finding.policy_tag,
            finding.explanation,
            finding.span.text if finding.span else "",
        ]
    ).lower()
    privacy_markers = {
        "pii",
        "privacy",
        "personal information",
        "personally identifiable",
        "name",
        "phone",
        "email",
        "address",
        "student id",
        "student identifier",
        "parent contact",
        "id card",
    }
    unsafe_markers = {
        "violence",
        "terror",
        "bomb",
        "weapon",
        "kill",
        "hate",
        "sexual",
        "self_harm",
        "suicide",
        "jailbreak",
        "minor sexual",
    }
    return any(marker in haystack for marker in privacy_markers) and not any(
        marker in haystack for marker in unsafe_markers
    )


def _severity(value: Any, fallback: Severity = Severity.MEDIUM) -> Severity:
    try:
        return Severity(str(value).strip().lower())
    except ValueError:
        return fallback


def _severity_from_text(value: str) -> Severity:
    text = str(value or "").strip().lower()
    if text == "critical":
        return Severity.CRITICAL
    if text == "high":
        return Severity.HIGH
    if text == "low":
        return Severity.LOW
    return Severity.MEDIUM


def _status_from_payloads(statuses: list[str], findings: list[DetectionFinding]) -> DetectionStatus:
    normalized = {str(item or "").strip().lower() for item in statuses}
    if normalized.intersection({"flagged", "unsafe", "reject", "blocked"}):
        return DetectionStatus.FLAGGED
    if normalized.intersection({"hard_case", "borderline", "controversial", "review"}):
        return DetectionStatus.HARD_CASE
    return DetectionStatus.FLAGGED if findings else DetectionStatus.CLEAR


def _unique_exact_match(text: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    matches: list[int] = []
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            break
        matches.append(index)
        if len(matches) > 1:
            return None
        start = index + 1
    if not matches:
        return None
    return matches[0], matches[0] + len(needle)


def _span(unit: IngestUnit, payload: dict[str, Any] | None) -> TextSpan | None:
    if not isinstance(payload, dict):
        return None
    payload_text = str(payload.get("text") or "")
    try:
        start = int(payload.get("start", 0))
        end = int(payload.get("end", 0))
    except (TypeError, ValueError):
        start = -1
        end = -1
    if start >= 0 and end > start and end <= len(unit.text):
        expected_text = unit.text[start:end]
        if not payload_text or payload_text == expected_text:
            return TextSpan(start=start, end=end, text=expected_text)

    corrected = _unique_exact_match(unit.text, payload_text)
    if corrected is None:
        return None
    corrected_start, corrected_end = corrected
    return TextSpan(start=corrected_start, end=corrected_end, text=payload_text)


def _finding(
    unit: IngestUnit,
    payload: dict[str, Any],
    settings: Settings,
    catalog: dict[str, dict[str, Any]],
    sub_operator: ContentSafetySubOperator,
) -> DetectionFinding:
    severity = _severity(payload.get("severity"))
    span = _span(unit, payload.get("span"))
    span_missing = span is None

    matched_label, label_spec = _resolve_label(payload, catalog)
    hierarchy = _label_hierarchy(matched_label or str(payload.get("policy_tag") or ""))
    risk_subcategory = _risk_subcategory(matched_label or str(payload.get("policy_tag") or ""), payload)
    risk_type = str(
        payload.get("risk_type")
        or payload.get("category")
        or label_spec.get("risk_type")
        or "general_content_safety"
    )
    needs_adjudication = bool(payload.get("needs_adjudication", False)) or span_missing
    governance = _governance_hint(severity, needs_adjudication)
    policy_tag = str(payload.get("policy_tag") or label_spec.get("default_policy_tag") or f"content.{risk_type}")
    custom_policy = settings.content_safety_custom_policy.strip()
    policy_version = sub_operator.policy_version or settings.policy_version
    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="content_safety",
        risk_type=risk_type,
        policy_tag=policy_tag,
        severity=severity,
        confidence=max(0.0, min(_safe_float(payload.get("confidence"), 0.75), 1.0)),
        explanation=str(payload.get("explanation") or payload.get("rationale") or "API content safety finding."),
        source_tool=f"{settings.api_compliance_source_tool_prefix}.content_safety.{sub_operator.sub_operator_id.lower()}",
        remediation_suggestion=str(payload.get("remediation_suggestion") or "manual_review"),
        needs_adjudication=needs_adjudication,
        hard_case_reason=str(payload.get("hard_case_reason") or ("api_content_span_unresolved" if span_missing else "")),
        span=span,
        attributes={
            "api_payload": payload,
            "span_missing": span_missing,
            "content_safety": {
                "target_labels": sub_operator.target_labels,
                "matched_label": matched_label or policy_tag,
                "label_hierarchy": hierarchy,
                "risk_subcategory": risk_subcategory,
                "label_name": label_spec.get("name", ""),
                "sub_operator_id": sub_operator.sub_operator_id,
                "sub_operator_name": sub_operator.display_name,
                "prompt_profile": sub_operator.prompt_profile,
                "policy_version": policy_version,
                "decision_source": sub_operator.decision_source,
                "custom_policy_applied": bool(custom_policy),
                "custom_policy_excerpt": custom_policy[:500],
                "custom_policy_config_applied": bool(settings.content_safety_custom_policy_config.get("enabled")),
                "custom_policy_config": settings.content_safety_custom_policy_config,
                "api_context_type": str(payload.get("context_type") or ""),
                "api_context_rationale": str(payload.get("context_rationale") or ""),
                "document_context_summary": str(payload.get("document_context_summary") or ""),
                "document_context_explanation": str(payload.get("document_context_explanation") or ""),
                "api_recommended_risk_level": str(payload.get("recommended_risk_level") or ""),
                "api_recommended_action": str(payload.get("recommended_action") or ""),
                "api_recommended_training_eligibility": str(payload.get("recommended_training_eligibility") or ""),
                "api_recommended_dataset_route": str(payload.get("recommended_dataset_route") or ""),
                "api_allow_downstream_annotation": payload.get("allow_downstream_annotation"),
                **governance,
            },
        },
    )


def _fallback_result(
    unit: IngestUnit,
    settings: Settings,
    error: str,
    sub_operator_ids: list[str],
) -> ContentSafetyResult:
    finding = DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="content_safety",
        risk_type="api_unavailable",
        policy_tag="content.api_unavailable",
        severity=Severity.MEDIUM,
        confidence=0.5,
        explanation=f"API content safety detector failed: {error}",
        source_tool=f"{settings.api_compliance_source_tool_prefix}.content_safety",
        remediation_suggestion="manual_review",
        needs_adjudication=True,
        hard_case_reason="api_content_safety_unavailable",
        span=TextSpan(start=0, end=min(len(unit.text), 500), text=unit.text[:500]),
        attributes={
            "content_safety": {
                "selected_sub_operator_ids": sub_operator_ids,
                "policy_version": settings.policy_version,
                "decision_source": "content_safety_orchestrator_fallback",
            }
        },
    )
    return ContentSafetyResult(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        status=DetectionStatus.HARD_CASE,
        risk_score=0.5,
        summary="API content safety detector unavailable; routed to hard-case review.",
        findings=[finding],
        needs_adjudication=True,
        hard_case_reasons=["api_content_safety_unavailable"],
        provider_name="api_content_safety",
        provider_version=settings.api_compliance_model,
        is_degraded=True,
    )


def _guard_and_rule_local_result(
    unit: IngestUnit,
    settings: Settings,
    catalog: dict[str, dict[str, Any]],
    sub_operators: list[ContentSafetySubOperator],
    rule_hits: list[ContentRuleHit],
    document_context: DocumentContextRecord | None,
) -> ContentSafetyResult:
    findings: list[DetectionFinding] = []
    statuses: list[str] = []
    hard_case_reasons: list[str] = []
    scene_context = _scene_context(unit, settings, document_context)

    if not sub_operators:
        return ContentSafetyResult(
            run_id=unit.run_id,
            doc_id=unit.doc_id,
            text_hash=unit.text_hash,
            status=DetectionStatus.CLEAR,
            risk_score=0.0,
            summary="No content-safety sub-operators were selected.",
            findings=[],
            provider_name="local_guard_rule_recall",
            provider_version=settings.qwen3guard_model_name,
        )

    if settings.enable_qwen3guard and settings.qwen3guard_endpoint:
        try:
            import httpx

            response = httpx.post(
                settings.qwen3guard_endpoint,
                json={
                    "doc_id": unit.doc_id,
                    "text": unit.text[: settings.qwen3guard_max_chars],
                    "model": settings.qwen3guard_model_name,
                },
                timeout=settings.qwen3guard_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                label = _normalize_guard_label(payload)
                if label != "safe":
                    statuses.append("hard_case" if label == "controversial" else "flagged")
                    hard_case_reasons.append("qwen3guard_candidate_recall")
                    categories = _qwen3guard_categories(payload)
                    risk_type = categories[0].lower().replace(" ", "_") if categories else "general_content_safety"
                    findings.append(
                        DetectionFinding(
                            doc_id=unit.doc_id,
                            finding_type="content_safety",
                            risk_type=risk_type,
                            policy_tag=f"content.qwen3guard.{label}",
                            severity=Severity.HIGH if label == "unsafe" else Severity.MEDIUM,
                            confidence=float(payload.get("score") or payload.get("confidence") or (0.95 if label == "unsafe" else 0.65)),
                            explanation=f"Qwen3Guard flagged the document as {label}. This is candidate recall, not the final contextual decision.",
                            source_tool=f"qwen3guard.{settings.qwen3guard_model_name}",
                            remediation_suggestion="manual_review",
                            needs_adjudication=True,
                            hard_case_reason="qwen3guard_candidate_recall",
                            span=TextSpan(
                                start=0,
                                end=min(len(unit.text), 240),
                                text=unit.text[:240],
                                context_before="",
                                context_after=unit.text[240:280],
                            ),
                            attributes={
                                "api_payload": payload,
                                "content_safety": {
                                    "api_context_type": scene_context.get("scene_type", ""),
                                    "api_context_rationale": "Guard output is treated as candidate recall only.",
                                    "matched_label": f"content.qwen3guard.{label}",
                                    "requires_manual_review": True,
                                },
                            },
                        )
                    )
        except Exception as exc:
            logger.warning("Local Qwen3Guard recall failed for %s: %s", unit.doc_id, exc)
            hard_case_reasons.append(f"qwen3guard_failed:{exc}")

    target_labels = _unique_labels(sub_operators)
    aliases = _alias_to_label(catalog)
    for hit in rule_hits:
        label_id = aliases.get(hit.policy_tag.lower()) or aliases.get(hit.risk_type.lower()) or hit.policy_tag
        if target_labels and label_id not in target_labels and hit.policy_tag not in target_labels and hit.risk_type not in target_labels:
            continue
        sub_operator = _sub_operator_for_payload(
            {
                "policy_tag": hit.policy_tag,
                "risk_type": hit.risk_type,
                "label": label_id,
                "text": hit.evidence,
                "start": hit.start,
                "end": hit.end,
                "confidence": hit.score,
                "severity": hit.severity,
                "explanation": hit.reason,
            },
            catalog,
            sub_operators,
        ) or sub_operators[0]
        finding = _finding(
            unit,
            {
                "policy_tag": hit.policy_tag,
                "risk_type": hit.risk_type,
                "label": label_id,
                "confidence": hit.score,
                "severity": hit.severity,
                "explanation": hit.reason,
                "needs_adjudication": True,
                "hard_case_reason": "rule_candidate_recall",
                "context_type": scene_context.get("scene_type", ""),
                "context_rationale": "Rule match promoted to candidate recall for contextual adjudication.",
                "span": {
                    "start": hit.start,
                    "end": hit.end,
                    "text": hit.evidence,
                },
            },
            settings,
            catalog,
            sub_operator,
        )
        findings.append(finding)
        statuses.append("hard_case")
        hard_case_reasons.append("rule_candidate_recall")

    findings = _merge_local_findings(findings, None)
    status = _status_from_payloads(statuses, findings)
    return ContentSafetyResult(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        status=status,
        risk_score=max((SEVERITY_WEIGHTS[f.severity] * f.confidence for f in findings), default=0.0),
        summary="Local guard and rule candidate recall completed.",
        findings=findings,
        needs_adjudication=bool(findings),
        hard_case_reasons=list(dict.fromkeys(hard_case_reasons)),
        provider_name="local_guard_rule_recall",
        provider_version=settings.qwen3guard_model_name,
        is_degraded=False,
    )


def _operator_payload(
    unit: IngestUnit,
    settings: Settings,
    sub_operator: ContentSafetySubOperator,
    catalog: dict[str, dict[str, Any]],
    document_context: DocumentContextRecord | None,
) -> dict[str, Any]:
    scene_metadata = _scene_context(unit, settings, document_context)
    provider = resolve_provider_config(settings)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": unit.text[: provider.max_chars],
        "metadata": scene_metadata,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "target_labels": sub_operator.target_labels,
        "label_catalog": _compact_catalog(catalog, sub_operator.target_labels),
        "custom_policy": settings.content_safety_custom_policy,
        "custom_policy_config": settings.content_safety_custom_policy_config,
        "training_context": settings.content_safety_training_context,
        "sub_operator": {
            "sub_operator_id": sub_operator.sub_operator_id,
            "display_name": sub_operator.display_name,
            "description": sub_operator.description,
            "prompt_profile": sub_operator.prompt_profile,
            "policy_version": sub_operator.policy_version or settings.policy_version,
            "decision_source": sub_operator.decision_source,
        },
    }


def _unique_labels(sub_operators: list[ContentSafetySubOperator]) -> list[str]:
    labels: dict[str, None] = {}
    for sub_operator in sub_operators:
        for label in sub_operator.target_labels:
            if label:
                labels[label] = None
    return list(labels)


def _combined_payload(
    unit: IngestUnit,
    settings: Settings,
    sub_operators: list[ContentSafetySubOperator],
    catalog: dict[str, dict[str, Any]],
    rule_hits: list[ContentRuleHit],
    document_context: DocumentContextRecord | None,
) -> dict[str, Any]:
    scene_metadata = _scene_context(unit, settings, document_context)
    provider = resolve_provider_config(settings)
    target_labels = _unique_labels(sub_operators)
    serialized_rule_hits = summarize_rule_hits(rule_hits)
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": unit.text[: provider.max_chars],
        "metadata": scene_metadata,
        "document_context": document_context.model_dump(mode="json") if document_context else {},
        "target_labels": target_labels,
        "label_catalog": _compact_catalog(catalog, target_labels),
        "custom_policy": settings.content_safety_custom_policy,
        "custom_policy_config": settings.content_safety_custom_policy_config,
        "training_context": settings.content_safety_training_context,
        "candidate_rule_hits": serialized_rule_hits,
        "candidate_labels": sorted({item["policy_tag"] for item in serialized_rule_hits if item.get("policy_tag")}),
        "execution_mode": "single_pass_candidate_recall",
        "sub_operator": {
            "sub_operator_id": "CSA_COMBINED",
            "display_name": "Combined content-safety candidate recall",
            "description": "Recall candidate unsafe spans once, then route findings to selected CSA sub-operators.",
            "prompt_profile": "candidate_recall",
            "policy_version": settings.policy_version,
            "decision_source": "single_pass_content_safety_recall",
        },
        "sub_operators": [
            {
                "sub_operator_id": item.sub_operator_id,
                "display_name": item.display_name,
                "description": item.description,
                "target_labels": item.target_labels,
                "prompt_profile": item.prompt_profile,
                "policy_version": item.policy_version or settings.policy_version,
                "decision_source": item.decision_source,
            }
            for item in sub_operators
        ],
    }


def _sub_operator_for_payload(
    payload: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    sub_operators: list[ContentSafetySubOperator],
) -> ContentSafetySubOperator | None:
    matched_label, label_spec = _resolve_label(payload, catalog)
    aliases = {
        str(payload.get("policy_tag") or "").strip().lower(),
        str(payload.get("risk_type") or "").strip().lower(),
        str(payload.get("category") or "").strip().lower(),
        str(payload.get("label") or "").strip().lower(),
        matched_label.lower(),
        str(label_spec.get("risk_type") or "").strip().lower(),
        str(label_spec.get("default_policy_tag") or "").strip().lower(),
    }
    aliases.update(item.strip().lower() for item in _string_items(payload.get("candidate_labels", [])))
    aliases.update(item.strip().lower() for item in _string_items(payload.get("labels", [])))
    aliases.discard("")

    for sub_operator in sub_operators:
        operator_aliases = {item.strip().lower() for item in sub_operator.target_labels}
        operator_aliases.update(item.strip().lower() for item in sub_operator.aliases)
        if aliases.intersection(operator_aliases):
            return sub_operator
        for alias in aliases:
            if any(alias.startswith(operator_alias + ".") for operator_alias in operator_aliases if operator_alias):
                return sub_operator
        for operator_alias in operator_aliases:
            if any(operator_alias.startswith(alias + ".") for alias in aliases if alias):
                return sub_operator

    if len(sub_operators) == 1 and not matched_label:
        return sub_operators[0]
    return None


def _scene_context(
    unit: IngestUnit,
    settings: Settings,
    document_context: DocumentContextRecord | None,
) -> dict[str, Any]:
    scene_context = {**unit.metadata, **settings.content_safety_metadata}
    if document_context is None:
        return scene_context
    scene_context.update(
        {
            "document_type": document_context.document_type,
            "scene_type": document_context.scene_type,
            "subject_type": document_context.subject_type,
            "usage_target": document_context.usage_target,
            "contains_education_context": document_context.contains_education_context,
            "contains_minor_context": document_context.contains_minor_context,
            "document_context_summary": document_context.summary,
            "document_context_explanation": document_context.explanation,
        }
    )
    return scene_context


def _attach_decision_context(
    finding: DetectionFinding,
    rule_hits: list[ContentRuleHit],
    policies: list[dict[str, Any]],
    context: dict[str, Any],
    settings: Settings,
) -> DetectionFinding:
    content_attrs = finding.attributes.get("content_safety", {})
    decision_context = {
        **context,
        "context_type": content_attrs.get("api_context_type") or context.get("context_type") or "",
        "training_context": settings.content_safety_training_context,
    }
    policy_hits = match_policy_hits(finding, policies, decision_context)
    decision = decide_content_finding(
        finding=finding,
        rule_hits=rule_hits,
        policy_hits=policy_hits,
        context=decision_context,
        training_context=settings.content_safety_training_context,
        custom_policy=settings.content_safety_custom_policy,
        custom_policy_config=settings.content_safety_custom_policy_config,
    )
    content_attrs = finding.attributes.setdefault("content_safety", {})
    content_attrs.update(decision)
    finding.needs_adjudication = bool(decision.get("requires_manual_review", finding.needs_adjudication))
    if finding.needs_adjudication and not finding.hard_case_reason:
        finding.hard_case_reason = "content_safety_policy_review_required"
    return finding


def _attach_semantic_adjudications(
    unit: IngestUnit,
    findings: list[DetectionFinding],
    settings: Settings,
    context: dict[str, Any],
) -> None:
    adjudications = api_content_semantic_adjudication.run(unit, findings, settings, context)
    for finding in findings:
        adjudication = adjudications.get(finding.finding_id)
        if not adjudication:
            continue
        content_attrs = finding.attributes.setdefault("content_safety", {})
        content_attrs["semantic_adjudication"] = adjudication
        content_attrs["semantic_context_type"] = adjudication.get("context_type", "")
        content_attrs["semantic_decision"] = adjudication.get("semantic_decision", "")
        content_attrs["semantic_risk_level"] = adjudication.get("final_risk_level", "")
        content_attrs["semantic_action"] = adjudication.get("final_action", "")
        content_attrs["semantic_training_eligibility"] = adjudication.get("final_training_eligibility", "")
        content_attrs["semantic_dataset_route"] = adjudication.get("final_dataset_route", "")
        content_attrs["semantic_allow_downstream_annotation"] = adjudication.get("allow_downstream_annotation")
        content_attrs["semantic_requires_manual_review"] = adjudication.get("requires_manual_review")
        content_attrs["semantic_downgrade_allowed"] = adjudication.get("downgrade_allowed")
        content_attrs["semantic_upgrade_required"] = adjudication.get("upgrade_required")
        content_attrs["semantic_reasoning_summary"] = adjudication.get("reasoning_summary", "")


def _merge_local_findings(
    findings: list[DetectionFinding],
    local_result: ContentSafetyResult | None,
) -> list[DetectionFinding]:
    merged: dict[tuple[str, int, int, str], DetectionFinding] = {}
    for finding in findings + list(local_result.findings if local_result else []):
        span = finding.span
        key = (
            finding.risk_type,
            getattr(span, "start", -1) if span else -1,
            getattr(span, "end", -1) if span else -1,
            finding.policy_tag,
        )
        existing = merged.get(key)
        if existing is None or finding.confidence > existing.confidence:
            merged[key] = finding
    return list(merged.values())


def run(
    ingest_units: list[IngestUnit],
    settings: Settings | None = None,
    document_contexts: list[DocumentContextRecord] | None = None,
) -> list[ContentSafetyResult]:
    settings = settings or get_settings()
    client = OpenAICompatibleComplianceClient(settings)
    provider = resolve_provider_config(settings)
    system_prompt = load_prompt(str(settings.api_content_safety_prompt_path))
    label_catalog = _load_label_catalog(str(settings.content_safety_labels_path))
    policies = load_content_safety_policies(settings.content_safety_policies_path)
    sub_operators = resolve_selected_sub_operators(settings, label_catalog=label_catalog)
    selected_sub_operator_ids = [item.sub_operator_id for item in sub_operators]
    context_by_doc = {item.doc_id: item for item in (document_contexts or [])}
    local_results_by_doc: dict[str, ContentSafetyResult] = {}
    if provider.mode == "local_model":
        local_results_by_doc = {
            item.doc_id: item for item in local_safety_moderation.run(ingest_units, settings)
        }
    results: list[ContentSafetyResult] = []

    for unit in ingest_units:
        document_context = context_by_doc.get(unit.doc_id)
        local_result = local_results_by_doc.get(unit.doc_id)
        if not sub_operators:
            results.append(
                ContentSafetyResult(
                    run_id=unit.run_id,
                    doc_id=unit.doc_id,
                    text_hash=unit.text_hash,
                    status=DetectionStatus.CLEAR,
                    risk_score=0.0,
                    summary="No content-safety sub-operators were selected.",
                    findings=[],
                    provider_name="api_content_safety",
                    provider_version=settings.api_compliance_model,
                )
            )
            continue

        all_findings: list[DetectionFinding] = []
        statuses: list[str] = []
        summaries: list[str] = []
        hard_case_reasons: list[str] = []
        privacy_like_findings: list[DetectionFinding] = []
        degraded = False
        target_labels = _unique_labels(sub_operators)
        recall_terms = _recall_label_terms(label_catalog, target_labels)
        rule_hits = _canonicalize_rule_hits(
            recall_content_rules(unit.text, settings.content_rules_path, recall_terms),
            label_catalog,
        )
        scene_context = _scene_context(unit, settings, document_context)

        if provider.mode == "local_model":
            results.append(
                _guard_and_rule_local_result(
                    unit,
                    settings,
                    label_catalog,
                    sub_operators,
                    rule_hits,
                    document_context,
                )
            )
            continue

        try:
            payload = client.complete_json(
                task_name="content_safety",
                system_prompt=system_prompt,
                payload=_combined_payload(unit, settings, sub_operators, label_catalog, rule_hits, document_context),
            )
            statuses.append(str(payload.get("status") or "clear"))
            summary = str(payload.get("summary") or "").strip()
            if summary:
                summaries.append(summary)
            hard_case_reasons.extend(_string_items(payload.get("hard_case_reasons", [])))
            for item in _dict_items(payload.get("findings", [])):
                sub_operator = _sub_operator_for_payload(item, label_catalog, sub_operators)
                if sub_operator is None:
                    probe_finding = _finding(unit, item, settings, label_catalog, sub_operators[0])
                    if _is_privacy_like_content_finding(probe_finding):
                        privacy_like_findings.append(probe_finding)
                        continue
                    hard_case_reasons.append("content_safety_unmapped_candidate_suppressed")
                    continue
                finding = _finding(unit, item, settings, label_catalog, sub_operator)
                if _is_privacy_like_content_finding(finding):
                    privacy_like_findings.append(finding)
                    continue
                if _should_keep_finding(item, label_catalog, sub_operator.target_labels):
                    all_findings.append(finding)
        except (OpenAICompatibleAPIError, Exception) as exc:
            logger.warning("API content safety failed for %s: %s", unit.doc_id, exc)
            results.append(_fallback_result(unit, settings, str(exc), selected_sub_operator_ids))
            continue

        _attach_semantic_adjudications(unit, all_findings, settings, scene_context)
        all_findings = [
            _attach_decision_context(finding, rule_hits, policies, scene_context, settings)
            for finding in all_findings
        ]
        all_findings = _merge_local_findings(all_findings, local_result)

        if privacy_like_findings:
            hard_case_reasons.append("privacy_like_content_findings_suppressed")
        if local_result is not None:
            hard_case_reasons.extend(local_result.hard_case_reasons)
        deduped_reasons = list(dict.fromkeys(reason for reason in hard_case_reasons if reason))
        status = _status_from_payloads(statuses, all_findings)
        if local_result is not None and local_result.status in {DetectionStatus.FLAGGED, DetectionStatus.HARD_CASE}:
            status = local_result.status if status == DetectionStatus.CLEAR else status
        if not all_findings and privacy_like_findings:
            status = DetectionStatus.CLEAR
        needs_adjudication = status == DetectionStatus.HARD_CASE or any(
            finding.needs_adjudication for finding in all_findings
        )
        if local_result is not None:
            needs_adjudication = needs_adjudication or local_result.needs_adjudication
        if not all_findings and privacy_like_findings:
            needs_adjudication = False

        risk_score = max(
            (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in all_findings),
            default=0.0,
        )
        summary = (
            "No content safety issues detected; privacy-like API findings were suppressed."
            if privacy_like_findings and not all_findings
            else " | ".join(summaries) if summaries else "API content safety detection completed."
        )

        results.append(
            ContentSafetyResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                status=status,
                risk_score=round(max(0.0, min(risk_score, 1.0)), 4),
                summary=summary,
                findings=all_findings,
                needs_adjudication=needs_adjudication,
                hard_case_reasons=deduped_reasons,
                provider_name="local_content_safety" if provider.mode == "local_model" else "api_content_safety",
                provider_version=provider.model,
                is_degraded=degraded or bool(local_result and local_result.is_degraded),
            )
        )

    logger.info(
        "API content safety detection completed: %d documents with sub-operators %s",
        len(results),
        selected_sub_operator_ids,
    )
    return results

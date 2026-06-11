from __future__ import annotations

import logging
import re
from typing import Any

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DetectionFinding,
    DocumentContextRecord,
    IngestUnit,
    PrivacyDetectionResult,
    Severity,
    TextSpan,
)
from text.prompt_loader import load_prompt
from text.steps import f_privacy_detection as local_privacy_detection

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    Severity.LOW: 0.30,
    Severity.MEDIUM: 0.55,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 1.00,
}

DEFAULT_REDACTIONS = {
    "person_name": "<PERSON>",
    "phone": "<PHONE>",
    "phone_number": "<PHONE>",
    "email": "<EMAIL>",
    "id_card": "<ID_CARD>",
    "bank_card": "<BANK_CARD>",
    "bank_account": "<BANK_ACCOUNT>",
    "address": "<ADDRESS>",
    "student_id": "<STUDENT_ID>",
    "parent_contact": "<PARENT_CONTACT>",
    "education_record": "<EDU_RECORD>",
    "organization": "<ORGANIZATION>",
    "location": "<LOCATION>",
    "social_account": "<SOCIAL_ACCOUNT>",
    "payment_account": "<PAYMENT_ACCOUNT>",
    "vehicle_identifier": "<VEHICLE_ID>",
    "combined_identity": "<COMBINED_IDENTITY>",
    "medical_record": "<MEDICAL_RECORD>",
    "psychological_record": "<MEDICAL_RECORD>",
    "secret": "<SECRET>",
    "api_key": "<SECRET>",
    "token": "<SECRET>",
    "password": "<SECRET>",
    "minor_info": "<MINOR_INFO>",
}

COMBINATION_BASE_TYPES = {
    "person_name",
    "phone_number",
    "phone",
    "id_card",
    "address",
    "student_id",
    "parent_contact",
    "bank_card",
    "bank_account",
    "social_account",
    "payment_account",
}
COMBINATION_RISK_ALIASES = {
    "phone": "phone_number",
    "bank_account": "bank_card",
    "social_account": "parent_contact",
    "payment_account": "bank_card",
}

OPERATOR_ENTITY_TYPES = {
    "PII_001": {"person_name"},
    "PII_002": {"phone", "phone_number", "email", "social_account"},
    "PII_003": {"id_card", "id_number", "passport"},
    "PII_004": {"address", "location"},
    "PII_005": {"student_id", "education_record", "score_record", "disciplinary_record"},
    "PII_006": {"parent_contact", "guardian_contact", "family_contact"},
    "PII_007": {"bank_card", "bank_account", "payment_account"},
    "PII_008": {"medical_record", "psychological_record", "health_record"},
    "PII_009": {"secret", "api_key", "token", "password", "credential"},
    "PII_010": {"combined_identity"},
    "PII_011": {"minor_info", "student_id", "education_record", "parent_contact"},
}
PUBLIC_ORGANIZATION_SUFFIXES = (
    "大学",
    "学院",
    "学校",
    "中学",
    "小学",
    "幼儿园",
    "教育局",
    "研究中心",
    "研究院",
    "出版社",
    "平台",
    "联盟",
    "基地",
    "公司",
    "集团",
    "医院",
)
PUBLIC_ORGANIZATION_EXACT = {
    "清华大学",
    "北京大学",
    "北大",
    "复旦大学",
    "中国科技大学",
    "麻省理工",
    "海南医科大学",
    "学堂在线",
    "教育部在线教育研究中心",
    "世界慕课联盟",
}


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


def _looks_like_pii_summary(payload: dict[str, Any]) -> bool:
    summary = str(payload.get("summary") or "").lower()
    pii_count = _safe_float(payload.get("pii_count"), 0.0)
    risk_score = _safe_float(payload.get("risk_score"), 0.0)
    normalized_summary = re.sub(r"\s+", " ", summary).strip()
    negative_markers = {
        "no pii",
        "no direct pii",
        "no personal",
        "no direct personal",
        "no personal identifier",
        "no personal identifiers",
        "no personally identifiable",
        "no privacy",
        "not detected",
        "none detected",
        "without pii",
        "no sensitive",
    }
    if pii_count <= 0 and risk_score <= 0 and any(marker in summary for marker in negative_markers):
        return False
    if pii_count <= 0 and risk_score <= 0:
        negative_patterns = (
            r"\bno\b.{0,120}\b(pii|personal identifiers?|personally identifiable|privacy risk|sensitive data)\b.{0,120}\b(detected|found|present)?\b",
            r"\b(no|not|none)\b.{0,80}\b(detected|found|present)\b",
            r"\bwithout\b.{0,80}\b(pii|personal identifiers?|personally identifiable|sensitive data)\b",
        )
        if any(re.search(pattern, normalized_summary) for pattern in negative_patterns):
            return False

    if pii_count > 0 or risk_score > 0:
        return True

    pii_markers = {
        "pii",
        "personal",
        "personally identifiable",
        "name",
        "phone",
        "email",
        "address",
        "student id",
        "student identifier",
        "id card",
        "identity",
        "parent contact",
    }
    positive_pattern_templates = (
        r"\b(detected|found|contains|includes|identified|present)\b.{0,80}\b(__PII_MARKERS__)\b",
        r"\b(__PII_MARKERS__)\b.{0,80}\b(detected|found|present|identified)\b",
    )
    marker_pattern = "|".join(re.escape(marker) for marker in sorted(pii_markers, key=len, reverse=True))
    return any(
        re.search(template.replace("__PII_MARKERS__", marker_pattern), normalized_summary)
        for template in positive_pattern_templates
    )


def _severity(value: Any, fallback: Severity = Severity.MEDIUM) -> Severity:
    try:
        return Severity(str(value).strip().lower())
    except ValueError:
        return fallback


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

    # Some OpenAI-compatible models return the right span text but calculate
    # offsets from the surrounding JSON/prompt. Correct only unambiguous cases.
    corrected = _unique_exact_match(unit.text, payload_text)
    if corrected is None:
        return None
    corrected_start, corrected_end = corrected
    return TextSpan(start=corrected_start, end=corrected_end, text=payload_text)


def _finding(
    unit: IngestUnit,
    payload: dict[str, Any],
    settings: Settings,
    document_context: DocumentContextRecord | None,
    invalid_span_payloads: list[dict[str, Any]],
) -> DetectionFinding | None:
    span = _span(unit, payload.get("span"))
    if span is None:
        invalid_span_payloads.append(payload)
        return None
    risk_type = str(payload.get("risk_type") or payload.get("entity_type") or "pii_entity").strip() or "pii_entity"
    replacement = str(payload.get("redaction_suggestion") or payload.get("replacement") or DEFAULT_REDACTIONS.get(risk_type, "<PII>"))
    severity = _severity(payload.get("severity"))

    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="privacy",
        risk_type=risk_type,
        policy_tag=str(payload.get("policy_tag") or f"pii.{risk_type}"),
        severity=severity,
        confidence=max(0.0, min(_safe_float(payload.get("confidence"), 0.75), 1.0)),
        explanation=str(payload.get("explanation") or "API privacy finding."),
        source_tool=f"{settings.api_compliance_source_tool_prefix}.privacy",
        remediation_suggestion=str(payload.get("remediation_suggestion") or "redact"),
        redaction_suggestion=replacement,
        needs_adjudication=bool(payload.get("needs_adjudication", False)),
        hard_case_reason=str(payload.get("hard_case_reason") or ""),
        span=span,
        attributes={
            "api_payload": payload,
            "privacy_context": {
                "document_type": document_context.document_type if document_context else "",
                "scene_type": document_context.scene_type if document_context else "",
                "subject_type": document_context.subject_type if document_context else "",
                "context_summary": document_context.summary if document_context else "",
                "context_explanation": str(
                    payload.get("context_explanation")
                    or payload.get("governance_explanation")
                    or document_context.explanation
                    if document_context
                    else ""
                ),
                "governance_action": str(payload.get("governance_action") or ""),
                "training_impact": str(payload.get("training_impact") or ""),
                "annotation_impact": str(payload.get("annotation_impact") or ""),
                "can_keep": payload.get("can_keep"),
                "is_real_pii": payload.get("is_real_pii"),
            },
        },
    )


def _invalid_span_finding(
    unit: IngestUnit,
    payloads: list[dict[str, Any]],
    settings: Settings,
) -> DetectionFinding:
    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="privacy",
        risk_type="api_privacy_invalid_span",
        policy_tag="pii.api_invalid_span",
        severity=Severity.LOW,
        confidence=0.5,
        explanation=(
            "API privacy response contained spans that could not be aligned to the raw "
            "input text after automatic correction, so those spans were ignored for deterministic redaction."
        ),
        source_tool=f"{settings.api_compliance_source_tool_prefix}.privacy",
        remediation_suggestion="manual_review",
        redaction_suggestion="",
        needs_adjudication=True,
        hard_case_reason="api_privacy_invalid_span",
        span=None,
        attributes={
            "invalid_span_count": len(payloads),
            "invalid_payload_samples": payloads[:5],
        },
    )


def _add_combined_identity_if_needed(
    unit: IngestUnit,
    findings: list[DetectionFinding],
    settings: Settings,
) -> None:
    matched_base_types = {
        COMBINATION_RISK_ALIASES.get(finding.risk_type, finding.risk_type)
        for finding in findings
        if finding.risk_type in COMBINATION_BASE_TYPES
        or COMBINATION_RISK_ALIASES.get(finding.risk_type) in COMBINATION_BASE_TYPES
    }
    if len(matched_base_types) < settings.privacy_combination_threshold:
        return
    if any(finding.risk_type == "combined_identity" for finding in findings):
        return

    findings.append(
        DetectionFinding(
            doc_id=unit.doc_id,
            finding_type="privacy",
            risk_type="combined_identity",
            policy_tag="pii.combined_identity",
            severity=Severity.CRITICAL,
            confidence=0.92,
            explanation=(
                f"Detected {len(matched_base_types)} distinct identity attributes "
                "that can combine into a stronger personal profile."
            ),
            source_tool=f"{settings.api_compliance_source_tool_prefix}.privacy_combined_identity",
            remediation_suggestion="restrict_and_review",
            redaction_suggestion="<COMBINED_IDENTITY>",
            needs_adjudication=True,
            hard_case_reason="combined_identity",
            span=None,
            attributes={"matched_types": sorted(matched_base_types)},
        )
    )


def _selected_entity_types(settings: Settings) -> set[str]:
    selected = {str(item).strip() for item in settings.privacy_target_types if str(item).strip()}
    for operator_id in settings.privacy_operator_ids:
        selected.update(OPERATOR_ENTITY_TYPES.get(str(operator_id).strip().upper(), set()))
    return selected


def _filter_selected_findings(findings: list[DetectionFinding], selected_types: set[str]) -> list[DetectionFinding]:
    if not selected_types:
        return findings
    aliases = {
        "phone": "phone_number",
        "id_number": "id_card",
        "bank_account": "bank_card",
        "payment_account": "bank_card",
        "guardian_contact": "parent_contact",
        "score_record": "education_record",
        "disciplinary_record": "education_record",
        "psychological_record": "medical_record",
        "health_record": "medical_record",
        "api_key": "secret",
        "token": "secret",
        "password": "secret",
        "credential": "secret",
    }
    normalized_selected = selected_types | {aliases.get(item, item) for item in selected_types}
    return [
        finding
        for finding in findings
        if finding.risk_type in normalized_selected or aliases.get(finding.risk_type, finding.risk_type) in normalized_selected
    ]


def _looks_public_organization(finding: DetectionFinding) -> bool:
    if finding.risk_type != "organization" or finding.span is None:
        return False
    text = re.sub(r"\s+", "", str(finding.span.text or "").strip())
    if not text:
        return False
    if text in PUBLIC_ORGANIZATION_EXACT:
        return True
    if any(name in text for name in PUBLIC_ORGANIZATION_EXACT):
        return True
    return any(text.endswith(suffix) for suffix in PUBLIC_ORGANIZATION_SUFFIXES)


def _suppress_public_organization_findings(findings: list[DetectionFinding]) -> list[DetectionFinding]:
    return [finding for finding in findings if not _looks_public_organization(finding)]


def _fallback_result(unit: IngestUnit, settings: Settings, error: str) -> PrivacyDetectionResult:
    finding = DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="privacy",
        risk_type="api_unavailable",
        policy_tag="pii.api_unavailable",
        severity=Severity.MEDIUM,
        confidence=0.5,
        explanation=f"API privacy detector failed: {error}",
        source_tool=f"{settings.api_compliance_source_tool_prefix}.privacy",
        remediation_suggestion="manual_review",
        needs_adjudication=True,
        hard_case_reason="api_privacy_unavailable",
        span=None,
    )
    return PrivacyDetectionResult(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        pii_count=1,
        risk_score=0.5,
        summary="API privacy detector unavailable; routed to hard-case review.",
        findings=[finding],
        needs_adjudication=True,
        hard_case_reasons=["api_privacy_unavailable"],
        provider_name="api_privacy_detector",
        provider_version=settings.api_compliance_model,
        is_degraded=True,
    )


def _privacy_request_payload(unit: IngestUnit, settings: Settings) -> dict[str, Any]:
    # Keep the privacy detector focused on the raw text itself. System identifiers
    # such as run_id/doc_id and transport metadata can bias the model and produce
    # spans that point into surrounding JSON instead of the actual text body.
    return {
        "language": unit.language,
        "text": unit.text[: settings.api_compliance_max_chars],
    }


def _request_payload(
    unit: IngestUnit,
    settings: Settings,
    document_context: DocumentContextRecord | None,
) -> dict[str, Any]:
    provider = resolve_provider_config(settings)
    payload = {
        "language": unit.language,
        "text": unit.text[: provider.max_chars],
    }
    if provider.mode == "local_model":
        payload["metadata"] = unit.metadata
        payload["document_context"] = document_context.model_dump(mode="json") if document_context else {}
        payload["target_entity_types"] = sorted(_selected_entity_types(settings))
        payload["selected_privacy_operators"] = list(settings.privacy_operator_ids)
        payload["governance_goal"] = (
            "First recall all candidate privacy spans, then judge whether each span is real PII in context, "
            "whether it can be kept, or whether it must be redacted, generalized, reviewed, or excluded."
        )
    else:
        payload["target_entity_types"] = sorted(_selected_entity_types(settings))
        payload["selected_privacy_operators"] = list(settings.privacy_operator_ids)
    return payload


def _apply_contextual_privacy_defaults(
    finding: DetectionFinding,
    document_context: DocumentContextRecord | None,
) -> DetectionFinding:
    privacy_context = finding.attributes.setdefault("privacy_context", {})
    if document_context is None:
        privacy_context.setdefault("context_explanation", finding.explanation or "")
        return finding

    privacy_context.setdefault("document_type", document_context.document_type)
    privacy_context.setdefault("scene_type", document_context.scene_type)
    privacy_context.setdefault("subject_type", document_context.subject_type)
    privacy_context.setdefault("context_summary", document_context.summary)

    existing_explanation = privacy_context.get("context_explanation") or ""
    if existing_explanation:
        finding.explanation = str(existing_explanation)
        return finding

    if document_context.document_type == "textbook_example" and finding.risk_type == "person_name":
        finding.explanation = (
            "This name appears in a textbook or example-style context rather than an obvious real student "
            "record, so it should be reviewed as a contextual privacy candidate instead of being released directly."
        )
        privacy_context["context_explanation"] = finding.explanation
        finding.needs_adjudication = True
        if not finding.hard_case_reason:
            finding.hard_case_reason = "contextual_example_name"
        privacy_context["governance_action"] = privacy_context.get("governance_action") or "manual_review"
        return finding

    if document_context.document_type in {"student_record", "grade_record", "home_school_communication"}:
        finding.explanation = (
            f"This {finding.risk_type} appears inside an education-record context tied to a likely real student or "
            "family record, so it should not be preserved in raw form and must enter privacy governance."
        )
        privacy_context["context_explanation"] = finding.explanation
        privacy_context["governance_action"] = privacy_context.get("governance_action") or "redact_or_generalize"
        return finding

    finding.explanation = (
        f"The span was recalled as {finding.risk_type}. The surrounding document context is "
        f"{document_context.document_type}/{document_context.scene_type}, so the item should remain in governance "
        "unless a reviewer confirms it is only an example or public reference."
    )
    privacy_context["context_explanation"] = finding.explanation
    return finding


def _merge_local_findings(
    findings: list[DetectionFinding],
    local_result: PrivacyDetectionResult | None,
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
) -> list[PrivacyDetectionResult]:
    settings = settings or get_settings()
    provider = resolve_provider_config(settings)
    context_by_doc = {item.doc_id: item for item in (document_contexts or [])}
    local_results_by_doc: dict[str, PrivacyDetectionResult] = {}
    if provider.mode == "local_model":
        local_results_by_doc = {
            item.doc_id: item for item in local_privacy_detection.run(ingest_units, settings)
        }
    results: list[PrivacyDetectionResult] = []

    for unit in ingest_units:
        document_context = context_by_doc.get(unit.doc_id)
        local_result = local_results_by_doc.get(unit.doc_id)
        if provider.mode == "local_model":
            local_findings = list(local_result.findings if local_result else [])
            local_findings = [_apply_contextual_privacy_defaults(finding, document_context) for finding in local_findings]
            _add_combined_identity_if_needed(unit, local_findings, settings)
            local_findings = _suppress_public_organization_findings(local_findings)
            local_findings = _filter_selected_findings(local_findings, _selected_entity_types(settings))
            results.append(
                PrivacyDetectionResult(
                    run_id=unit.run_id,
                    doc_id=unit.doc_id,
                    text_hash=unit.text_hash,
                    pii_count=len(local_findings),
                    risk_score=round(
                        max(
                            (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in local_findings),
                            default=0.0,
                        ),
                        4,
                    ),
                    summary=(
                        f"Local privacy recall completed with {len(local_findings)} candidate findings."
                        if local_findings
                        else "Local privacy recall found no candidate PII."
                    ),
                    findings=local_findings,
                    needs_adjudication=bool(local_findings),
                    hard_case_reasons=sorted(set(list(local_result.hard_case_reasons if local_result else []) + (["manual_resolution_needed"] if local_findings else []))),
                    provider_name="local_privacy_detector",
                    provider_version=provider.model,
                    is_degraded=bool(local_result and local_result.is_degraded),
                )
            )
            continue

        client = OpenAICompatibleComplianceClient(settings)
        system_prompt = load_prompt(str(settings.api_privacy_detection_prompt_path))
        try:
            payload = client.complete_json(
                task_name="privacy_detection",
                system_prompt=system_prompt,
                payload=_request_payload(unit, settings, document_context),
            )
        except (OpenAICompatibleAPIError, Exception) as exc:
            logger.warning("API privacy detection failed for %s: %s", unit.doc_id, exc)
            results.append(_fallback_result(unit, settings, str(exc)))
            continue

        invalid_span_payloads: list[dict[str, Any]] = []
        findings = [
            finding
            for item in _dict_items(payload.get("findings", []))
            for finding in [_finding(unit, item, settings, document_context, invalid_span_payloads)]
            if finding is not None
        ]
        findings = _merge_local_findings(findings, local_result)
        findings = [_apply_contextual_privacy_defaults(finding, document_context) for finding in findings]
        if invalid_span_payloads:
            findings.append(_invalid_span_finding(unit, invalid_span_payloads, settings))
        _add_combined_identity_if_needed(unit, findings, settings)
        findings = _suppress_public_organization_findings(findings)
        findings = _filter_selected_findings(findings, _selected_entity_types(settings))
        missing_structured_findings = not findings and _looks_like_pii_summary(payload)
        if missing_structured_findings:
            findings.append(
                DetectionFinding(
                    doc_id=unit.doc_id,
                    finding_type="privacy",
                    risk_type="api_privacy_missing_structured_findings",
                    policy_tag="pii.api_missing_structured_findings",
                    severity=Severity.MEDIUM,
                    confidence=0.5,
                    explanation=(
                        "API privacy response indicated possible PII but did not return "
                        "structured span findings required for deterministic redaction."
                    ),
                    source_tool=f"{settings.api_compliance_source_tool_prefix}.privacy",
                    remediation_suggestion="manual_review",
                    redaction_suggestion="",
                    needs_adjudication=True,
                    hard_case_reason="api_privacy_missing_structured_findings",
                    span=None,
                    attributes={"api_payload": payload},
                )
            )
        needs_adjudication = bool(payload.get("needs_adjudication")) or any(finding.needs_adjudication for finding in findings)
        hard_case_reasons = _string_items(payload.get("hard_case_reasons", []))
        if local_result is not None:
            needs_adjudication = needs_adjudication or local_result.needs_adjudication
            hard_case_reasons.extend(local_result.hard_case_reasons)
        if missing_structured_findings:
            hard_case_reasons.append("api_privacy_missing_structured_findings")
        if needs_adjudication and "manual_resolution_needed" not in hard_case_reasons:
            hard_case_reasons.append("manual_resolution_needed")

        risk_score = payload.get("risk_score")
        try:
            risk_score_f = float(risk_score)
        except (TypeError, ValueError):
            risk_score_f = max(
                (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in findings),
                default=0.0,
            )

        results.append(
            PrivacyDetectionResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                pii_count=len(findings),
                risk_score=round(max(0.0, min(risk_score_f, 1.0)), 4),
                summary=str(payload.get("summary") or f"API privacy detection completed with {len(findings)} findings."),
                findings=findings,
                needs_adjudication=needs_adjudication,
                hard_case_reasons=sorted(set(hard_case_reasons)),
                provider_name="local_privacy_detector" if provider.mode == "local_model" else "api_privacy_detector",
                provider_version=provider.model,
                is_degraded=bool(local_result and local_result.is_degraded),
            )
        )

    logger.info("API privacy detection completed: %d documents", len(results))
    return results

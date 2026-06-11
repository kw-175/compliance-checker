from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DetectionFinding,
    IngestUnit,
    PrivacyDetectionResult,
    Severity,
    TextSpan,
)

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    Severity.LOW: 0.30,
    Severity.MEDIUM: 0.55,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 1.00,
}
REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
}


@lru_cache(maxsize=4)
def _load_rules(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_flags(flag_names: list[str]) -> int:
    value = 0
    for name in flag_names:
        value |= REGEX_FLAGS.get(name, 0)
    return value


def _context(text: str, start: int, end: int, window: int = 40) -> tuple[str, str]:
    before = text[max(0, start - window):start]
    after = text[end:min(len(text), end + window)]
    return before, after


def _build_finding(
    unit: IngestUnit,
    *,
    policy_tag: str,
    risk_type: str,
    severity: Severity,
    replacement: str,
    match_text: str,
    start: int,
    end: int,
    source_tool: str,
    needs_adjudication: bool,
    hard_case_reason: str,
) -> DetectionFinding:
    before, after = _context(unit.text, start, end)
    confidence = min(0.99, SEVERITY_WEIGHTS[severity] + 0.08)
    if needs_adjudication:
        confidence = max(0.4, confidence - 0.22)

    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="privacy",
        risk_type=risk_type,
        policy_tag=policy_tag,
        severity=severity,
        confidence=round(confidence, 4),
        explanation=f"Matched privacy pattern for {risk_type}.",
        source_tool=source_tool,
        remediation_suggestion="redact" if replacement else "manual_review",
        redaction_suggestion=replacement,
        needs_adjudication=needs_adjudication,
        hard_case_reason=hard_case_reason,
        span=TextSpan(
            start=start,
            end=end,
            text=match_text,
            context_before=before,
            context_after=after,
        ),
    )


def _deduplicate(findings: list[DetectionFinding]) -> list[DetectionFinding]:
    deduped: dict[tuple[int, int, str], DetectionFinding] = {}
    for finding in findings:
        if finding.span is None:
            continue
        key = (finding.span.start, finding.span.end, finding.policy_tag)
        existing = deduped.get(key)
        if existing is None or finding.confidence > existing.confidence:
            deduped[key] = finding
    return list(deduped.values())


def run(
    ingest_units: list[IngestUnit],
    settings: Settings | None = None,
) -> list[PrivacyDetectionResult]:
    settings = settings or get_settings()
    rules = _load_rules(str(settings.pii_rules_path))
    pattern_rules = rules.get("patterns", {})
    combination_rules = rules.get("combination_rules", {})

    results: list[PrivacyDetectionResult] = []
    for unit in ingest_units:
        findings: list[DetectionFinding] = []
        hard_case_reasons: list[str] = []
        distinct_types: set[str] = set()

        for rule_name, rule in pattern_rules.items():
            pattern = str(rule["regex"])
            flags = _resolve_flags(list(rule.get("flags", [])))
            compiled = re.compile(pattern, flags)
            capture_group = int(rule.get("capture_group", 0))
            severity = Severity(rule["severity"])
            policy_tag = str(rule["policy_tag"])
            risk_type = str(rule["risk_type"])
            replacement = str(rule.get("redaction", ""))

            for match in compiled.finditer(unit.text):
                start, end = match.span(capture_group)
                match_text = match.group(capture_group)
                needs_adjudication = risk_type in {"person_name", "education_record", "bank_card"}
                hard_case_reason = "context_dependent_pii" if needs_adjudication else ""
                if needs_adjudication and "context_dependent_pii" not in hard_case_reasons:
                    hard_case_reasons.append("context_dependent_pii")

                findings.append(
                    _build_finding(
                        unit,
                        policy_tag=policy_tag,
                        risk_type=risk_type,
                        severity=severity,
                        replacement=replacement,
                        match_text=match_text,
                        start=start,
                        end=end,
                        source_tool=f"privacy_rule_engine.{rule_name}",
                        needs_adjudication=needs_adjudication,
                        hard_case_reason=hard_case_reason,
                    )
                )
                distinct_types.add(risk_type)

        findings = _deduplicate(findings)

        combined_rule = combination_rules.get("combined_identity")
        if combined_rule:
            base_types = set(combined_rule.get("base_types", []))
            matched_base_types = {finding.risk_type for finding in findings if finding.risk_type in base_types}
            if len(matched_base_types) >= settings.privacy_combination_threshold:
                severity = Severity(combined_rule["severity"])
                needs_adjudication = True
                hard_case_reasons.append("combined_identity")
                findings.append(
                    DetectionFinding(
                        doc_id=unit.doc_id,
                        finding_type="privacy",
                        risk_type=str(combined_rule["risk_type"]),
                        policy_tag=str(combined_rule["policy_tag"]),
                        severity=severity,
                        confidence=0.92,
                        explanation=(
                            f"Detected {len(matched_base_types)} distinct identity attributes "
                            "that can combine into a stronger personal profile."
                        ),
                        source_tool="privacy_rule_engine.combined_identity",
                        remediation_suggestion="restrict_and_review",
                        redaction_suggestion=str(combined_rule.get("redaction", "")),
                        needs_adjudication=needs_adjudication,
                        hard_case_reason="combined_identity",
                        span=None,
                        attributes={"matched_types": sorted(matched_base_types)},
                    )
                )

        risk_score = max(
            (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in findings),
            default=0.0,
        )
        needs_adjudication = any(finding.needs_adjudication for finding in findings)
        if needs_adjudication and "manual_resolution_needed" not in hard_case_reasons:
            hard_case_reasons.append("manual_resolution_needed")

        summary = "No privacy risks detected."
        if findings:
            summary = f"Detected {len(findings)} privacy findings."

        results.append(
            PrivacyDetectionResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                pii_count=len(findings),
                risk_score=round(risk_score, 4),
                summary=summary,
                findings=findings,
                needs_adjudication=needs_adjudication,
                hard_case_reasons=sorted(set(hard_case_reasons)),
            )
        )

    logger.info("Privacy detection completed: %d documents", len(results))
    return results

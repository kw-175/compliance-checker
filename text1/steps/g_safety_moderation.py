from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    ContentSafetyResult,
    DetectionFinding,
    DetectionStatus,
    IngestUnit,
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


@lru_cache(maxsize=4)
def _load_rules(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _find_matches(text: str, keyword: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    haystack = text.lower()
    needle = keyword.lower()
    start = 0
    while True:
        index = haystack.find(needle, start)
        if index == -1:
            return matches
        matches.append((index, index + len(needle)))
        start = index + len(needle)


def _context(text: str, start: int, end: int, window: int = 40) -> tuple[str, str]:
    before = text[max(0, start - window):start]
    after = text[end:min(len(text), end + window)]
    return before, after


def _context_is_ambiguous(text: str, start: int, end: int, markers: list[str]) -> bool:
    before, after = _context(text, start, end, 50)
    snippet = f"{before} {text[start:end]} {after}".lower()
    return any(marker.lower() in snippet for marker in markers)


def _build_finding(
    unit: IngestUnit,
    *,
    risk_type: str,
    policy_tag: str,
    severity: Severity,
    keyword: str,
    start: int,
    end: int,
    ambiguous: bool,
) -> DetectionFinding:
    before, after = _context(unit.text, start, end)
    confidence = min(0.98, SEVERITY_WEIGHTS[severity] + 0.1)
    if ambiguous:
        confidence = max(0.35, confidence - 0.28)

    explanation = f"Matched content keyword '{keyword}'"
    if ambiguous:
        explanation += " in an educational/reporting context"

    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="content_safety",
        risk_type=risk_type,
        policy_tag=policy_tag,
        severity=severity,
        confidence=round(confidence, 4),
        explanation=explanation,
        source_tool="content_rule_engine",
        remediation_suggestion="manual_review" if ambiguous else "block_or_isolate",
        needs_adjudication=ambiguous,
        hard_case_reason="ambiguous_context" if ambiguous else "",
        span=TextSpan(
            start=start,
            end=end,
            text=unit.text[start:end],
            context_before=before,
            context_after=after,
        ),
        attributes={"matched_keyword": keyword},
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
) -> list[ContentSafetyResult]:
    settings = settings or get_settings()
    rules = _load_rules(str(settings.content_rules_path))
    categories = rules.get("categories", {})
    markers = list(rules.get("ambiguous_context_markers", []))

    results: list[ContentSafetyResult] = []
    for unit in ingest_units:
        findings: list[DetectionFinding] = []
        hard_case_reasons: list[str] = []

        for category_name, category_rule in categories.items():
            severity = Severity(category_rule["severity"])
            risk_type = str(category_rule["risk_type"])
            policy_tag = str(category_rule["policy_tag"])
            for keyword in category_rule.get("keywords", []):
                for start, end in _find_matches(unit.text, str(keyword)):
                    ambiguous = _context_is_ambiguous(unit.text, start, end, markers)
                    if ambiguous and "ambiguous_context" not in hard_case_reasons:
                        hard_case_reasons.append("ambiguous_context")
                    findings.append(
                        _build_finding(
                            unit,
                            risk_type=risk_type,
                            policy_tag=policy_tag,
                            severity=severity,
                            keyword=str(keyword),
                            start=start,
                            end=end,
                            ambiguous=ambiguous,
                        )
                    )

        findings = _deduplicate(findings)
        risk_score = max(
            (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in findings),
            default=0.0,
        )

        needs_adjudication = any(finding.needs_adjudication for finding in findings)
        if findings and settings.safety_hard_case_score_floor <= risk_score <= settings.safety_hard_case_score_ceiling:
            needs_adjudication = True
            if "score_band_uncertain" not in hard_case_reasons:
                hard_case_reasons.append("score_band_uncertain")

        if not findings:
            status = DetectionStatus.CLEAR
            summary = "No content safety issues detected."
        elif needs_adjudication:
            status = DetectionStatus.HARD_CASE
            summary = f"{len(findings)} content safety findings require hard-case adjudication."
        else:
            status = DetectionStatus.FLAGGED
            summary = f"{len(findings)} content safety findings detected."

        results.append(
            ContentSafetyResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                status=status,
                risk_score=round(risk_score, 4),
                summary=summary,
                findings=findings,
                needs_adjudication=needs_adjudication,
                hard_case_reasons=hard_case_reasons,
            )
        )

    logger.info("Content safety detection completed: %d documents", len(results))
    return results

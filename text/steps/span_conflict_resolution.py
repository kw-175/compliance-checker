from __future__ import annotations

import logging
from dataclasses import dataclass

from text.models.schemas import (
    DetectionFinding,
    IngestUnit,
    PrivacyDetectionResult,
    RedactionConflict,
    RedactionTarget,
    Severity,
    SpanConflictResolutionResult,
)

logger = logging.getLogger(__name__)

SEVERITY_RANK = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

RISK_PRIORITY = {
    "id_card": 100,
    "email": 98,
    "bank_card": 92,
    "bank_account": 92,
    "payment_account": 90,
    "secret": 96,
    "api_key": 96,
    "token": 96,
    "password": 96,
    "phone": 85,
    "phone_number": 85,
    "student_id": 80,
    "medical_record": 82,
    "psychological_record": 82,
    "education_record": 79,
    "minor_info": 77,
    "parent_contact": 78,
    "social_account": 76,
    "vehicle_identifier": 75,
    "address": 74,
    "person_name": 60,
    "organization": 55,
    "url": 30,
    "pii_entity": 20,
}

DEFAULT_REPLACEMENTS = {
    "id_card": "<ID_CARD>",
    "email": "<EMAIL>",
    "bank_card": "<BANK_CARD>",
    "bank_account": "<BANK_ACCOUNT>",
    "payment_account": "<PAYMENT_ACCOUNT>",
    "secret": "<SECRET>",
    "api_key": "<SECRET>",
    "token": "<SECRET>",
    "password": "<SECRET>",
    "phone": "<PHONE>",
    "phone_number": "<PHONE>",
    "student_id": "<STUDENT_ID>",
    "parent_contact": "<PARENT_CONTACT>",
    "social_account": "<SOCIAL_ACCOUNT>",
    "vehicle_identifier": "<LICENSE_PLATE>",
    "address": "<ADDRESS>",
    "person_name": "<PERSON>",
    "education_record": "<EDU_RECORD>",
    "medical_record": "<MEDICAL_RECORD>",
    "psychological_record": "<MEDICAL_RECORD>",
    "minor_info": "<MINOR_INFO>",
    "organization": "<ORGANIZATION>",
    "url": "<URL>",
}

LABEL_ONLY_TEXT = {
    "person_name": {
        "姓名", "名字", "学生", "家长", "联系人", "本人手机号", "联系电话", "联系方式",
        "家庭住址", "家庭住址登记", "家庭住址登记为", "name", "student", "parent", "guardian"
    },
}


@dataclass(frozen=True)
class _Candidate:
    finding: DetectionFinding
    start: int
    end: int
    text: str
    replacement: str

    @property
    def finding_id(self) -> str:
        return self.finding.finding_id

    @property
    def risk_type(self) -> str:
        return self.finding.risk_type


def _replacement_for(finding: DetectionFinding) -> str:
    replacement = finding.redaction_suggestion or finding.remediation_suggestion
    if replacement and replacement.startswith("<") and replacement.endswith(">"):
        return replacement
    return DEFAULT_REPLACEMENTS.get(finding.risk_type, "<REDACTED>")


def _source_adjustment(source_tool: str) -> int:
    source = source_tool.lower()
    if "phone_generic" in source:
        return -25
    if "bank_card" in source:
        return -5
    if "id_card" in source:
        return 10
    if "presidio" in source or "gliner" in source:
        return 3
    return 0


def _score(candidate: _Candidate) -> tuple[int, int, float, int]:
    risk_score = RISK_PRIORITY.get(candidate.risk_type, 10) + _source_adjustment(candidate.finding.source_tool)
    severity_score = SEVERITY_RANK.get(candidate.finding.severity, 0)
    length = candidate.end - candidate.start
    return risk_score, severity_score, candidate.finding.confidence, length


def _overlaps(left: _Candidate, right: _Candidate) -> bool:
    return left.start < right.end and right.start < left.end


def _overlap_bounds(left: _Candidate, right: _Candidate) -> tuple[int, int]:
    return min(left.start, right.start), max(left.end, right.end)


def _label_only(candidate: _Candidate) -> bool:
    labels = LABEL_ONLY_TEXT.get(candidate.risk_type, set())
    return candidate.text.strip().lower() in labels


def _invalid_candidate(candidate: _Candidate) -> tuple[bool, str]:
    text = candidate.text.strip()
    source = candidate.finding.source_tool.lower()
    if candidate.risk_type == "person_name" and (
        any(token in text for token in ("手机号", "联系电话", "联系方式", "住址", "邮箱"))
        or ("person_name_cn" in source and candidate.finding.confidence <= 0.45 and len(text) > 4)
    ):
        return True, "Suppressed a low-confidence person-name span that is a field label or surrounding context."
    if candidate.risk_type == "phone_number" and "phone_generic" in source:
        digits = "".join(ch for ch in text if ch.isdigit())
        context = (
            candidate.finding.span.context_before[-16:] if candidate.finding.span else ""
        ) + text + (
            candidate.finding.span.context_after[:16] if candidate.finding.span else ""
        )
        has_phone_context = any(token in context for token in ("电话", "手机号", "联系方式", "联系电话", "tel", "phone"))
        looks_cn_mobile = len(digits) == 11 and digits.startswith("1")
        has_separator = any(ch in text for ch in "- ()")
        if not looks_cn_mobile and not has_separator and not has_phone_context:
            return True, "Suppressed a generic numeric span without phone-number context."
    if candidate.risk_type == "address" and (
        text.startswith(("、", "，", ",")) or "共同出现" in text or "唯一定位" in text
    ):
        return True, "Suppressed a summary sentence that is not a concrete address span."
    return False, ""


def _candidate_from_finding(unit: IngestUnit, finding: DetectionFinding) -> _Candidate | None:
    if finding.finding_type != "privacy" or finding.span is None:
        return None
    start = finding.span.start
    end = finding.span.end
    if start < 0 or end <= start or end > len(unit.text):
        return None
    if finding.span.text and finding.span.text != unit.text[start:end]:
        logger.warning(
            "Suppressed mismatched privacy span for %s: finding=%s start=%s end=%s",
            unit.doc_id,
            finding.finding_id,
            start,
            end,
        )
        return None
    return _Candidate(
        finding=finding,
        start=start,
        end=end,
        text=unit.text[start:end],
        replacement=_replacement_for(finding),
    )


def _target_from_candidate(candidate: _Candidate) -> RedactionTarget:
    return RedactionTarget(
        finding_id=candidate.finding_id,
        event_id="",
        start=candidate.start,
        end=candidate.end,
        original_text=candidate.text,
        replacement=candidate.replacement,
        pii_type=candidate.risk_type,
    )


def _suppression_conflict(
    *,
    unit: IngestUnit,
    conflict_type: str,
    suppressed: _Candidate,
    selected: _Candidate | None,
    rationale: str,
) -> RedactionConflict:
    start, end = (suppressed.start, suppressed.end) if selected is None else _overlap_bounds(suppressed, selected)
    return RedactionConflict(
        conflict_type=conflict_type,
        start=start,
        end=end,
        text=unit.text[start:end],
        selected_finding_id=selected.finding_id if selected else "",
        selected_risk_type=selected.risk_type if selected else "",
        suppressed_finding_ids=[suppressed.finding_id],
        suppressed_risk_types=[suppressed.risk_type],
        resolution_source="deterministic_priority",
        rationale=rationale,
    )


def _resolve_doc(unit: IngestUnit, privacy_result: PrivacyDetectionResult | None) -> SpanConflictResolutionResult:
    raw_findings = privacy_result.findings if privacy_result else []
    candidates: list[_Candidate] = []
    conflicts: list[RedactionConflict] = []

    for finding in raw_findings:
        candidate = _candidate_from_finding(unit, finding)
        if candidate is None:
            continue
        if _label_only(candidate):
            conflicts.append(
                _suppression_conflict(
                    unit=unit,
                    conflict_type="label_only_span",
                    suppressed=candidate,
                    selected=None,
                    rationale="Suppressed a field label token that is not itself personal data.",
                )
            )
            continue
        invalid, rationale = _invalid_candidate(candidate)
        if invalid:
            conflicts.append(
                _suppression_conflict(
                    unit=unit,
                    conflict_type="invalid_privacy_span",
                    suppressed=candidate,
                    selected=None,
                    rationale=rationale,
                )
            )
            continue
        candidates.append(candidate)

    selected: list[_Candidate] = []
    for candidate in sorted(candidates, key=_score, reverse=True):
        overlapping_selected = [item for item in selected if _overlaps(candidate, item)]
        if not overlapping_selected:
            selected.append(candidate)
            continue

        winner = max(
            overlapping_selected,
            key=lambda item: (
                min(candidate.end, item.end) - max(candidate.start, item.start),
                _score(item),
            ),
        )
        conflicts.append(
            _suppression_conflict(
                unit=unit,
                conflict_type="overlap",
                suppressed=candidate,
                selected=winner,
                rationale=(
                    f"Suppressed overlapping {candidate.risk_type} span in favor of "
                    f"{winner.risk_type} using deterministic risk priority."
                ),
            )
        )

    redaction_targets = [_target_from_candidate(candidate) for candidate in sorted(selected, key=lambda item: (item.start, item.end))]
    suppressed_count = sum(len(conflict.suppressed_finding_ids) for conflict in conflicts)
    return SpanConflictResolutionResult(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        input_finding_count=sum(1 for finding in raw_findings if finding.span is not None),
        selected_span_count=len(redaction_targets),
        suppressed_finding_count=suppressed_count,
        redaction_targets=redaction_targets,
        conflicts=conflicts,
        needs_model_resolution=False,
        is_degraded=False,
        summary=(
            f"Selected {len(redaction_targets)} redaction spans and suppressed "
            f"{suppressed_count} conflicting spans."
        ),
    )


def run(
    ingest_units: list[IngestUnit],
    privacy_results: list[PrivacyDetectionResult],
) -> list[SpanConflictResolutionResult]:
    privacy_by_doc = {result.doc_id: result for result in privacy_results}
    results = [_resolve_doc(unit, privacy_by_doc.get(unit.doc_id)) for unit in ingest_units]
    logger.info("Span conflict resolution completed: %d documents", len(results))
    return results

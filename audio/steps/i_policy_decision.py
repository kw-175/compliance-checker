"""
Step I: policy decision.
"""

from __future__ import annotations

import logging

from audio.config.settings import Settings
from audio.models.schemas import Decision, EvidenceBundle, PolicyDecision, SafetyLevel, TranscriptEvidence, UnitDecision

logger = logging.getLogger(__name__)


def _evaluate_unit(unit: TranscriptEvidence) -> UnitDecision:
    scores: dict[str, float] = {}
    reasons: list[str] = []

    scores["secrets"] = 0.0 if unit.secret_hits else 1.0
    if unit.secret_hits:
        reasons.append(f"{len(unit.secret_hits)} secret hit(s)")

    if unit.safety and unit.safety.safety_level == SafetyLevel.UNSAFE:
        scores["safety"] = 0.0
        reasons.append("unsafe content")
    elif unit.safety and unit.safety.safety_level == SafetyLevel.CONTROVERSIAL:
        scores["safety"] = 0.5
        reasons.append("controversial content")
    else:
        scores["safety"] = 1.0

    pii_count = unit.privacy.pii_count if unit.privacy else 0
    if pii_count > 3:
        scores["privacy"] = 0.3
        reasons.append(f"high PII density ({pii_count})")
    elif pii_count > 0:
        scores["privacy"] = 0.7
    else:
        scores["privacy"] = 1.0

    if any(any(token in lic.license_expression.lower() for token in ["gpl", "agpl"]) for hit in unit.compliance_hits for lic in hit.licenses):
        scores["compliance"] = 0.2
        reasons.append("copyleft license signal")
    else:
        scores["compliance"] = 1.0

    text_hits = len(unit.keyword_hits) + len(unit.regex_hits)
    if text_hits > 10:
        scores["text_scan"] = 0.2
        reasons.append(f"dense rule hits ({text_hits})")
    elif text_hits > 0:
        scores["text_scan"] = 0.6
    else:
        scores["text_scan"] = 1.0

    worst = min(scores.values()) if scores else 1.0
    if worst <= 0.0:
        decision = Decision.REJECT
    elif worst <= 0.3:
        decision = Decision.QUARANTINE
    elif worst <= 0.6:
        decision = Decision.REVIEW
    else:
        decision = Decision.ALLOW

    return UnitDecision(unit_id=unit.unit_id, decision=decision, reasons=reasons, scores=scores)


def _local_decision(bundle: EvidenceBundle) -> PolicyDecision:
    decisions = [_evaluate_unit(unit) for unit in bundle.transcript_units]
    priority = {Decision.REJECT: 0, Decision.QUARANTINE: 1, Decision.REVIEW: 2, Decision.ALLOW: 3}
    overall = min((decision.decision for decision in decisions), key=lambda item: priority[item], default=Decision.ALLOW)
    return PolicyDecision(pipeline_run_id=bundle.pipeline_run_id, overall_decision=overall, unit_decisions=decisions)


def _query_opa(bundle: EvidenceBundle, settings: Settings) -> PolicyDecision | None:
    try:
        import httpx
    except ImportError:
        logger.warning("httpx unavailable, skip OPA and fallback to local rules")
        return None

    payload = {
        "input": {
            "pipeline_run_id": bundle.pipeline_run_id,
            "summary": bundle.summary,
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "source_id": unit.source_id,
                    "is_duplicate": unit.is_duplicate,
                    "secret_count": len(unit.secret_hits),
                    "compliance_count": len(unit.compliance_hits),
                    "keyword_count": len(unit.keyword_hits),
                    "regex_count": len(unit.regex_hits),
                    "pii_count": unit.privacy.pii_count if unit.privacy else 0,
                    "safety_level": unit.safety.safety_level.value if unit.safety else "safe",
                }
                for unit in bundle.transcript_units
            ],
        }
    }

    url = f"{settings.opa_url.rstrip('/')}/{settings.opa_policy_path.lstrip('/')}"
    try:
        response = httpx.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json().get("result", {})

        decisions: list[UnitDecision] = []
        for item in result.get("unit_decisions", []):
            raw_decision = str(item.get("decision", "review")).lower()
            try:
                parsed = Decision(raw_decision)
            except ValueError:
                parsed = Decision.REVIEW
            decisions.append(
                UnitDecision(
                    unit_id=str(item.get("unit_id", "")),
                    decision=parsed,
                    reasons=[str(reason) for reason in item.get("reasons", [])],
                    scores={str(key): float(value) for key, value in item.get("scores", {}).items()},
                )
            )

        raw_overall = str(result.get("overall_decision", "review")).lower()
        try:
            overall = Decision(raw_overall)
        except ValueError:
            overall = Decision.REVIEW

        return PolicyDecision(
            pipeline_run_id=bundle.pipeline_run_id,
            overall_decision=overall,
            unit_decisions=decisions,
        )
    except Exception as exc:
        logger.warning("OPA evaluation failed, fallback to local rules: %s", exc)
        return None


def run(bundle: EvidenceBundle, settings: Settings | None = None) -> PolicyDecision:
    if settings is None:
        from audio.config.settings import get_settings
        settings = get_settings()
    if settings.opa_enabled:
        opa_result = _query_opa(bundle, settings)
        if opa_result is not None:
            return opa_result
    return _local_decision(bundle)

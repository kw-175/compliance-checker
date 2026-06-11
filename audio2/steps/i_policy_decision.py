"""
Step I: privacy and safety policy decision.
"""

from __future__ import annotations

import logging
from typing import Any

from audio.config.settings import Settings, get_settings
from audio.models.schemas import Decision, EvidenceBundle, PolicyDecision, SafetyLevel, TranscriptEvidence, UnitDecision

logger = logging.getLogger(__name__)

DECISION_PRIORITY = {
    Decision.ALLOW: 0,
    Decision.REVIEW: 1,
    Decision.QUARANTINE: 2,
    Decision.REJECT: 3,
}


def _max_decision(left: Decision, right: Decision) -> Decision:
    return right if DECISION_PRIORITY[right] > DECISION_PRIORITY[left] else left


def _evaluate_unit(unit: TranscriptEvidence, extension_scores: dict[str, Any] | None = None) -> UnitDecision:
    scores: dict[str, float] = {}
    reasons: list[str] = []

    if unit.safety and unit.safety.safety_level == SafetyLevel.UNSAFE:
        scores["safety"] = 0.0
        reasons.append("unsafe content")
        decision = Decision.REJECT
    elif unit.safety and unit.safety.safety_level == SafetyLevel.CONTROVERSIAL:
        scores["safety"] = 0.5
        reasons.append("controversial content")
        decision = Decision.REVIEW
    else:
        scores["safety"] = 1.0
        decision = Decision.ALLOW

    pii_count = unit.privacy.pii_count if unit.privacy else 0
    if pii_count > 3:
        scores["privacy"] = 0.3
        reasons.append(f"high PII density ({pii_count})")
        decision = _max_decision(decision, Decision.QUARANTINE)
    elif pii_count > 0:
        scores["privacy"] = 0.7
        reasons.append(f"PII detected ({pii_count})")
        decision = _max_decision(decision, Decision.REVIEW)
    else:
        scores["privacy"] = 1.0

    if unit.hard_case:
        hard_case = unit.hard_case
        hard_decision = hard_case.judgement.recommended_decision
        scores["hard_case_confidence"] = hard_case.judgement.confidence
        scores["hard_case_uncertainty"] = hard_case.uncertainty
        if hard_decision != Decision.ALLOW:
            decision = _max_decision(decision, hard_decision)
            reasons.append(f"hard-case adjudication: {hard_decision.value}")
        if hard_case.judgement.requires_manual_review:
            decision = _max_decision(decision, Decision.REVIEW)
            if "hard-case manual review" not in reasons:
                reasons.append("hard-case manual review")
        if hard_case.is_degraded:
            reasons.append("hard-case fallback used")

    for key, raw_score in (extension_scores or {}).items():
        try:
            scores[f"extension:{key}"] = float(raw_score)
        except (TypeError, ValueError):
            continue

    return UnitDecision(unit_id=unit.unit_id, decision=decision, reasons=reasons, scores=scores)


def _local_decision(bundle: EvidenceBundle, extension_scores: dict[str, Any] | None = None) -> PolicyDecision:
    decisions = [
        _evaluate_unit(unit, extension_scores=extension_scores)
        for unit in bundle.transcript_units
    ]
    overall = Decision.ALLOW
    for decision in decisions:
        overall = _max_decision(overall, decision.decision)
    return PolicyDecision(
        pipeline_run_id=bundle.pipeline_run_id,
        overall_decision=overall,
        unit_decisions=decisions,
        trust_level=bundle.trust_level,
        degrade_summary="" if bundle.trust_level == "full" else "Hard-case adjudication used a degraded fallback provider.",
        profile_name="audio-privacy-safety-v1",
    )


def run(
    bundle: EvidenceBundle,
    settings: Settings | None = None,
    extension_scores: dict[str, Any] | None = None,
) -> PolicyDecision:
    settings = settings or get_settings()
    if getattr(settings, "opa_enabled", False):
        logger.info("OPA is configured but audio privacy/safety policy currently uses local rules.")
    return _local_decision(bundle, extension_scores=extension_scores)

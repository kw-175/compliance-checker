"""
Hard-case adjudication for uncertain audio compliance records.
"""

from __future__ import annotations

import logging

from audio.adapters import qwen_hard_case_adapter
from audio.config.settings import Settings, get_settings
from audio.models.schemas import (
    AudioHardCaseJudgement,
    AudioHardCaseResult,
    Decision,
    PrivacyResult,
    SafetyLevel,
    SafetyResult,
    TranscriptUnit,
)

logger = logging.getLogger(__name__)

DECISION_PRIORITY = {
    Decision.ALLOW: 0,
    Decision.REVIEW: 1,
    Decision.QUARANTINE: 2,
    Decision.REJECT: 3,
}


def _max_decision(left: Decision, right: Decision) -> Decision:
    return right if DECISION_PRIORITY[right] > DECISION_PRIORITY[left] else left


def _add_trigger(sources: list[str], reasons: list[str], source: str, reason: str) -> None:
    if source not in sources:
        sources.append(source)
    if reason not in reasons:
        reasons.append(reason)


def _risk_present(privacy: PrivacyResult | None, safety: SafetyResult | None) -> bool:
    if privacy and privacy.pii_count > 0:
        return True
    return bool(safety and safety.safety_level != SafetyLevel.SAFE)


def _trigger_sources(
    unit: TranscriptUnit,
    privacy: PrivacyResult | None,
    safety: SafetyResult | None,
    settings: Settings,
) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    reasons: list[str] = []

    if safety:
        if safety.safety_level == SafetyLevel.CONTROVERSIAL:
            _add_trigger(sources, reasons, "content_safety", "controversial_content")
        if safety.is_degraded and safety.safety_level != SafetyLevel.SAFE:
            _add_trigger(sources, reasons, "content_safety", "safety_detector_degraded")
        if (
            safety.safety_level != SafetyLevel.SAFE
            and settings.hard_case_safety_score_floor <= safety.score <= settings.hard_case_safety_score_ceiling
        ):
            _add_trigger(sources, reasons, "content_safety", "safety_score_band_uncertain")
        raw = f"{safety.raw_output} {safety.explanation}".lower()
        if any(marker in raw for marker in ("borderline", "uncertain", "ambiguous", "review")):
            _add_trigger(sources, reasons, "content_safety", "safety_model_boundary_language")

    if privacy:
        if privacy.is_degraded and privacy.pii_count > 0:
            _add_trigger(sources, reasons, "privacy", "privacy_detector_degraded")
        ceiling = settings.pii_score_threshold + settings.hard_case_privacy_score_margin
        borderline_entities = [
            entity
            for entity in privacy.pii_entities
            if settings.pii_score_threshold <= entity.score <= ceiling
        ]
        if borderline_entities:
            _add_trigger(sources, reasons, "privacy", "privacy_score_band_uncertain")

    if _risk_present(privacy, safety):
        if 0 < unit.confidence < settings.hard_case_asr_confidence_threshold:
            _add_trigger(sources, reasons, "asr", "low_asr_confidence")
        if "fallback" in (unit.engine_name or "").lower():
            _add_trigger(sources, reasons, "asr", "fallback_transcript_engine")

    return sources, reasons


def _heuristic_judgement(
    unit: TranscriptUnit,
    privacy: PrivacyResult | None,
    safety: SafetyResult | None,
    trigger_reasons: list[str],
) -> AudioHardCaseJudgement:
    content_status = "clear"
    privacy_status = "clear"
    recommended = Decision.ALLOW
    confidence = 0.92
    requires_manual_review = False
    reasons: list[str] = list(trigger_reasons)
    rationale_parts: list[str] = []

    if safety:
        if safety.safety_level == SafetyLevel.UNSAFE:
            content_status = "unsafe"
            recommended = _max_decision(recommended, Decision.REJECT)
            confidence = min(confidence, 0.88)
            rationale_parts.append("Preliminary safety detection marks the unit unsafe.")
        elif safety.safety_level == SafetyLevel.CONTROVERSIAL:
            content_status = "borderline"
            recommended = _max_decision(recommended, Decision.REVIEW)
            confidence = min(confidence, 0.58)
            requires_manual_review = True
            rationale_parts.append("The safety signal is contextual or controversial.")
        if safety.is_degraded and safety.safety_level != SafetyLevel.SAFE:
            confidence = min(confidence, 0.6)
            requires_manual_review = True
            rationale_parts.append("The safety detector used a degraded provider.")

    if privacy:
        if privacy.is_degraded and privacy.pii_count > 0:
            privacy_status = "borderline"
            recommended = _max_decision(recommended, Decision.REVIEW)
            confidence = min(confidence, 0.55)
            requires_manual_review = True
            rationale_parts.append("PII was detected while the privacy detector was degraded.")
        elif privacy.pii_count > 3:
            privacy_status = "contains_pii"
            recommended = _max_decision(recommended, Decision.QUARANTINE)
            confidence = min(confidence, 0.7)
            rationale_parts.append("The unit contains dense PII and should be held.")
        elif privacy.pii_count > 0:
            privacy_status = "contains_pii"
            recommended = _max_decision(recommended, Decision.REVIEW)
            confidence = min(confidence, 0.76)
            rationale_parts.append("The unit contains PII that can be handled with redaction.")

    if 0 < unit.confidence < 0.65:
        confidence = min(confidence, 0.55)
        requires_manual_review = True
        rationale_parts.append("The ASR confidence is low enough to keep the result uncertain.")

    if recommended in {Decision.REVIEW, Decision.QUARANTINE}:
        requires_manual_review = True

    if not rationale_parts:
        rationale_parts.append("No persistent risk remains after preliminary screening.")

    return AudioHardCaseJudgement(
        content_status=content_status,
        privacy_status=privacy_status,
        confidence=round(confidence, 4),
        rationale=" ".join(rationale_parts),
        recommended_decision=recommended,
        requires_manual_review=requires_manual_review,
        final_reasons=reasons,
    )


def run(
    transcript_units: list[TranscriptUnit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    settings: Settings | None = None,
    run_id: str = "",
) -> list[AudioHardCaseResult]:
    settings = settings or get_settings()
    if not settings.enable_hard_case_adjudication:
        return []

    privacy_by_unit = {item.unit_id: item for item in privacy_results}
    safety_by_unit = {item.unit_id: item for item in safety_results}
    results: list[AudioHardCaseResult] = []

    for unit in transcript_units:
        privacy = privacy_by_unit.get(unit.unit_id)
        safety = safety_by_unit.get(unit.unit_id)
        trigger_sources, trigger_reasons = _trigger_sources(unit, privacy, safety, settings)
        if not trigger_sources:
            continue

        prompt = qwen_hard_case_adapter.build_prompt(
            unit,
            privacy,
            safety,
            trigger_sources,
            trigger_reasons,
            settings,
        )
        adapter_result = qwen_hard_case_adapter.adjudicate(prompt, settings)
        judgement = adapter_result.judgement
        provider_name = "heuristic_fallback"
        is_degraded = True
        raw_response = adapter_result.raw_response
        notes = list(adapter_result.notes)

        if judgement is not None:
            provider_name = adapter_result.provider_name or "qwen_adapter"
            is_degraded = False
        else:
            judgement = _heuristic_judgement(unit, privacy, safety, trigger_reasons)
            raw_response = judgement.model_dump_json()
            notes.append("heuristic fallback used because no Qwen hard-case adjudicator was available.")

        results.append(
            AudioHardCaseResult(
                run_id=run_id,
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                trigger_sources=trigger_sources,
                trigger_reasons=trigger_reasons,
                model_name=settings.hard_case_model_name,
                provider_name=provider_name,
                prompt_version=settings.hard_case_prompt_version,
                adjudicated=True,
                is_degraded=is_degraded,
                uncertainty=round(1.0 - judgement.confidence, 4),
                judgement=judgement,
                raw_response=raw_response[:4000],
                notes=notes,
            )
        )

    logger.info("Audio hard-case adjudication completed: %d units", len(results))
    return results


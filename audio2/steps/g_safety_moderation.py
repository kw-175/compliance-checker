"""
Step G: semantic safety moderation.
"""

from __future__ import annotations

import logging

from audio.adapters import qwen_guard_adapter
from audio.config.settings import Settings
from audio.models.schemas import PrivacyResult, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

_UNSAFE = {"bomb", "kill", "terrorism", "attack", "exploit", "murder", "weapon"}
_CONTROVERSIAL = {"political", "protest", "censorship", "classified"}


def _mock_classify(text: str) -> tuple[SafetyLevel, list[str], str]:
    lower = text.lower()
    for keyword in _UNSAFE:
        if keyword in lower:
            return SafetyLevel.UNSAFE, ["violent_content"], f"mock matched {keyword}"
    for keyword in _CONTROVERSIAL:
        if keyword in lower:
            return SafetyLevel.CONTROVERSIAL, ["politically_sensitive"], f"mock matched {keyword}"
    return SafetyLevel.SAFE, [], "mock safe"


def run(results: list[PrivacyResult], settings: Settings | None = None) -> list[SafetyResult]:
    if settings is None:
        from audio.config.settings import get_settings

        settings = get_settings()

    use_model = settings.qwen_guard_enabled
    is_degraded = False
    if use_model:
        try:
            qwen_guard_adapter.load_model(settings)
        except Exception as exc:
            logger.warning("Qwen guard unavailable, fallback to mock moderation: %s", exc)
            use_model = False
            is_degraded = True

    output: list[SafetyResult] = []
    for result in results:
        text = result.redacted_text or result.original_text
        result_degraded = is_degraded
        provider_name = "keyword_fallback"
        model_version = ""

        if use_model:
            try:
                moderation = qwen_guard_adapter.moderate(text, settings)
                level = moderation.level
                categories = moderation.categories
                raw = moderation.raw_output
                provider_name = "qwen_guard"
                model_version = moderation.model_version
            except Exception as exc:
                logger.warning("Qwen guard failed, fallback to mock moderation: %s", exc)
                level, categories, raw = _mock_classify(text)
                result_degraded = True
        else:
            level, categories, raw = _mock_classify(text)

        score = 1.0 if level == SafetyLevel.SAFE else (0.5 if level == SafetyLevel.CONTROVERSIAL else 0.0)
        output.append(
            SafetyResult(
                unit_id=result.unit_id,
                source_id=result.source_id,
                safety_level=level,
                harm_categories=categories,
                raw_output=raw[:500],
                score=score,
                explanation=raw[:500],
                provider_name=provider_name,
                model_version=model_version,
                is_degraded=result_degraded,
            )
        )
    return output


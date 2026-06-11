"""
Step G: semantic safety moderation.
"""

from __future__ import annotations

import logging

from audio.adapters import qwen_guard_adapter
from audio.config.settings import Settings
from audio.models.schemas import PrivacyResult, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

_UNSAFE = {
    "bomb",
    "kill",
    "terrorism",
    "attack",
    "exploit",
    "murder",
    "weapon",
    "炸弹",
    "恐怖袭击",
    "带刀",
    "拿刀",
    "堵人",
    "打死",
    "杀了",
    "改成绩",
    "盗号",
}
_CONTROVERSIAL = {
    "political",
    "protest",
    "censorship",
    "classified",
    "隐瞒",
    "集合",
    "游行",
    "公共事件",
}
_CATEGORY_LABELS = {
    "violent_content": {"content.violent", "violent", "violence"},
    "politically_sensitive": {"content.political", "political", "politics"},
    "sexual_content": {"content.pornographic", "content.sexual", "pornographic", "sexual"},
    "hate_speech": {"content.hate", "hate"},
    "suicide_self_harm": {"content.self_harm", "self_harm"},
    "non_violent_illegal": {"content.illegal_instruction", "illegal_instruction"},
    "jailbreak_attempt": {"content.jailbreak", "jailbreak"},
}


def _selected_labels(labels: list[str] | None) -> set[str]:
    return {str(item).strip().lower() for item in (labels or []) if str(item).strip()}


def _category_allowed(category: str, labels: set[str]) -> bool:
    if not labels:
        return True
    aliases = _CATEGORY_LABELS.get(category, {category})
    return bool(labels & {str(item).lower() for item in aliases})


def _filter_categories(categories: list[str], labels: set[str]) -> list[str]:
    if not labels:
        return categories
    return [category for category in categories if _category_allowed(str(category).lower(), labels)]


def _mock_classify(text: str, target_labels: list[str] | None = None) -> tuple[SafetyLevel, list[str], str]:
    lower = text.lower()
    labels = _selected_labels(target_labels)
    if _category_allowed("violent_content", labels):
        for keyword in _UNSAFE:
            if keyword in lower:
                return SafetyLevel.UNSAFE, ["violent_content"], f"mock matched {keyword}"
    if _category_allowed("politically_sensitive", labels):
        for keyword in _CONTROVERSIAL:
            if keyword in lower:
                return SafetyLevel.CONTROVERSIAL, ["politically_sensitive"], f"mock matched {keyword}"
    if _category_allowed("sexual_content", labels):
        for keyword in {"sex", "porn", "explicit", "nsfw", "色情", "裸聊", "成人聊天群", "露骨"}:
            if keyword in lower:
                return SafetyLevel.UNSAFE, ["sexual_content"], f"mock matched {keyword}"
    if _category_allowed("suicide_self_harm", labels):
        for keyword in {"自杀", "自残", "轻生", "不想活", "伤害自己", "suicide", "self harm"}:
            if keyword in lower:
                return SafetyLevel.UNSAFE, ["suicide_self_harm"], f"mock matched {keyword}"
    if _category_allowed("hate_speech", labels):
        for keyword in {"歧视", "外地学生", "滚出学校", "racial hatred", "genocide"}:
            if keyword in lower:
                return SafetyLevel.UNSAFE, ["hate_speech"], f"mock matched {keyword}"
    return SafetyLevel.SAFE, [], "mock safe"


def _level_after_filter(level: SafetyLevel, categories: list[str]) -> SafetyLevel:
    if categories:
        return level
    return SafetyLevel.SAFE


def run(results: list[PrivacyResult], settings: Settings | None = None, target_labels: list[str] | None = None) -> list[SafetyResult]:
    if settings is None:
        from audio.config.settings import get_settings

        settings = get_settings()

    use_model = settings.qwen_guard_enabled
    is_degraded = False
    if use_model and not settings.qwen_guard_endpoint:
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
                if target_labels:
                    moderation = qwen_guard_adapter.moderate(text, settings, target_labels=target_labels)
                else:
                    moderation = qwen_guard_adapter.moderate(text, settings)
                categories = _filter_categories(moderation.categories, _selected_labels(target_labels))
                level = _level_after_filter(moderation.level, categories)
                raw = moderation.raw_output
                provider_name = "qwen_guard_endpoint" if settings.qwen_guard_endpoint else "qwen_guard"
                model_version = moderation.model_version
            except Exception as exc:
                logger.warning("Qwen guard failed, fallback to mock moderation: %s", exc)
                level, categories, raw = _mock_classify(text, target_labels=target_labels)
                result_degraded = True
        else:
            level, categories, raw = _mock_classify(text, target_labels=target_labels)

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

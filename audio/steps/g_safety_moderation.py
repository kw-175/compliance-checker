"""
Step G: semantic safety moderation.
"""

from __future__ import annotations

import logging

from audio.config.settings import Settings
from audio.models.schemas import PrivacyResult, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

_UNSAFE = {"bomb", "kill", "terrorism", "attack", "exploit", "murder", "weapon"}
_CONTROVERSIAL = {"political", "protest", "censorship", "classified"}


def _load_model(settings: Settings):
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = settings.qwen_guard_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(settings.qwen_guard_model, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    _model.eval()
    return _model, _tokenizer


def _parse_output(raw: str) -> tuple[SafetyLevel, list[str]]:
    raw_lower = raw.lower()
    if "unsafe" in raw_lower:
        level = SafetyLevel.UNSAFE
    elif "controversial" in raw_lower:
        level = SafetyLevel.CONTROVERSIAL
    else:
        level = SafetyLevel.SAFE

    categories = []
    for category in [
        "violent_content",
        "non_violent_illegal",
        "sexual_content",
        "pii_exposure",
        "suicide_self_harm",
        "unethical_acts",
        "politically_sensitive",
        "copyright_violation",
        "jailbreak_attempt",
        "hate_speech",
        "discrimination",
    ]:
        if category in raw_lower:
            categories.append(category)
    return level, categories


def _classify_with_model(text: str, model, tokenizer, device: str) -> tuple[SafetyLevel, list[str], str]:
    import torch

    prompt = (
        "Classify the text as Safe, Controversial, or Unsafe. "
        "Also list harm categories from: violent_content, non_violent_illegal, sexual_content, "
        "pii_exposure, suicide_self_harm, unethical_acts, politically_sensitive, copyright_violation, "
        "jailbreak_attempt, hate_speech, discrimination.\n"
        "Respond as: Safety: <Safe|Controversial|Unsafe>; Categories: <comma-separated or none>.\n"
        f"Text: {text[:2000]}"
    )
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )
    completion = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(completion, skip_special_tokens=True)
    level, categories = _parse_output(raw)
    return level, categories, raw


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
    model = tokenizer = None
    device = settings.qwen_guard_device
    if use_model:
        try:
            model, tokenizer = _load_model(settings)
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            logger.warning("Qwen guard unavailable, fallback to mock moderation: %s", exc)
            use_model = False

    output: list[SafetyResult] = []
    for result in results:
        text = result.redacted_text or result.original_text
        if use_model and model is not None and tokenizer is not None:
            level, categories, raw = _classify_with_model(text, model, tokenizer, device)
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
            )
        )
    return output

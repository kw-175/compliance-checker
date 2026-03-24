"""
Step G – Semantic Safety Moderation

Uses Qwen3Guard (HuggingFace transformers) for tri-class safety classification
of redacted text:  Safe / Controversial / Unsafe

Also extracts specific harm categories from the model output.

Output → safety_checked.jsonl
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import PrivacyResult, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

# Module-level singleton
_model = None
_tokenizer = None


# ────────────────────────────────────────────────────────────
# Qwen3Guard loader
# ────────────────────────────────────────────────────────────

def _load_model(settings: Settings):
    """Lazy-load the Qwen3Guard model and tokenizer."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    device = settings.qwen_guard_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading Qwen3Guard model '%s' on device '%s'...",
        settings.qwen_guard_model, device,
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(device)
    _model.eval()

    logger.info("Qwen3Guard model loaded successfully")
    return _model, _tokenizer


# ────────────────────────────────────────────────────────────
# Safety classification
# ────────────────────────────────────────────────────────────

_SAFETY_PROMPT_TEMPLATE = """You are a content safety classifier. Classify the following text into one of three categories: Safe, Controversial, or Unsafe.

Also identify any specific harm categories present from this list:
- violent_content
- non_violent_illegal
- sexual_content
- pii_exposure
- suicide_self_harm
- unethical_acts
- politically_sensitive
- copyright_violation
- jailbreak_attempt
- hate_speech
- discrimination

Text to classify:
---
{text}
---

Respond in this exact format:
Safety: <Safe|Controversial|Unsafe>
Categories: <comma-separated list or "none">
"""

_HARM_CATEGORIES = {
    "violent_content", "non_violent_illegal", "sexual_content",
    "pii_exposure", "suicide_self_harm", "unethical_acts",
    "politically_sensitive", "copyright_violation", "jailbreak_attempt",
    "hate_speech", "discrimination",
}


def _parse_safety_output(raw_output: str) -> tuple[SafetyLevel, list[str]]:
    """Parse the model output to extract safety level and harm categories."""
    raw_lower = raw_output.lower()

    # Extract safety level
    safety_level = SafetyLevel.SAFE
    if "unsafe" in raw_lower:
        safety_level = SafetyLevel.UNSAFE
    elif "controversial" in raw_lower:
        safety_level = SafetyLevel.CONTROVERSIAL

    # Extract harm categories
    categories: list[str] = []
    for cat in _HARM_CATEGORIES:
        if cat.lower() in raw_lower:
            categories.append(cat)

    return safety_level, categories


def _classify_text(
    text: str,
    model,
    tokenizer,
    device: str,
    max_input_len: int = 2048,
) -> tuple[SafetyLevel, list[str], str]:
    """Run a single text through Qwen3Guard."""
    import torch

    prompt = _SAFETY_PROMPT_TEMPLATE.format(text=text[:max_input_len])

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
        )

    # Decode only the new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    safety_level, categories = _parse_safety_output(raw_output)
    return safety_level, categories, raw_output


# ────────────────────────────────────────────────────────────
# Mock fallback (no GPU / no model)
# ────────────────────────────────────────────────────────────

# Simple keyword heuristic for mock safety scoring
_UNSAFE_KEYWORDS = {
    "bomb", "kill", "murder", "terrorism", "exploit", "hack",
    "drug", "cocaine", "heroin", "meth",
    "炸弹", "杀人", "恐怖", "毒品",
}

_CONTROVERSIAL_KEYWORDS = {
    "political", "protest", "rebellion", "censorship",
    "政治", "抗议", "审查",
}


def _mock_classify(text: str) -> tuple[SafetyLevel, list[str], str]:
    """Simple keyword-based mock classifier."""
    text_lower = text.lower()
    categories: list[str] = []

    for kw in _UNSAFE_KEYWORDS:
        if kw in text_lower:
            return SafetyLevel.UNSAFE, ["violent_content"], f"mock: matched '{kw}'"

    for kw in _CONTROVERSIAL_KEYWORDS:
        if kw in text_lower:
            return SafetyLevel.CONTROVERSIAL, ["politically_sensitive"], f"mock: matched '{kw}'"

    return SafetyLevel.SAFE, [], "mock: no risk keywords found"


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def run(
    privacy_results: list[PrivacyResult],
    settings: Settings | None = None,
) -> list[SafetyResult]:
    """
    Execute semantic safety moderation on redacted texts.

    Parameters
    ----------
    privacy_results : list[PrivacyResult]
        Output from Step F containing redacted_text.
    settings : Settings, optional

    Returns
    -------
    list[SafetyResult]
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    use_model = settings.qwen_guard_enabled
    model = tokenizer = device = None

    if use_model:
        try:
            model, tokenizer = _load_model(settings)
            device = settings.qwen_guard_device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            logger.warning(
                "Cannot load Qwen3Guard (%s); falling back to mock classifier", e
            )
            use_model = False

    results: list[SafetyResult] = []
    for pr in privacy_results:
        text = pr.redacted_text or pr.original_text

        if use_model and model is not None:
            safety_level, categories, raw = _classify_text(
                text, model, tokenizer, device
            )
        else:
            safety_level, categories, raw = _mock_classify(text)

        score = 1.0 if safety_level == SafetyLevel.SAFE else (
            0.5 if safety_level == SafetyLevel.CONTROVERSIAL else 0.0
        )

        results.append(
            SafetyResult(
                doc_id=pr.doc_id,
                safety_level=safety_level,
                harm_categories=categories,
                raw_output=raw[:500],
                score=score,
            )
        )
        if safety_level != SafetyLevel.SAFE:
            logger.debug(
                "Doc %s: safety=%s categories=%s",
                pr.doc_id, safety_level.value, categories,
            )

    unsafe_count = sum(1 for r in results if r.safety_level == SafetyLevel.UNSAFE)
    controv_count = sum(1 for r in results if r.safety_level == SafetyLevel.CONTROVERSIAL)
    logger.info(
        "Safety moderation complete: %d documents (%d unsafe, %d controversial)",
        len(results), unsafe_count, controv_count,
    )
    return results

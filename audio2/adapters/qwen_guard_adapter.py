"""
Qwen Guard adapter.

The step layer should not need to know the model's prompt format or raw output
shape. It asks this adapter for a normalized moderation result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock

from audio.config.settings import Settings
from audio.models.schemas import SafetyLevel

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None
_cache_key: tuple[str, str] | None = None
_lock = Lock()


@dataclass(frozen=True)
class QwenGuardResult:
    level: SafetyLevel
    categories: list[str]
    raw_output: str
    model_version: str


def resolve_device(settings: Settings) -> str:
    requested = str(getattr(settings, "qwen_guard_device", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(settings: Settings, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading Qwen Guard model: %s on %s", settings.qwen_guard_model, device)
    tokenizer = AutoTokenizer.from_pretrained(settings.qwen_guard_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,
        dtype="auto",
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def load_model(settings: Settings):
    global _cache_key, _model, _tokenizer

    device = resolve_device(settings)
    cache_key = (settings.qwen_guard_model, device)
    if _model is not None and _tokenizer is not None and _cache_key == cache_key:
        return _model, _tokenizer, device

    with _lock:
        if _model is not None and _tokenizer is not None and _cache_key == cache_key:
            return _model, _tokenizer, device
        _model, _tokenizer = build_model(settings, device)
        _cache_key = cache_key
        return _model, _tokenizer, device


def reset_cache() -> None:
    global _cache_key, _model, _tokenizer

    with _lock:
        _model = None
        _tokenizer = None
        _cache_key = None


def parse_output(raw: str) -> tuple[SafetyLevel, list[str]]:
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


def moderate(text: str, settings: Settings) -> QwenGuardResult:
    import torch

    model, tokenizer, device = load_model(settings)
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
        )
    completion = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(completion, skip_special_tokens=True)
    level, categories = parse_output(raw)
    return QwenGuardResult(
        level=level,
        categories=categories,
        raw_output=raw,
        model_version=settings.qwen_guard_model,
    )

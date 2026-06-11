"""
Local PII detection engine for audio transcripts.

This module intentionally lives inside audio so the audio service can run
without importing or calling the text service. It reuses shared model files via
configuration paths only.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from inspect import signature
from pathlib import Path
from typing import Any

from audio.config.settings import Settings
from audio.models.schemas import PIIEntity

logger = logging.getLogger(__name__)

ZH_CONTEXT = {
    "id_card": ["\u8eab\u4efd\u8bc1", "\u8eab\u4efd\u8bc1\u53f7", "\u516c\u6c11\u8eab\u4efd\u53f7\u7801", "\u8bc1\u4ef6\u53f7"],
    "phone": ["\u624b\u673a", "\u7535\u8bdd", "\u8054\u7cfb\u65b9\u5f0f", "\u8054\u7cfb\u7535\u8bdd", "\u5bb6\u957f\u7535\u8bdd"],
    "student_id": ["\u5b66\u53f7", "\u5b66\u751f\u7f16\u53f7", "\u51c6\u8003\u8bc1\u53f7", "\u8003\u53f7"],
    "address": ["\u5730\u5740", "\u4f4f\u5740", "\u5bb6\u5ead\u4f4f\u5740", "\u5bdd\u5ba4", "\u5bbf\u820d"],
    "bank_card": ["\u94f6\u884c\u5361", "\u5361\u53f7", "\u8d26\u53f7", "\u6536\u6b3e\u8d26\u6237"],
    "wechat": ["\u5fae\u4fe1", "\u5fae\u4fe1\u53f7", "wechat"],
    "qq": ["qq", "QQ", "\u817e\u8bafQQ"],
    "name": ["\u59d3\u540d", "\u540d\u5b57", "\u5b66\u751f", "\u5bb6\u957f", "\u8054\u7cfb\u4eba"],
}

GLINER_LABEL_TO_ENTITY = {
    "person": "PERSON",
    "person name": "PERSON",
    "name": "PERSON",
    "organization": "ORGANIZATION",
    "address": "LOCATION",
    "location": "LOCATION",
    "email": "EMAIL_ADDRESS",
    "email address": "EMAIL_ADDRESS",
    "phone number": "PHONE_NUMBER",
    "mobile phone number": "PHONE_NUMBER",
    "id number": "ID_CARD",
    "national id number": "ID_CARD",
    "identity card number": "ID_CARD",
    "passport number": "PASSPORT",
    "driver license number": "DRIVER_LICENSE",
    "credit card number": "CREDIT_CARD",
    "bank card number": "CREDIT_CARD",
    "bank account number": "BANK_ACCOUNT",
    "student id": "STUDENT_ID",
    "student number": "STUDENT_ID",
    "wechat id": "WECHAT_ID",
    "qq number": "QQ_NUMBER",
    "license plate": "LICENSE_PLATE",
    "ip address": "IP_ADDRESS",
    "url": "URL",
}


@dataclass
class LocalPIIDetection:
    entities: list[PIIEntity]
    provider_name: str
    provider_version: str
    is_degraded: bool = False


def _normalize_language(language: str) -> str:
    normalized = (language or "").strip().lower().replace("_", "-")
    if normalized in {"zh", "zh-cn", "zh-hans", "chinese", "cn"}:
        return "zh"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return ""


def _languages_for(language: str, configured: list[str]) -> list[str]:
    normalized = _normalize_language(language)
    if normalized:
        return [normalized]
    output: list[str] = []
    for item in configured or ["en", "zh"]:
        value = _normalize_language(item)
        if value and value not in output:
            output.append(value)
    return output or ["en"]


def _window(text: str, start: int, end: int, size: int = 24) -> str:
    return text[max(0, start - size): min(len(text), end + size)]


def _has_context(text: str, start: int, end: int, terms: list[str]) -> bool:
    around = _window(text, start, end).lower()
    return any(term.lower() in around for term in terms)


def _entity(entity_type: str, text: str, start: int, end: int, score: float) -> PIIEntity:
    return PIIEntity(
        entity_type=entity_type,
        start=start,
        end=end,
        score=round(max(0.0, min(1.0, score)), 4),
        original_text=text[start:end][:100],
    )


def _is_valid_cn_id_card(value: str) -> bool:
    value = value.strip()
    pattern = r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"
    if not re.fullmatch(pattern, value):
        return False
    try:
        date(int(value[6:10]), int(value[10:12]), int(value[12:14]))
    except ValueError:
        return False

    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    checks = "10X98765432"
    checksum = sum(int(digit) * weight for digit, weight in zip(value[:17], weights, strict=True)) % 11
    return checks[checksum] == value[-1].upper()


def _luhn_valid(value: str) -> bool:
    digits = [int(char) for char in re.sub(r"\D", "", value)]
    if not 12 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _regex_detect(text: str) -> list[PIIEntity]:
    entities: list[PIIEntity] = []

    for match in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        entities.append(_entity("EMAIL_ADDRESS", text, match.start(), match.end(), 0.95))

    for match in re.finditer(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)", text):
        score = 0.93 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["phone"]) else 0.82
        entities.append(_entity("CN_PHONE_NUMBER", text, match.start(), match.end(), score))

    us_phone = r"(?:\+?1[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}"
    for match in re.finditer(us_phone, text):
        entities.append(_entity("PHONE_NUMBER", text, match.start(), match.end(), 0.82))

    for match in re.finditer(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)", text):
        if _has_context(text, match.start(), match.end(), ZH_CONTEXT["phone"]):
            entities.append(_entity("PHONE_NUMBER", text, match.start(), match.end(), 0.78))

    cn_id = r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
    for match in re.finditer(cn_id, text):
        if _is_valid_cn_id_card(match.group(0)):
            score = 0.98 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["id_card"]) else 0.94
            entities.append(_entity("CN_ID_CARD", text, match.start(), match.end(), score))

    for match in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text):
        entities.append(_entity("US_SSN", text, match.start(), match.end(), 0.9))

    for match in re.finditer(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", text):
        if _luhn_valid(match.group(0)):
            score = 0.95 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["bank_card"]) else 0.86
            entities.append(_entity("CREDIT_CARD", text, match.start(), match.end(), score))

    contextual_patterns = [
        (
            "STUDENT_ID",
            r"(?:\u5b66\u53f7|\u5b66\u751f\u7f16\u53f7|\u51c6\u8003\u8bc1\u53f7|\u8003\u53f7|student\s*id)[:\uff1a\s-]*([A-Za-z0-9_-]{4,32})",
            0.86,
        ),
        (
            "WECHAT_ID",
            r"(?:\u5fae\u4fe1\u53f7?|\bwechat\b)[:\uff1a\s-]*([A-Za-z][A-Za-z0-9_-]{5,19})",
            0.82,
        ),
        (
            "QQ_NUMBER",
            r"(?:\bQQ\b|\u817e\u8bafQQ)[:\uff1a\s-]*([1-9]\d{4,11})",
            0.82,
        ),
        (
            "PARENT_CONTACT",
            r"(?:\u5bb6\u957f(?:\u7535\u8bdd|\u624b\u673a|\u8054\u7cfb\u65b9\u5f0f)|parent(?:\s+contact|\s+phone)?)[:\uff1a\s-]*([+0-9A-Za-z@._ -]{5,40})",
            0.84,
        ),
    ]
    for entity_type, pattern, score in contextual_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span(1)
            entities.append(_entity(entity_type, text, start, end, score))

    for match in re.finditer(r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}", text):
        entities.append(_entity("LICENSE_PLATE", text, match.start(), match.end(), 0.82))

    address_pattern = (
        r"[\u4e00-\u9fa5]{2,}(?:\u7701|\u5e02|\u81ea\u6cbb\u533a)"
        r"[\u4e00-\u9fa5]{0,20}(?:\u533a|\u53bf|\u9547|\u4e61|\u8857\u9053)"
        r"[\u4e00-\u9fa5A-Za-z0-9\-]{0,40}(?:\u8def|\u8857|\u5df7|\u5f04|\u53f7|\u5c0f\u533a|\u5bbf\u820d|\u697c|\u5ba4)"
    )
    for match in re.finditer(address_pattern, text):
        entities.append(_entity("LOCATION", text, match.start(), match.end(), 0.8))

    return entities


def _configure_stanza_resources_dir(stanza_resources_dir: str) -> None:
    if not stanza_resources_dir:
        return
    os.environ["STANZA_RESOURCES_DIR"] = stanza_resources_dir

    try:
        import stanza
        from stanza.pipeline import core as stanza_core
        from stanza.resources import common as stanza_common
    except Exception as exc:
        logger.debug("Unable to configure Stanza resources dir: %s", exc)
        return

    stanza_common.DEFAULT_MODEL_DIR = stanza_resources_dir
    stanza_core.DEFAULT_MODEL_DIR = stanza_resources_dir

    original_pipeline = getattr(stanza, "Pipeline", None)
    if original_pipeline is None or getattr(original_pipeline, "_audio_compliance_dir_wrapped", False):
        return

    def pipeline_with_local_dir(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("dir", stanza_resources_dir)
        return original_pipeline(*args, **kwargs)

    pipeline_with_local_dir._audio_compliance_dir_wrapped = True  # type: ignore[attr-defined]
    stanza.Pipeline = pipeline_with_local_dir

    try:
        import presidio_analyzer.nlp_engine.stanza_nlp_engine as stanza_engine_module

        if hasattr(stanza_engine_module, "stanza"):
            stanza_engine_module.stanza.Pipeline = pipeline_with_local_dir
        if hasattr(stanza_engine_module, "Pipeline"):
            stanza_engine_module.Pipeline = pipeline_with_local_dir
    except Exception as exc:
        logger.debug("Presidio Stanza module patch not applied: %s", exc)


def _zh_pattern_recognizers():
    from presidio_analyzer import Pattern, PatternRecognizer

    return [
        PatternRecognizer(
            supported_entity="CN_ID_CARD",
            supported_language="zh",
            patterns=[Pattern(name="cn_id_card_candidate", regex=r"(?<!\d)[1-9]\d{16}[\dXx](?!\d)", score=0.65)],
            context=ZH_CONTEXT["id_card"],
        ),
        PatternRecognizer(
            supported_entity="CN_PHONE_NUMBER",
            supported_language="zh",
            patterns=[Pattern(name="cn_mobile_phone", regex=r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)", score=0.75)],
            context=ZH_CONTEXT["phone"],
        ),
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            supported_language="zh",
            patterns=[Pattern(name="email", regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", score=0.9)],
            context=["email", "\u90ae\u7bb1", "\u7535\u5b50\u90ae\u4ef6"],
        ),
    ]


@lru_cache(maxsize=8)
def _build_presidio_engine(
    stanza_resources_dir: str,
    en_model: str,
    zh_model: str,
    download_if_missing: bool,
    languages: tuple[str, ...],
):
    if stanza_resources_dir and not Path(stanza_resources_dir).exists() and not download_if_missing:
        raise FileNotFoundError(f"Stanza resources directory not found: {stanza_resources_dir}")

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NerModelConfiguration, StanzaNlpEngine

    _configure_stanza_resources_dir(stanza_resources_dir)
    stanza_engine_params = signature(StanzaNlpEngine.__init__).parameters
    model_config = []
    if "en" in languages:
        model_config.append({"lang_code": "en", "model_name": en_model})
    if "zh" in languages:
        model_config.append({"lang_code": "zh", "model_name": zh_model})

    entity_mapping = {
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
    }
    nlp_engine_kwargs: dict[str, Any] = {
        "models": model_config,
        "ner_model_configuration": NerModelConfiguration(model_to_presidio_entity_mapping=entity_mapping),
    }
    if "download_if_missing" in stanza_engine_params:
        nlp_engine_kwargs["download_if_missing"] = download_if_missing
    if stanza_resources_dir:
        if "dir" in stanza_engine_params:
            nlp_engine_kwargs["dir"] = stanza_resources_dir
        elif "model_dir" in stanza_engine_params:
            nlp_engine_kwargs["model_dir"] = stanza_resources_dir

    registry = RecognizerRegistry(supported_languages=list(languages))
    try:
        registry.load_predefined_recognizers(languages=["en"])
    except TypeError:
        registry.load_predefined_recognizers()
    for recognizer in _zh_pattern_recognizers():
        if "zh" in languages:
            registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=StanzaNlpEngine(**nlp_engine_kwargs),
        supported_languages=list(languages),
    )


def _run_presidio(text: str, settings: Settings, languages: list[str]) -> tuple[list[PIIEntity], bool]:
    if not settings.pii_enable_presidio:
        return [], False
    try:
        engine = _build_presidio_engine(
            str(settings.pii_stanza_resources_dir),
            settings.pii_stanza_en_model,
            settings.pii_stanza_zh_model,
            settings.pii_stanza_download_if_missing,
            tuple(languages),
        )
    except Exception as exc:
        logger.warning("Presidio Stanza engine unavailable, using remaining PII detectors: %s", exc)
        return [], True

    entities: list[PIIEntity] = []
    degraded = False
    for language in languages:
        try:
            rows = engine.analyze(
                text=text,
                language=language,
                score_threshold=settings.pii_score_threshold,
            )
        except Exception as exc:
            logger.debug("Presidio language pass failed (%s): %s", language, exc)
            degraded = True
            continue
        for item in rows:
            if 0 <= item.start < item.end <= len(text):
                entities.append(_entity(item.entity_type, text, int(item.start), int(item.end), float(item.score)))
    return entities, degraded


@lru_cache(maxsize=4)
def _load_gliner_model(model_name: str):
    if os.path.isabs(model_name) and not Path(model_name).exists():
        logger.warning("GLiNER model path not found, using remaining PII detectors: %s", model_name)
        return None
    try:
        from gliner import GLiNER

        logger.info("Loading GLiNER PII model: %s", model_name)
        return GLiNER.from_pretrained(model_name)
    except Exception as exc:
        logger.warning("GLiNER model unavailable, using remaining PII detectors: %s", exc)
        return None


def _gliner_labels(settings: Settings) -> list[str]:
    if settings.pii_gliner_labels.strip():
        return [label.strip() for label in settings.pii_gliner_labels.split(",") if label.strip()]
    return list(GLINER_LABEL_TO_ENTITY)


def _run_gliner(text: str, settings: Settings) -> tuple[list[PIIEntity], bool]:
    if not settings.pii_enable_gliner:
        return [], False
    model = _load_gliner_model(settings.pii_gliner_model)
    if model is None:
        return [], True

    clipped = text[: max(0, settings.pii_gliner_max_chars)]
    try:
        raw_entities = model.predict_entities(clipped, _gliner_labels(settings), threshold=settings.pii_gliner_threshold)
    except Exception as exc:
        logger.warning("GLiNER analysis failed, using remaining PII detectors: %s", exc)
        return [], True

    entities: list[PIIEntity] = []
    for item in raw_entities:
        label = str(item.get("label", "")).strip().lower()
        entity_type = GLINER_LABEL_TO_ENTITY.get(label)
        if not entity_type:
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
            score = float(item.get("score", settings.pii_gliner_threshold))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(clipped):
            entities.append(_entity(entity_type, text, start, end, score))
    return entities, False


def _overlaps(left: PIIEntity, right: PIIEntity) -> bool:
    return left.start < right.end and left.end > right.start


def _deduplicate(entities: list[PIIEntity]) -> list[PIIEntity]:
    ordered = sorted(entities, key=lambda item: (-item.score, item.start, -(item.end - item.start)))
    accepted: list[PIIEntity] = []
    for entity in ordered:
        if not any(_overlaps(entity, existing) for existing in accepted):
            accepted.append(entity)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.entity_type))


def detect(text: str, settings: Settings, language: str = "") -> LocalPIIDetection:
    """Detect PII entities in one transcript unit."""

    if not text:
        return LocalPIIDetection([], "regex+presidio-stanza+gliner", "", False)

    languages = _languages_for(language, settings.presidio_languages)
    providers: list[str] = []
    entities: list[PIIEntity] = []
    degraded = False

    if settings.pii_enable_regex_rules:
        providers.append("regex")
        entities.extend(_regex_detect(text))

    if settings.pii_enable_presidio:
        providers.append("presidio-stanza")
        presidio_entities, presidio_degraded = _run_presidio(text, settings, languages)
        entities.extend(presidio_entities)
        degraded = degraded or presidio_degraded

    if settings.pii_enable_gliner:
        providers.append("gliner")
        gliner_entities, gliner_degraded = _run_gliner(text, settings)
        entities.extend(gliner_entities)
        degraded = degraded or gliner_degraded

    threshold = max(0.0, settings.pii_score_threshold)
    filtered = [entity for entity in entities if entity.score >= threshold]
    provider_name = "+".join(providers) if providers else "none"
    provider_version = (
        f"stanza={settings.pii_stanza_resources_dir};"
        f"gliner={settings.pii_gliner_model};"
        f"languages={','.join(languages)}"
    )
    return LocalPIIDetection(_deduplicate(filtered), provider_name, provider_version, degraded)

from __future__ import annotations

import logging
import os
import re
from inspect import signature
from datetime import date
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NerModelConfiguration, StanzaNlpEngine

logger = logging.getLogger("pii_gateway")

app = FastAPI(title="Presidio-compatible PII Gateway")


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "en"
    score_threshold: float | None = None
    entities: list[str] | None = None
    context: list[str] | None = None
    return_decision_process: bool = False


class PiiCandidate(BaseModel):
    entity_type: str
    start: int
    end: int
    score: float
    recognizer: str
    score_source: str
    text: str | None = None
    metadata: dict[str, Any] | None = None


ZH_CONTEXT = {
    "id_card": [
        "\u8eab\u4efd\u8bc1",
        "\u8eab\u4efd\u8bc1\u53f7",
        "\u516c\u6c11\u8eab\u4efd\u53f7\u7801",
        "\u8bc1\u4ef6\u53f7",
    ],
    "phone": [
        "\u624b\u673a",
        "\u7535\u8bdd",
        "\u8054\u7cfb\u65b9\u5f0f",
        "\u8054\u7cfb\u7535\u8bdd",
        "\u5bb6\u957f\u7535\u8bdd",
    ],
    "student_id": [
        "\u5b66\u53f7",
        "\u5b66\u751f\u7f16\u53f7",
        "\u51c6\u8003\u8bc1\u53f7",
        "\u8003\u53f7",
    ],
    "address": [
        "\u5730\u5740",
        "\u4f4f\u5740",
        "\u5bb6\u5ead\u4f4f\u5740",
        "\u5bdd\u5ba4",
        "\u5bbf\u820d",
    ],
    "bank_card": [
        "\u94f6\u884c\u5361",
        "\u5361\u53f7",
        "\u8d26\u53f7",
        "\u6536\u6b3e\u8d26\u6237",
    ],
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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using %s", name, raw, default)
        return default


def _normalize_language(language: str) -> str:
    normalized = (language or "").strip().lower().replace("_", "-")
    if normalized in {"zh", "zh-cn", "zh-hans", "chinese", "cn"}:
        return "zh"
    if normalized in {"en", "en-us", "en-gb", "english"}:
        return "en"
    return "en"


def _configure_stanza_resources_dir(stanza_resources_dir: str) -> None:
    if not stanza_resources_dir:
        return

    os.environ["STANZA_RESOURCES_DIR"] = stanza_resources_dir

    try:
        import stanza
        from stanza.pipeline import core as stanza_core
        from stanza.resources import common as stanza_common
    except Exception as exc:
        logger.warning("Unable to configure Stanza resources dir: %s", exc)
        return

    stanza_common.DEFAULT_MODEL_DIR = stanza_resources_dir
    stanza_core.DEFAULT_MODEL_DIR = stanza_resources_dir

    original_pipeline = getattr(stanza, "Pipeline", None)
    if original_pipeline is None or getattr(original_pipeline, "_compliance_dir_wrapped", False):
        return

    def pipeline_with_local_dir(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("dir", stanza_resources_dir)
        return original_pipeline(*args, **kwargs)

    pipeline_with_local_dir._compliance_dir_wrapped = True  # type: ignore[attr-defined]
    stanza.Pipeline = pipeline_with_local_dir

    try:
        import presidio_analyzer.nlp_engine.stanza_nlp_engine as stanza_engine_module

        if hasattr(stanza_engine_module, "stanza"):
            stanza_engine_module.stanza.Pipeline = pipeline_with_local_dir
        if hasattr(stanza_engine_module, "Pipeline"):
            stanza_engine_module.Pipeline = pipeline_with_local_dir
    except Exception as exc:
        logger.info("Presidio Stanza module patch not applied: %s", exc)


def _window(text: str, start: int, end: int, size: int = 24) -> str:
    return text[max(0, start - size): min(len(text), end + size)]


def _has_context(text: str, start: int, end: int, terms: list[str]) -> bool:
    around = _window(text, start, end).lower()
    return any(term.lower() in around for term in terms)


def _candidate(
    *,
    entity_type: str,
    text: str,
    start: int,
    end: int,
    score: float,
    recognizer: str,
    score_source: str,
    metadata: dict[str, Any] | None = None,
) -> PiiCandidate:
    return PiiCandidate(
        entity_type=entity_type,
        start=start,
        end=end,
        score=round(max(0.0, min(1.0, score)), 4),
        recognizer=recognizer,
        score_source=score_source,
        text=text[start:end],
        metadata=metadata or {},
    )


def _is_valid_cn_id_card(value: str) -> bool:
    value = value.strip()
    if not re.fullmatch(r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]", value):
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


def _regex_candidates(text: str) -> list[PiiCandidate]:
    candidates: list[PiiCandidate] = []

    for match in re.finditer(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
        candidates.append(
            _candidate(
                entity_type="EMAIL_ADDRESS",
                text=text,
                start=match.start(),
                end=match.end(),
                score=0.95,
                recognizer="regex.email",
                score_source="regex",
            )
        )

    for match in re.finditer(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)", text):
        score = 0.93 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["phone"]) else 0.82
        candidates.append(
            _candidate(
                entity_type="CN_PHONE_NUMBER",
                text=text,
                start=match.start(),
                end=match.end(),
                score=score,
                recognizer="regex.cn_mobile_phone",
                score_source="regex_context",
            )
        )

    for match in re.finditer(r"(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)", text):
        if not _has_context(text, match.start(), match.end(), ZH_CONTEXT["phone"]):
            continue
        candidates.append(
            _candidate(
                entity_type="PHONE_NUMBER",
                text=text,
                start=match.start(),
                end=match.end(),
                score=0.78,
                recognizer="regex.cn_landline_phone",
                score_source="regex_context",
            )
        )

    for match in re.finditer(r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)", text):
        if not _is_valid_cn_id_card(match.group(0)):
            continue
        score = 0.98 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["id_card"]) else 0.94
        candidates.append(
            _candidate(
                entity_type="CN_ID_CARD",
                text=text,
                start=match.start(),
                end=match.end(),
                score=score,
                recognizer="regex.cn_id_card_checksum",
                score_source="regex_checksum_context",
            )
        )

    for match in re.finditer(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)", text):
        if not _luhn_valid(match.group(0)):
            continue
        score = 0.95 if _has_context(text, match.start(), match.end(), ZH_CONTEXT["bank_card"]) else 0.86
        candidates.append(
            _candidate(
                entity_type="CREDIT_CARD",
                text=text,
                start=match.start(),
                end=match.end(),
                score=score,
                recognizer="regex.bank_card_luhn",
                score_source="regex_checksum_context",
            )
        )

    contextual_patterns = [
        (
            "STUDENT_ID",
            r"(?:\u5b66\u53f7|\u5b66\u751f\u7f16\u53f7|\u51c6\u8003\u8bc1\u53f7|\u8003\u53f7|student\s*id)[:\uff1a\s-]*([A-Za-z0-9_-]{4,32})",
            "regex.student_id_context",
            0.86,
        ),
        (
            "WECHAT_ID",
            r"(?:\u5fae\u4fe1\u53f7?|\bwechat\b)[:\uff1a\s-]*([A-Za-z][A-Za-z0-9_-]{5,19})",
            "regex.wechat_context",
            0.82,
        ),
        (
            "QQ_NUMBER",
            r"(?:\bQQ\b|\u817e\u8bafQQ)[:\uff1a\s-]*([1-9]\d{4,11})",
            "regex.qq_context",
            0.82,
        ),
        (
            "PARENT_CONTACT",
            r"(?:\u5bb6\u957f(?:\u7535\u8bdd|\u624b\u673a|\u8054\u7cfb\u65b9\u5f0f)|parent(?:\s+contact|\s+phone)?)[:\uff1a\s-]*([+0-9A-Za-z@._ -]{5,40})",
            "regex.parent_contact_context",
            0.84,
        ),
    ]
    for entity_type, pattern, recognizer, score in contextual_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span(1)
            candidates.append(
                _candidate(
                    entity_type=entity_type,
                    text=text,
                    start=start,
                    end=end,
                    score=score,
                    recognizer=recognizer,
                    score_source="regex_context",
                )
            )

    for match in re.finditer(r"[\u4e00-\u9fa5][A-Z][A-Z0-9]{5,6}", text):
        candidates.append(
            _candidate(
                entity_type="LICENSE_PLATE",
                text=text,
                start=match.start(),
                end=match.end(),
                score=0.82,
                recognizer="regex.cn_license_plate",
                score_source="regex",
            )
        )

    address_pattern = (
        r"[\u4e00-\u9fa5]{2,}(?:\u7701|\u5e02|\u81ea\u6cbb\u533a)"
        r"[\u4e00-\u9fa5]{0,20}(?:\u533a|\u53bf|\u9547|\u4e61|\u8857\u9053)"
        r"[\u4e00-\u9fa5A-Za-z0-9\-]{0,40}(?:\u8def|\u8857|\u5df7|\u5f04|\u53f7|\u5c0f\u533a|\u5bbf\u820d|\u697c|\u5ba4)"
    )
    for match in re.finditer(address_pattern, text):
        candidates.append(
            _candidate(
                entity_type="LOCATION",
                text=text,
                start=match.start(),
                end=match.end(),
                score=0.80,
                recognizer="regex.zh_address_structure",
                score_source="regex_context",
            )
        )

    return candidates


def _zh_pattern_recognizers() -> list[PatternRecognizer]:
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


@lru_cache(maxsize=1)
def _engine() -> AnalyzerEngine:
    stanza_resources_dir = os.getenv("STANZA_RESOURCES_DIR", "").strip()
    _configure_stanza_resources_dir(stanza_resources_dir)
    stanza_engine_params = signature(StanzaNlpEngine.__init__).parameters

    model_config = [
        {"lang_code": "en", "model_name": os.getenv("PII_STANZA_EN_MODEL", "en")},
        {"lang_code": "zh", "model_name": os.getenv("PII_STANZA_ZH_MODEL", "zh")},
    ]
    entity_mapping = {
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
    }
    ner_config = NerModelConfiguration(model_to_presidio_entity_mapping=entity_mapping)
    nlp_engine_kwargs = {
        "models": model_config,
        "ner_model_configuration": ner_config,
    }
    if "download_if_missing" in stanza_engine_params:
        nlp_engine_kwargs["download_if_missing"] = _env_bool("PII_STANZA_DOWNLOAD_IF_MISSING", False)
    if stanza_resources_dir:
        if "dir" in stanza_engine_params:
            nlp_engine_kwargs["dir"] = stanza_resources_dir
        elif "model_dir" in stanza_engine_params:
            nlp_engine_kwargs["model_dir"] = stanza_resources_dir
        else:
            logger.info(
                "StanzaNlpEngine does not expose a model directory argument; "
                "using STANZA_RESOURCES_DIR=%s",
                stanza_resources_dir,
            )
    nlp_engine = StanzaNlpEngine(**nlp_engine_kwargs)

    registry = RecognizerRegistry(supported_languages=["en", "zh"])
    try:
        registry.load_predefined_recognizers(languages=["en"])
    except TypeError:
        registry.load_predefined_recognizers()
    for recognizer in _zh_pattern_recognizers():
        registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en", "zh"],
    )


def _run_presidio(request: AnalyzeRequest) -> list[PiiCandidate]:
    if not _env_bool("PII_ENABLE_PRESIDIO", True):
        return []
    language = _normalize_language(request.language)
    try:
        results = _engine().analyze(
            text=request.text,
            language=language,
            entities=request.entities,
            score_threshold=request.score_threshold,
            context=request.context,
            return_decision_process=request.return_decision_process,
        )
    except Exception as exc:
        logger.warning("Presidio analysis failed: %s", exc)
        return []

    return [
        _candidate(
            entity_type=result.entity_type,
            text=request.text,
            start=result.start,
            end=result.end,
            score=result.score,
            recognizer="presidio",
            score_source="presidio_score",
            metadata={"analysis_explanation": str(getattr(result, "analysis_explanation", "") or "")},
        )
        for result in results
        if 0 <= result.start < result.end <= len(request.text)
    ]


@lru_cache(maxsize=1)
def _gliner_model():
    if not _env_bool("PII_ENABLE_GLINER", False):
        return None
    model_name = os.getenv("PII_GLINER_MODEL", "urchade/gliner_multi_pii-v1")
    try:
        from gliner import GLiNER

        logger.info("Loading GLiNER PII model: %s", model_name)
        return GLiNER.from_pretrained(model_name)
    except Exception as exc:
        logger.warning("GLiNER model unavailable: %s", exc)
        return None


def _gliner_labels() -> list[str]:
    configured = os.getenv("PII_GLINER_LABELS", "")
    if configured.strip():
        return [label.strip() for label in configured.split(",") if label.strip()]
    return list(GLINER_LABEL_TO_ENTITY)


def _run_gliner(request: AnalyzeRequest) -> list[PiiCandidate]:
    model = _gliner_model()
    if model is None:
        return []
    threshold = _env_float("PII_GLINER_THRESHOLD", 0.50)
    max_chars = int(os.getenv("PII_GLINER_MAX_CHARS", "12000"))
    text = request.text[:max_chars]
    try:
        raw_entities = model.predict_entities(text, _gliner_labels(), threshold=threshold)
    except Exception as exc:
        logger.warning("GLiNER analysis failed: %s", exc)
        return []

    candidates: list[PiiCandidate] = []
    for item in raw_entities:
        label = str(item.get("label", "")).strip().lower()
        entity_type = GLINER_LABEL_TO_ENTITY.get(label)
        if not entity_type:
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
            score = float(item.get("score", threshold))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text):
            candidates.append(
                _candidate(
                    entity_type=entity_type,
                    text=request.text,
                    start=start,
                    end=end,
                    score=score,
                    recognizer=f"gliner.{os.getenv('PII_GLINER_MODEL', 'urchade/gliner_multi_pii-v1')}",
                    score_source="model_score",
                    metadata={"label": label},
                )
            )
    return candidates


def _overlap_ratio(left: PiiCandidate, right: PiiCandidate) -> float:
    overlap = max(0, min(left.end, right.end) - max(left.start, right.start))
    if overlap == 0:
        return 0.0
    shorter = min(left.end - left.start, right.end - right.start)
    return overlap / max(1, shorter)


def _deduplicate(candidates: list[PiiCandidate]) -> list[PiiCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.start, -(item.end - item.start), -item.score))
    deduped: list[PiiCandidate] = []
    for candidate in ordered:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduped)
                if candidate.entity_type == existing.entity_type and _overlap_ratio(candidate, existing) >= 0.6
            ),
            None,
        )
        if duplicate_index is None:
            deduped.append(candidate)
            continue
        existing = deduped[duplicate_index]
        if candidate.score > existing.score:
            deduped[duplicate_index] = candidate
    return sorted(deduped, key=lambda item: (item.start, item.end, item.entity_type))


def _to_response(candidate: PiiCandidate) -> dict[str, Any]:
    return {
        "entity_type": candidate.entity_type,
        "start": candidate.start,
        "end": candidate.end,
        "score": candidate.score,
        "analysis_explanation": {
            "recognizer": candidate.recognizer,
            "score_source": candidate.score_source,
            "metadata": candidate.metadata or {},
        },
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> list[dict[str, Any]]:
    threshold = request.score_threshold if request.score_threshold is not None else 0.0
    candidates: list[PiiCandidate] = []

    if _env_bool("PII_ENABLE_REGEX_RULES", True):
        candidates.extend(_regex_candidates(request.text))
    candidates.extend(_run_presidio(request))
    candidates.extend(_run_gliner(request))

    filtered = [candidate for candidate in candidates if candidate.score >= threshold]
    if request.entities:
        allowed = {entity.upper() for entity in request.entities}
        filtered = [candidate for candidate in filtered if candidate.entity_type.upper() in allowed]

    return [_to_response(candidate) for candidate in _deduplicate(filtered)]

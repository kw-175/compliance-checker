"""
Step F: privacy detection and redaction span generation.
"""

from __future__ import annotations

import logging
import re

from audio.config.settings import Settings
from audio.models.schemas import DedupTranscriptUnit, PIIEntity, PrivacyResult, RedactionSpan, TranscriptUnit

logger = logging.getLogger(__name__)

_analyzer = None
_anonymizer = None

_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

_REPLACEMENTS = {
    "PERSON": "<PERSON>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
    "US_SSN": "<SSN>",
    "CREDIT_CARD": "<CREDIT_CARD>",
}


def _get_analyzer(settings: Settings):
    # 懒加载 Presidio 分析器，避免每次调用重复初始化 NLP 引擎。
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    _analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=settings.presidio_languages,
    )

    if settings.pii_model_name:
        try:
            from presidio_analyzer.predefined_recognizers import TransformersRecognizer

            recognizer = TransformersRecognizer(
                model_path=settings.pii_model_name,
                supported_entities=[
                    "PERSON",
                    "EMAIL_ADDRESS",
                    "PHONE_NUMBER",
                    "CREDIT_CARD",
                    "US_SSN",
                    "LOCATION",
                    "ORGANIZATION",
                ],
                supported_language="en",
            )
            _analyzer.registry.add_recognizer(recognizer)
        except Exception as exc:
            # 自定义模型注册失败时保留内置识别器继续运行。
            logger.warning("PII model registration failed, fallback to built-ins: %s", exc)

    return _analyzer


def _get_anonymizer():
    # 懒加载匿名化引擎，保持与 analyzer 生命周期一致。
    global _anonymizer
    if _anonymizer is not None:
        return _anonymizer
    from presidio_anonymizer import AnonymizerEngine

    _anonymizer = AnonymizerEngine()
    return _anonymizer


def _fallback_detect(text: str) -> list[PIIEntity]:
    # 当 Presidio 不可用时，使用正则进行基础 PII 检测。
    entities: list[PIIEntity] = []
    for entity_type, pattern in [
        ("EMAIL_ADDRESS", _EMAIL_RE),
        ("PHONE_NUMBER", _PHONE_RE),
        ("US_SSN", _SSN_RE),
        ("CREDIT_CARD", _CREDIT_RE),
    ]:
        for match in pattern.finditer(text):
            entities.append(
                PIIEntity(
                    entity_type=entity_type,
                    start=match.start(),
                    end=match.end(),
                    score=0.8,
                    original_text=match.group()[:100],
                )
            )
    entities.sort(key=lambda item: item.start)
    return entities


def _presidio_detect(text: str, analyzer, settings: Settings) -> list[PIIEntity]:
    from presidio_analyzer import RecognizerResult

    candidates: list[RecognizerResult] = []
    for lang in settings.presidio_languages:
        try:
            rows = analyzer.analyze(
                text=text,
                language=lang,
                score_threshold=settings.pii_score_threshold,
            )
            candidates.extend(rows)
        except Exception as exc:
            # 某语言检测失败不影响其他语言继续执行。
            logger.debug("Presidio language pass failed (%s): %s", lang, exc)

    # Keep higher-score non-overlapping entities.
    candidates.sort(key=lambda item: (-item.score, item.start))
    entities: list[PIIEntity] = []
    occupied: list[tuple[int, int]] = []
    for item in candidates:
        overlap = any(item.start < end and item.end > start for start, end in occupied)
        if overlap:
            continue
        entities.append(
            PIIEntity(
                entity_type=item.entity_type,
                start=int(item.start),
                end=int(item.end),
                score=float(item.score),
                original_text=text[item.start:item.end][:100],
            )
        )
        occupied.append((int(item.start), int(item.end)))
    entities.sort(key=lambda item: item.start)
    return entities


def _to_time(unit: DedupTranscriptUnit | TranscriptUnit, start: int, end: int) -> tuple[float, float]:
    # 将文本字符偏移近似映射为音频时间区间，用于后续音频脱敏。
    if not unit.text or len(unit.text) == 0:
        return unit.start_time, unit.end_time
    duration = max(unit.end_time - unit.start_time, 0.0)
    start_ratio = start / max(len(unit.text), 1)
    end_ratio = end / max(len(unit.text), 1)
    return (
        unit.start_time + duration * start_ratio,
        unit.start_time + duration * end_ratio,
    )


def _redact_text(text: str, entities: list[PIIEntity]) -> str:
    # 纯文本替换兜底实现，不依赖外部匿名化组件。
    if not entities:
        return text
    parts: list[str] = []
    cursor = 0
    for entity in entities:
        parts.append(text[cursor:entity.start])
        parts.append(_REPLACEMENTS.get(entity.entity_type, "<REDACTED>"))
        cursor = entity.end
    parts.append(text[cursor:])
    return "".join(parts)


def _presidio_redact(text: str, entities: list[PIIEntity], anonymizer) -> str:
    # 使用 Presidio 按实体类型执行可配置替换。
    if not entities:
        return text
    from presidio_anonymizer.entities import OperatorConfig
    from presidio_analyzer import RecognizerResult

    results = [
        RecognizerResult(
            entity_type=item.entity_type,
            start=item.start,
            end=item.end,
            score=item.score,
        )
        for item in entities
    ]
    redacted = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
            "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
            "US_SSN": OperatorConfig("replace", {"new_value": "<SSN>"}),
            "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        },
    )
    return redacted.text


def run(units: list[DedupTranscriptUnit] | list[TranscriptUnit], settings: Settings | None = None) -> tuple[list[PrivacyResult], list[RedactionSpan]]:
    if settings is None:
        from audio.config.settings import get_settings

        settings = get_settings()

    analyzer = anonymizer = None
    use_presidio = True
    try:
        analyzer = _get_analyzer(settings)
        anonymizer = _get_anonymizer()
    except Exception as exc:
        # 依赖缺失时自动切换到正则兜底，确保流程可用。
        logger.warning("Presidio unavailable, fallback to regex detectors: %s", exc)
        use_presidio = False

    results: list[PrivacyResult] = []
    spans: list[RedactionSpan] = []
    for unit in units:
        if getattr(unit, "is_duplicate", False):
            continue
        # 优先 Presidio；若无结果或不可用，则退化到正则检测。
        entities = _presidio_detect(unit.text, analyzer, settings) if use_presidio else _fallback_detect(unit.text)
        if not entities and use_presidio:
            entities = _fallback_detect(unit.text)
        if use_presidio and anonymizer is not None:
            redacted = _presidio_redact(unit.text, entities, anonymizer)
        else:
            redacted = _redact_text(unit.text, entities)
        results.append(
            PrivacyResult(
                unit_id=unit.unit_id,
                source_id=unit.source_id,
                original_text=unit.text,
                redacted_text=redacted,
                pii_entities=entities,
                pii_count=len(entities),
            )
        )
        for entity in entities:
            start_time, end_time = _to_time(unit, entity.start, entity.end)
            # 为每个实体输出独立时间跨度，供音频层精准处理。
            spans.append(
                RedactionSpan(
                    source_id=unit.source_id,
                    unit_id=unit.unit_id,
                    start_time=start_time,
                    end_time=end_time,
                    entity_type=entity.entity_type,
                    original_text=entity.original_text,
                    replacement=_REPLACEMENTS.get(entity.entity_type, "<REDACTED>"),
                )
            )
    return results, spans

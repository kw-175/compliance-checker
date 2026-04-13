from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NerModelConfiguration, StanzaNlpEngine
from presidio_analyzer.predefined_recognizers import EmailRecognizer, PhoneRecognizer

app = FastAPI(title="Bilingual Presidio Analyzer")


class AnalyzeRequest(BaseModel):
    text: str
    language: str = "en"
    score_threshold: float | None = None
    entities: list[str] | None = None
    context: list[str] | None = None
    return_decision_process: bool = False


def _zh_recognizers() -> list[PatternRecognizer]:
    return [
        PatternRecognizer(
            supported_entity="CN_ID_CARD",
            supported_language="zh",
            patterns=[
                Pattern(
                    name="cn_id_card",
                    regex=r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)",
                    score=0.85,
                )
            ],
            context=["身份证", "证件号", "居民身份证", "身份号码"],
        ),
        PatternRecognizer(
            supported_entity="CN_PHONE_NUMBER",
            supported_language="zh",
            patterns=[
                Pattern(
                    name="cn_mobile_phone",
                    regex=r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)",
                    score=0.8,
                )
            ],
            context=["电话", "手机号", "手机", "联系方式", "家长电话"],
        ),
        PatternRecognizer(
            supported_entity="STUDENT_ID",
            supported_language="zh",
            patterns=[
                Pattern(
                    name="student_id_contextual",
                    regex=r"(?:学号|学生编号|student\s*id)[:：\s]*([A-Za-z0-9_-]{4,32})",
                    score=0.78,
                )
            ],
            context=["学号", "学生编号", "学生信息"],
        ),
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            supported_language="zh",
            patterns=[
                Pattern(
                    name="email",
                    regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
                    score=0.9,
                )
            ],
            context=["邮箱", "电子邮件", "邮件"],
        ),
    ]


@lru_cache(maxsize=1)
def _engine() -> AnalyzerEngine:
    model_config = [
        {"lang_code": "en", "model_name": "en"},
        {"lang_code": "zh", "model_name": "zh"},
    ]
    entity_mapping = {
        "PERSON": "PERSON",
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
    }
    ner_config = NerModelConfiguration(model_to_presidio_entity_mapping=entity_mapping)
    nlp_engine = StanzaNlpEngine(models=model_config, ner_model_configuration=ner_config)

    registry = RecognizerRegistry(supported_languages=["en", "zh"])
    registry.add_recognizer(EmailRecognizer(supported_language="en"))
    registry.add_recognizer(PhoneRecognizer(supported_language="en"))
    for recognizer in _zh_recognizers():
        registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en", "zh"],
    )


def _to_presidio_response(result) -> dict[str, Any]:
    return {
        "entity_type": result.entity_type,
        "start": result.start,
        "end": result.end,
        "score": result.score,
        "analysis_explanation": None,
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> list[dict[str, Any]]:
    language = request.language if request.language in {"en", "zh"} else "en"
    results = _engine().analyze(
        text=request.text,
        language=language,
        entities=request.entities,
        score_threshold=request.score_threshold,
        context=request.context,
        return_decision_process=request.return_decision_process,
    )
    return [_to_presidio_response(result) for result in results]

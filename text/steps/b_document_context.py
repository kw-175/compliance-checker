from __future__ import annotations

import logging
from typing import Any

from text.api_clients import OpenAICompatibleAPIError, OpenAICompatibleComplianceClient, resolve_provider_config
from text.config.settings import Settings, get_settings
from text.models.schemas import DocumentContextRecord, IngestUnit
from text.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return default


def _normalize_text(value: Any, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _contains_any(text: str, patterns: set[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


def _heuristic_context(unit: IngestUnit) -> DocumentContextRecord:
    text = unit.text.lower()
    metadata = {str(key).lower(): value for key, value in unit.metadata.items()}

    document_type = "other"
    scene_type = "other"
    subject_type = "unknown"
    source_type = _normalize_text(unit.source_type or metadata.get("source_type"), "other")
    usage_target = "annotation_preprocess" if "annot" in str(metadata.get("usage", "")).lower() else "training_dataset"
    topic = "general text"
    confidence = 0.46

    if _contains_any(text, {"学号", "成绩", "班级", "家长", "监护人", "学生档案", "student id", "grade"}):
        document_type = "student_record"
        scene_type = "education"
        subject_type = "student"
        topic = "student education record"
        confidence = 0.74
    elif _contains_any(text, {"教材", "例题", "示例", "课本", "textbook", "example"}):
        document_type = "textbook_example"
        scene_type = "education"
        subject_type = "mixed"
        topic = "teaching example"
        confidence = 0.70
    elif _contains_any(text, {"新闻", "报道", "记者", "news", "reported"}):
        document_type = "news_report"
        scene_type = "public_communication"
        subject_type = "mixed"
        topic = "news reporting"
        confidence = 0.66
    elif _contains_any(text, {"测试", "样例", "mock", "fixture", "test sample"}):
        document_type = "test_sample"
        scene_type = "training_material"
        subject_type = "unknown"
        topic = "test sample"
        confidence = 0.62

    contains_education_context = scene_type == "education" or _contains_any(
        text, {"学校", "老师", "学生", "课堂", "school", "teacher", "student"}
    )
    contains_minor_context = subject_type in {"student", "minor"} or _contains_any(
        text, {"未成年人", "儿童", "学生", "minor", "child", "teenager"}
    )
    if contains_minor_context and subject_type == "unknown":
        subject_type = "minor"

    explanation = (
        "The document context was inferred heuristically from obvious education, record, example, "
        "or reporting cues in the raw text and metadata."
    )
    summary = f"Inferred {document_type} / {scene_type} context."

    return DocumentContextRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        topic=topic,
        document_type=document_type,
        scene_type=scene_type,
        subject_type=subject_type,
        source_type=source_type,
        usage_target=usage_target,
        contains_education_context=contains_education_context,
        contains_minor_context=contains_minor_context,
        confidence=confidence,
        summary=summary,
        explanation=explanation,
        provider_name="heuristic_document_context",
        provider_version="builtin-2026.05",
        is_degraded=False,
        attributes={"metadata_snapshot": unit.metadata},
    )


def _payload(unit: IngestUnit, settings: Settings, provider_max_chars: int) -> dict[str, Any]:
    return {
        "run_id": unit.run_id,
        "doc_id": unit.doc_id,
        "language": unit.language,
        "text": unit.text[: min(settings.max_text_chars_per_document, provider_max_chars)],
        "metadata": unit.metadata,
        "source_type": unit.source_type,
        "candidate_profiles": unit.candidate_profiles,
    }


def _normalize_payload(
    unit: IngestUnit,
    payload: dict[str, Any],
    provider_name: str,
    provider_version: str,
    is_degraded: bool,
) -> DocumentContextRecord:
    return DocumentContextRecord(
        run_id=unit.run_id,
        doc_id=unit.doc_id,
        text_hash=unit.text_hash,
        topic=_normalize_text(payload.get("topic"), "general text"),
        document_type=_normalize_text(payload.get("document_type"), "other"),
        scene_type=_normalize_text(payload.get("scene_type"), "other"),
        subject_type=_normalize_text(payload.get("subject_type"), "unknown"),
        source_type=_normalize_text(payload.get("source_type") or unit.source_type, "other"),
        usage_target=_normalize_text(payload.get("usage_target"), "training_dataset"),
        contains_education_context=bool(payload.get("contains_education_context", False)),
        contains_minor_context=bool(payload.get("contains_minor_context", False)),
        confidence=_safe_float(payload.get("confidence"), 0.55),
        summary=_normalize_text(payload.get("summary"), "Document context inferred."),
        explanation=_normalize_text(
            payload.get("explanation"),
            "The document context was inferred by the local compliance model.",
        ),
        provider_name=provider_name,
        provider_version=provider_version,
        is_degraded=is_degraded,
        attributes={"raw_payload": payload},
    )


def run(
    ingest_units: list[IngestUnit],
    settings: Settings | None = None,
) -> list[DocumentContextRecord]:
    settings = settings or get_settings()

    try:
        provider = resolve_provider_config(settings)
    except OpenAICompatibleAPIError:
        provider = None

    if provider is None or provider.mode != "local_model":
        return [_heuristic_context(unit) for unit in ingest_units]

    client = OpenAICompatibleComplianceClient(settings)
    system_prompt = load_prompt(str(settings.local_document_context_prompt_path))
    results: list[DocumentContextRecord] = []

    for unit in ingest_units:
        try:
            payload = client.complete_json(
                task_name="document_context",
                system_prompt=system_prompt,
                payload=_payload(unit, settings, provider.max_chars),
            )
            results.append(
                _normalize_payload(
                    unit,
                    payload,
                    provider_name="local_document_context",
                    provider_version=provider.model,
                    is_degraded=False,
                )
            )
        except Exception as exc:
            logger.warning("Local document-context inference failed for %s: %s", unit.doc_id, exc)
            heuristic = _heuristic_context(unit)
            results.append(
                heuristic.model_copy(
                    update={
                        "is_degraded": True,
                        "summary": f"{heuristic.summary} Fallback heuristic was used.",
                        "attributes": {
                            **heuristic.attributes,
                            "degrade_reason": str(exc),
                        },
                    }
                )
            )

    logger.info("Document-context build completed: %d documents", len(results))
    return results

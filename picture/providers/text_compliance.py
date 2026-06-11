from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

from picture.domain.enums import FindingType
from picture.domain.models import BBox, OCRLayoutResult, OCRTextBlock, PictureFinding, Polygon, RegionMask


def run_text_pipeline_for_ocr(
    ocr_result: OCRLayoutResult,
    *,
    profile: str,
    run_id: str,
    work_dir: str | Path,
    text_api_base_url: str = "",
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 2.0,
    config_overrides: dict[str, Any] | None = None,
) -> list[PictureFinding]:
    """
    Reuse the completed text compliance pipeline for OCR text.

    The picture module does not load Qwen3.5-9B itself. The text pipeline resolves
    the configured local/API provider and therefore reuses the already running
    text-compliance Qwen endpoint.
    """
    audit_index = _build_ocr_audit_index(ocr_result)
    text = str(audit_index.get("text") or "").strip()
    block_spans = list(audit_index.get("block_spans") or [])
    if not text or not block_spans:
        return []
    metadata = dict(ocr_result.metadata or {})
    if metadata.get("valid_text") is False:
        return []

    package_path = _write_ocr_text_package(ocr_result, run_id=run_id, work_dir=work_dir)

    if text_api_base_url.strip():
        payload = _run_text_api(
            package_path,
            profile=profile,
            base_url=text_api_base_url,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            config_overrides=config_overrides,
        )
    else:
        from text.api_pipeline import APICompliancePipeline
        from text.config.settings import get_settings as get_text_settings

        base_text_settings = get_text_settings()
        valid_overrides = {
            key: value
            for key, value in dict(config_overrides or {}).items()
            if hasattr(base_text_settings, key)
        }
        text_settings = base_text_settings.model_copy(update=valid_overrides)
        pipeline = APICompliancePipeline(settings=text_settings, run_id=f"{run_id}_ocr_text_{uuid.uuid4().hex[:8]}")
        output = pipeline.execute([str(package_path)], profile=profile)
        payload = {
            "legacy_decision": output.legacy_decision,
            "metadata": dict(output.metadata or {}),
            "pipeline_run_id": output.pipeline_run_id,
        }

    picture_findings: list[PictureFinding] = []
    picture_findings.extend(_extract_artifact_findings(payload, ocr_result))

    legacy = payload.get("legacy_decision") if isinstance(payload, dict) else {}
    documents = legacy.get("documents", []) if isinstance(legacy, dict) else []
    if not isinstance(documents, list):
        documents = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        picture_findings.extend(_extract_findings(document, ocr_result))
    return _dedupe_picture_findings(picture_findings)


def _run_text_api(
    package_path: Path,
    *,
    profile: str,
    base_url: str,
    timeout_seconds: float,
    poll_interval_seconds: float,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import httpx

    normalized = base_url.strip().rstrip("/")
    deadline = time.monotonic() + timeout_seconds
    overrides = {"pipeline_profile": profile, **dict(config_overrides or {})}
    overrides["pipeline_profile"] = profile
    with httpx.Client(timeout=min(30.0, timeout_seconds)) as client:
        response = client.post(
            f"{normalized}/api/v1/check",
            json={
                "package_paths": [str(package_path)],
                "config_overrides": overrides,
            },
        )
        response.raise_for_status()
        task_id = response.json()["task_id"]

        while time.monotonic() < deadline:
            status_response = client.get(f"{normalized}/api/v1/status/{task_id}")
            status_response.raise_for_status()
            status_payload = status_response.json()
            status = str(status_payload.get("status", "")).lower()
            if status == "completed":
                result_response = client.get(f"{normalized}/api/v1/result/{task_id}")
                result_response.raise_for_status()
                result_payload = result_response.json()
                result_payload["text_api_task_id"] = task_id
                return result_payload
            if status == "failed":
                raise RuntimeError(status_payload.get("error") or f"text API task {task_id} failed")
            time.sleep(poll_interval_seconds)

    raise TimeoutError(f"text API task did not finish within {timeout_seconds:.1f}s")


def _write_ocr_text_package(ocr_result: OCRLayoutResult, *, run_id: str, work_dir: str | Path) -> Path:
    package_dir = Path(work_dir) / "ocr_text_compliance"
    package_dir.mkdir(parents=True, exist_ok=True)
    package_path = package_dir / "cleaned_docs.jsonl"
    audit_index = _build_ocr_audit_index(ocr_result)
    block_spans = list(audit_index.get("block_spans") or [])
    layout_path = package_dir / "ocr_layout_blocks.jsonl"
    layout_records = [
        {
            "doc_id": f"{run_id}_ocr_text",
            "block_id": item["block_id"],
            "text": item["text"],
            "start": item["start"],
            "end": item["end"],
            "bbox": item["bbox"],
            "polygon": item.get("polygon"),
            "confidence": item["confidence"],
            "source_text_start": item.get("source_text_start"),
            "source_text_end": item.get("source_text_end"),
            "source_visible_start": item.get("source_visible_start"),
            "source_visible_end": item.get("source_visible_end"),
            "ocr_block_index": item.get("ocr_block_index"),
            "unit_kind": item.get("unit_kind"),
        }
        for item in block_spans
    ]
    layout_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in layout_records),
        encoding="utf-8",
    )
    record = {
        "doc_id": f"{run_id}_ocr_text",
            "text": audit_index.get("text") or "",
        "source_type": "picture_ocr",
        "metadata": {
            "source_modality": "image",
            "ocr_engine": ocr_result.engine_name,
            "ocr_block_count": len(ocr_result.text_blocks),
            "ocr_mean_confidence": _mean_confidence(ocr_result),
            "ocr_layout_blocks_path": str(layout_path),
            "ocr_offset_strategy": "canonical_ocr_units_with_virtual_newlines",
            "ocr_unit_schema": "ocr_units_v2",
            "spatially_mappable_text": bool(block_spans),
            "ocr_text_instance_count": len(_ocr_text_instances(ocr_result)),
        },
    }
    package_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return package_path


def _build_ocr_block_spans(ocr_result: OCRLayoutResult) -> list[dict[str, Any]]:
    return list(_build_ocr_audit_index(ocr_result).get("block_spans") or [])


def _build_ocr_audit_index(ocr_result: OCRLayoutResult) -> dict[str, Any]:
    """
    Build the exact text sent to text compliance and a deterministic OCR offset map.

    The text compliance service returns character spans against this virtual string,
    so every non-virtual character must be traceable back to one OCR block.
    """
    cursor = 0
    spans: list[dict[str, Any]] = []
    offset_map: list[dict[str, Any]] = []
    parts: list[str] = []
    for index, block in enumerate(ocr_result.text_blocks):
        text = block.text or ""
        if not text:
            continue
        if parts:
            parts.append("\n")
            offset_map.append({"index": cursor, "virtual": True, "char": "\n"})
            cursor += 1
        start = cursor
        end = start + len(text)
        for local_index, char in enumerate(text):
            offset_map.append(
                {
                    "index": start + local_index,
                    "virtual": False,
                    "ocr_block_index": index,
                    "block_id": f"ocr_block_{index + 1:04d}",
                    "local_char_index": local_index,
                    "char": char,
                }
            )
        parts.append(text)
        cursor = max(cursor, end)
        spans.extend(_ocr_unit_spans_for_block(block, index, start))
    return {"text": "".join(parts), "block_spans": spans, "offset_map": offset_map}


def _ocr_unit_spans_for_block(block: OCRTextBlock, block_index: int, global_start: int) -> list[dict[str, Any]]:
    text = block.text or ""
    visible_ranges = _visible_char_poly_ranges(text)
    if "\n" not in text:
        return [
            {
                "block_id": f"ocr_block_{block_index + 1:04d}",
                "ocr_block_index": block_index,
                "text": text,
                "start": global_start,
                "end": global_start + len(text),
                "bbox": block.bbox.model_dump(mode="json"),
                "polygon": block.polygon.model_dump(mode="json") if block.polygon is not None else None,
                "confidence": block.confidence,
                "source_text_start": 0,
                "source_text_end": len(text),
                "source_visible_start": 0,
                "source_visible_end": len(visible_ranges),
                "unit_kind": "ocr_block",
            }
        ]

    spans: list[dict[str, Any]] = []
    lines = text.splitlines(keepends=True)
    visual_line_count = max(1, sum(1 for line in lines if line.strip()))
    visual_index = 0
    local_cursor = 0
    for line_index, raw_line in enumerate(lines):
        line_text = raw_line.rstrip("\r\n")
        line_start = local_cursor
        local_cursor += len(raw_line)
        if not line_text.strip():
            continue
        bbox = _line_bbox(block.bbox, visual_index, visual_line_count)
        visible_start = _visible_count_before(text, line_start)
        visible_end = _visible_count_before(text, line_start + len(line_text))
        visual_index += 1
        spans.append(
            {
                "block_id": f"ocr_block_{block_index + 1:04d}_line_{line_index + 1:04d}",
                "ocr_block_index": block_index,
                "text": line_text,
                "start": global_start + line_start,
                "end": global_start + line_start + len(line_text),
                "bbox": bbox.model_dump(mode="json"),
                "polygon": _sub_polygon_from_bbox(bbox).model_dump(mode="json"),
                "confidence": block.confidence,
                "source_text_start": line_start,
                "source_text_end": line_start + len(line_text),
                "source_visible_start": visible_start,
                "source_visible_end": visible_end,
                "unit_kind": "ocr_line",
            }
        )
    return spans


def _line_bbox(bbox: BBox, line_index: int, line_count: int) -> BBox:
    line_count = max(1, line_count)
    line_h = max(1.0, float(bbox.h) / line_count)
    return BBox(x=float(bbox.x), y=float(bbox.y) + line_h * line_index, w=float(bbox.w), h=line_h)


def _visible_char_poly_ranges(text: str) -> list[int]:
    return [index for index, char in enumerate(text or "") if char not in {"\n", "\r"}]


def _visible_count_before(text: str, offset: int) -> int:
    return sum(1 for char in (text or "")[: max(0, offset)] if char not in {"\n", "\r"})


def _mean_confidence(ocr_result: OCRLayoutResult) -> float:
    blocks = list(ocr_result.text_blocks)
    if not blocks:
        return 0.0
    return round(sum(float(block.confidence) for block in blocks) / len(blocks), 4)


def _extract_artifact_findings(
    payload: dict[str, Any],
    ocr_result: OCRLayoutResult,
) -> list[PictureFinding]:
    artifact_paths = _artifact_paths_from_payload(payload)
    if not artifact_paths:
        return []

    privacy_audit = _read_jsonl(artifact_paths.get("privacy_audit", ""))
    policy_records = _read_jsonl(artifact_paths.get("policy", ""))
    redaction_records = _read_jsonl(artifact_paths.get("redaction_plan", ""))
    content_audit = _read_jsonl(artifact_paths.get("content_safety_audit", ""))
    content_final = _read_jsonl(artifact_paths.get("content_safety_final_decisions", ""))

    findings: list[PictureFinding] = []
    audit_by_id = {
        str(item.get("finding_id")): item
        for item in privacy_audit
        if isinstance(item, dict) and item.get("finding_id")
    }

    for item in privacy_audit:
        if isinstance(item, dict):
            finding = _privacy_audit_to_finding(item, ocr_result, artifact_paths)
            if finding is not None:
                findings.append(finding)

    for item in _policy_redaction_targets(policy_records, redaction_records):
        if not isinstance(item, dict):
            continue
        source = audit_by_id.get(str(item.get("finding_id"))) or {}
        merged = {**source, **item}
        finding = _privacy_audit_to_finding(merged, ocr_result, artifact_paths, source_kind="redaction_target")
        if finding is not None:
            findings.append(finding)

    for item in content_audit + content_final:
        if isinstance(item, dict):
            content = _content_audit_to_finding(item, ocr_result, artifact_paths)
            if content is not None:
                findings.append(content)

    if not any(
        finding.finding_type == FindingType.TEXT_PII and finding.category == "person_name"
        for finding in findings
    ):
        findings.extend(_isolated_ocr_name_findings(ocr_result, artifact_paths))

    return findings


def _artifact_paths_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    legacy = payload.get("legacy_decision") if isinstance(payload.get("legacy_decision"), dict) else {}
    artifact_paths = metadata.get("artifact_paths") or legacy.get("artifact_paths") or {}
    if not isinstance(artifact_paths, dict):
        return {}
    return {str(key): str(value) for key, value in artifact_paths.items() if value}


def _read_jsonl(path_value: str) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def _policy_redaction_targets(
    policy_records: list[dict[str, Any]],
    redaction_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for record in policy_records:
        values = record.get("redaction_targets")
        if isinstance(values, list):
            targets.extend(item for item in values if isinstance(item, dict))
    for record in redaction_records:
        values = record.get("redaction_targets")
        if isinstance(values, list):
            targets.extend(item for item in values if isinstance(item, dict))
    return targets


def _privacy_audit_to_finding(
    item: dict[str, Any],
    ocr_result: OCRLayoutResult,
    artifact_paths: dict[str, str],
    *,
    source_kind: str = "privacy_audit",
) -> PictureFinding | None:
    text_span = _first_text(item, "original_text", "text", "matched_text", "evidence_text", "snippet")
    risk_type = _first_text(item, "pii_type", "risk_type", "entity_type", "policy_tag", default="pii_entity")
    category = _normalize_privacy_category(risk_type, item)
    start = _optional_int(item.get("start"))
    end = _optional_int(item.get("end"))
    if _is_machine_code_privacy_finding(category, text_span, ocr_result, start=start, end=end):
        return None
    region = _map_text_to_region(text_span, ocr_result, start=start, end=end) if text_span or start is not None else None
    region_trace = _ocr_region_trace(ocr_result, region, text_span, start=start, end=end)
    label = _privacy_label(item, category)
    explanation = _first_text(item, "reason_zh", "explanation", "summary") or _default_explanation(
        FindingType.TEXT_PII,
        label,
        text_span,
        region is not None,
    )
    score = _effective_privacy_score(item)
    metadata = {
        "text_pipeline_finding": item,
        "source_modality": "image_ocr",
        "source_kind": source_kind,
        "artifact_paths": artifact_paths,
        "char_start": start,
        "char_end": end,
        "region_source": "ocr_offset_map_or_text_match" if region is not None else "",
        "region_missing": region is None,
        "requires_manual_region_review": region is None,
        **region_trace,
        "privacy_action": item.get("privacy_action") or item.get("action"),
        "dataset_route": item.get("dataset_route"),
        "sensitivity_level": item.get("sensitivity_level"),
        "operator_id": item.get("operator_id"),
        "operator_name_zh": item.get("operator_name_zh"),
    }
    return PictureFinding(
        finding_type=FindingType.TEXT_PII,
        category=category,
        label=label,
        score=score,
        region=region,
        text_span=text_span or None,
        reason_code=f"OCR_TEXT_PII_{category.upper()}",
        provider=str(item.get("source") or "text_compliance_pipeline"),
        provider_version=str((item.get("versions") or {}).get("provider_model") or item.get("provider_version") or ""),
        explanation=explanation,
        metadata=metadata,
    )


def _content_audit_to_finding(
    item: dict[str, Any],
    ocr_result: OCRLayoutResult,
    artifact_paths: dict[str, str],
) -> PictureFinding | None:
    category = _first_text(item, "risk_type", "category", "policy_tag", default="")
    text_span = _first_text(item, "text", "original_text", "matched_text", "evidence_text", "snippet")
    if not category and not text_span:
        return None
    normalized = _normalize_category(category or "content_safety")
    start = _optional_int(item.get("start"))
    end = _optional_int(item.get("end"))
    region = _map_text_to_region(text_span, ocr_result, start=start, end=end) if text_span or start is not None else None
    region_trace = _ocr_region_trace(ocr_result, region, text_span, start=start, end=end)
    return PictureFinding(
        finding_type=FindingType.TEXT_CONTENT,
        category=normalized,
        label=_content_label(item, normalized),
        score=float(item.get("confidence") or item.get("score") or 0.8),
        region=region,
        text_span=text_span or None,
        reason_code=f"OCR_TEXT_CONTENT_{normalized.upper()}",
        provider=str(item.get("source") or item.get("provider_name") or "text_compliance_pipeline"),
        provider_version=str(item.get("provider_version") or ""),
        explanation=_first_text(item, "reason_zh", "explanation", "summary") or _default_explanation(
            FindingType.TEXT_CONTENT,
            normalized,
            text_span,
            region is not None,
        ),
        metadata={
            "text_pipeline_finding": item,
            "source_modality": "image_ocr",
            "source_kind": "content_safety_artifact",
            "artifact_paths": artifact_paths,
            "char_start": start,
            "char_end": end,
            "region_source": "ocr_offset_map_or_text_match" if region is not None else "",
            "region_missing": region is None,
            "requires_manual_region_review": region is None,
            **region_trace,
        },
    )


def _extract_findings(document: dict[str, Any], ocr_result: OCRLayoutResult) -> list[PictureFinding]:
    candidates = _collect_text_compliance_findings(document)

    results: list[PictureFinding] = []
    seen: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        raw_type = _first_text(item, "finding_type", "type", "risk_domain", "domain", "source")
        risk_type = _first_text(
            item,
            "risk_type",
            "category",
            "label",
            "policy_tag",
            "operator_id",
            default="unknown",
        ).strip() or "unknown"
        normalized_raw = raw_type.lower()
        normalized_risk = risk_type.lower()
        if (
            normalized_raw == "privacy"
            or "privacy" in normalized_raw
            or normalized_risk.startswith("pii")
            or normalized_risk in _PRIVACY_CATEGORY_HINTS
        ):
            finding_type = FindingType.TEXT_PII
            reason_prefix = "OCR_TEXT_PII"
        elif (
            normalized_raw == "content_safety"
            or "content" in normalized_raw
            or normalized_risk in _CONTENT_CATEGORY_HINTS
            or any(token in normalized_risk for token in ("safe", "violence", "explicit", "illegal", "hate", "politic"))
        ):
            finding_type = FindingType.TEXT_CONTENT
            reason_prefix = "OCR_TEXT_CONTENT"
        else:
            continue

        span = item.get("span") if isinstance(item.get("span"), dict) else {}
        text_span = _first_text(
            span,
            "text",
            "matched_text",
            "snippet",
            "evidence",
            "content",
            "value",
            "raw_text",
        ) or _first_text(
            item,
            "text",
            "matched_text",
            "snippet",
            "evidence",
            "evidence_text",
            "content",
            "value",
            "raw_text",
        )
        text_span = text_span.strip()
        start = _optional_int(span.get("start") if span else item.get("start"))
        end = _optional_int(span.get("end") if span else item.get("end"))
        dedupe_key = f"{finding_type.value}:{risk_type}:{text_span}:{item.get('finding_id', '')}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        category = _normalize_category(risk_type)
        if finding_type == FindingType.TEXT_PII and _is_machine_code_privacy_finding(
            category,
            text_span,
            ocr_result,
            start=start,
            end=end,
        ):
            continue
        region = _map_text_to_region(text_span, ocr_result, start=start, end=end) if text_span or start is not None else None
        if finding_type == FindingType.TEXT_PII and category == "address" and region is not None:
            region = _expand_address_region(text_span, ocr_result, start=start, end=end) or region
        region_trace = _ocr_region_trace(ocr_result, region, text_span, start=start, end=end)
        if finding_type == FindingType.TEXT_PII and category == "address" and region is not None:
            address_units = _ocr_units_overlapping_region(ocr_result, region)
            if address_units:
                region_trace.update(
                    {
                        "ocr_unit_ids": [str(unit.get("block_id")) for unit in address_units],
                        "ocr_unit_texts": [str(unit.get("text") or "") for unit in address_units],
                        "ocr_region_source": "ocr_address_group",
                        "ocr_region_quality": "medium",
                        "region_source": "ocr_address_group",
                    }
                )
        explanation = str(
            item.get("explanation")
            or item.get("reason")
            or item.get("summary")
            or _default_explanation(finding_type, risk_type, text_span, region is not None)
        )
        results.append(
            PictureFinding(
                finding_type=finding_type,
                category=category,
                label=str(item.get("policy_tag") or item.get("label_zh") or item.get("label") or risk_type),
                score=float(item.get("confidence") or item.get("score") or 0.0),
                region=region,
                text_span=text_span or None,
                reason_code=f"{reason_prefix}_{category.upper()}",
                provider=str(item.get("source_tool") or "text_compliance_pipeline"),
                provider_version=str(item.get("provider_version") or ""),
                explanation=explanation,
                metadata={
                    "text_pipeline_finding": item,
                    "source_modality": "image_ocr",
                    "char_start": start,
                    "char_end": end,
                    "region_source": "ocr_offset_map" if region is not None else "",
                    "region_missing": region is None,
                    "requires_manual_region_review": region is None,
                    **region_trace,
                },
            )
        )
    if not any(finding.finding_type == FindingType.TEXT_PII and finding.category == "address" for finding in results):
        results.extend(_address_recall_findings(document, ocr_result, seen))
    return results


def _isolated_ocr_name_findings(
    ocr_result: OCRLayoutResult,
    artifact_paths: dict[str, str] | None = None,
) -> list[PictureFinding]:
    """Recall short standalone Chinese names in document-style OCR layouts.

    Generic text PII intentionally avoids treating every 2-4 Chinese character span as
    a name. Picture OCR has stronger layout evidence, so we add a conservative recall
    only when a standalone short Chinese line sits near class/date/contact lines.
    """
    units = _build_ocr_block_spans(ocr_result)
    findings: list[PictureFinding] = []
    for index, unit in enumerate(units):
        text = re.sub(r"\s+", "", str(unit.get("text") or ""))
        if not _looks_like_isolated_cn_name(text):
            continue
        context = "\n".join(
            str(item.get("text") or "")
            for item in units[max(0, index - 2): min(len(units), index + 4)]
            if item is not unit
        )
        if not _isolated_name_context_supports_pii(context):
            continue
        region = _region_from_ocr_span_items([unit], "ocr_isolated_name_recall")
        if region is None:
            continue
        start = _optional_int(unit.get("start"))
        end = _optional_int(unit.get("end"))
        findings.append(
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="person_name",
                label="姓名信息",
                score=0.68,
                region=region,
                text_span=text,
                reason_code="OCR_TEXT_PII_PERSON_NAME",
                provider="picture_ocr_isolated_name_recall",
                provider_version="builtin-2026.05",
                explanation="OCR 识别到独立中文姓名行，且邻近班级、日期或联系方式等身份上下文，需要作为姓名隐私信息进入治理。",
                metadata={
                    "source_modality": "image_ocr",
                    "source_kind": "ocr_isolated_name_recall",
                    "text_pipeline_finding": {
                        "risk_type": "person_name",
                        "policy_tag": "pii.person_name",
                        "text": text,
                        "start": start,
                        "end": end,
                        "confidence": 0.68,
                        "source": "picture_ocr_isolated_name_recall",
                        "reason_zh": "独立中文姓名行位于教育文档身份上下文中，需要脱敏或复核。",
                    },
                    "artifact_paths": dict(artifact_paths or {}),
                    "char_start": start,
                    "char_end": end,
                    "region_source": "ocr_isolated_name_line",
                    "region_missing": False,
                    "requires_manual_region_review": False,
                    "ocr_unit_ids": [str(unit.get("block_id") or "")],
                    "ocr_unit_texts": [str(unit.get("text") or "")],
                    "ocr_region_source": "ocr_isolated_name_line",
                    "ocr_region_quality": "medium",
                    "operator_id": "PII_001",
                    "operator_name_zh": "姓名检测",
                    "privacy_action": "mask",
                    "dataset_route": "training_after_redaction",
                    "sensitivity_level": "S2",
                },
            )
        )
    return findings


def _looks_like_isolated_cn_name(text: str) -> bool:
    value = str(text or "").strip()
    if not re.fullmatch(r"[\u4e00-\u9fa5·]{2,4}", value):
        return False
    blocked_exact = {
        "内科",
        "护理",
        "血糖",
        "案例",
        "学院",
        "教研室",
        "助产",
        "小组",
        "学时",
    }
    if value in blocked_exact:
        return False
    blocked_tokens = ("学院", "大学", "医院", "护理", "教研", "班", "组", "学时", "案例", "课程", "PBL")
    return not any(token in value for token in blocked_tokens)


def _isolated_name_context_supports_pii(context: str) -> bool:
    value = str(context or "")
    if not value.strip():
        return False
    return any(
        re.search(pattern, value, flags=re.IGNORECASE)
        for pattern in (
            r"\d{2,4}.*班",
            r"小组",
            r"联系电话|联系方式|手机号|电话|邮箱|email",
            r"1[3-9]\d{9}",
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"学院|学校|大学|教研室",
        )
    )


def _normalize_category(value: str) -> str:
    return value.lower().replace(".", "_").replace("-", "_").replace(" ", "_")


def _is_machine_code_privacy_finding(
    category: str,
    text_span: str,
    ocr_result: OCRLayoutResult,
    *,
    start: int | None = None,
    end: int | None = None,
) -> bool:
    normalized = _normalize_category(category or "")
    if normalized not in {
        "social_account",
        "account",
        "bank_account",
        "phone_number",
        "student_id",
        "pii_entity",
        "combined_identity",
    }:
        return False
    span = str(text_span or "").strip()
    if not span:
        return False
    if normalized == "phone_number" and _looks_like_cn_mobile(span):
        return False
    block_text = _block_text_for_span(ocr_result, start=start, end=end) or span
    return _looks_like_machine_code_text(block_text, span)


def _looks_like_cn_mobile(value: str) -> bool:
    digits = re.sub(r"\D", "", str(value or ""))
    return bool(re.fullmatch(r"1[3-9]\d{9}", digits))


def _block_text_for_span(ocr_result: OCRLayoutResult, *, start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    span_end = end if end is not None and end >= start else start + 1
    for item in _build_ocr_block_spans(ocr_result):
        block_start = int(item["start"])
        block_end = int(item["end"])
        if max(start, block_start) < min(span_end, block_end):
            return str(item.get("text") or "")
    return ""


def _looks_like_machine_code_text(block_text: str, span_text: str) -> bool:
    block = str(block_text or "").strip()
    span = str(span_text or "").strip()
    if not block or not span:
        return False
    compact_block = re.sub(r"\s+", "", block)
    compact_span = re.sub(r"\s+", "", span)
    if len(compact_block) < 6 and len(compact_span) < 6:
        return False
    if re.search(r"[一-龥A-Za-z]", compact_block):
        return False
    digit_count = len(re.findall(r"\d", compact_block))
    alnum_count = len(re.findall(r"[A-Za-z0-9]", compact_block))
    separator_count = len(re.findall(r"[|:：/\\_.\-#]", compact_block))
    if digit_count < 6:
        return False
    if alnum_count and digit_count / alnum_count < 0.85:
        return False
    if separator_count >= 2:
        return True
    return bool(re.fullmatch(r"\d{8,}", compact_block)) and bool(re.fullmatch(r"\d{4,}", compact_span))


def _ocr_region_trace(
    ocr_result: OCRLayoutResult,
    region: RegionMask | None,
    text_span: str,
    *,
    start: int | None,
    end: int | None,
) -> dict[str, Any]:
    units = _ocr_units_for_span(ocr_result, start=start, end=end)
    source = "missing"
    quality = "missing"
    if region is not None:
        instance_source = _matching_text_instance_source(ocr_result, region)
        if instance_source:
            source = instance_source
            quality = "high" if "rec_poly" in instance_source or "char" in instance_source else "medium"
        else:
            has_char_polys = any(bool(unit.get("has_char_polys")) for unit in units)
            if has_char_polys:
                source = "char_polys"
                quality = "high"
            elif units and all(str(unit.get("unit_kind")) == "ocr_line" for unit in units):
                source = "ocr_estimated_line_weighted_span"
                quality = "low"
            elif units:
                source = "ocr_block_weighted_span"
                quality = "medium"
            else:
                source = "text_match_weighted_span" if text_span else "ocr_region"
                quality = "medium"
    return {
        "ocr_unit_ids": [str(unit.get("block_id")) for unit in units if unit.get("block_id")],
        "ocr_unit_texts": [str(unit.get("text")) for unit in units if unit.get("text")],
        "ocr_region_source": source,
        "ocr_region_quality": quality,
        "region_source": source if region is not None else "",
        "requires_manual_region_review": region is None,
    }


def _ocr_units_for_span(ocr_result: OCRLayoutResult, *, start: int | None, end: int | None) -> list[dict[str, Any]]:
    if start is None:
        return []
    span_end = end if end is not None and end >= start else start + 1
    units: list[dict[str, Any]] = []
    for item in _build_ocr_block_spans(ocr_result):
        block_start = int(item["start"])
        block_end = int(item["end"])
        if max(start, block_start) >= min(span_end, block_end):
            continue
        copied = dict(item)
        block_index = _optional_int(copied.get("ocr_block_index"))
        if block_index is not None and 0 <= block_index < len(ocr_result.text_blocks):
            source_block = ocr_result.text_blocks[block_index]
            char_polys = (source_block.metadata or {}).get("char_polys") or (source_block.metadata or {}).get("char_boxes")
            copied["has_char_polys"] = isinstance(char_polys, list) and bool(char_polys)
        units.append(copied)
    return units


def _map_text_to_region(
    text_span: str,
    ocr_result: OCRLayoutResult,
    *,
    start: int | None = None,
    end: int | None = None,
) -> RegionMask | None:
    if text_span:
        region = _map_text_to_spotting_instance(text_span, ocr_result)
        if region is not None:
            return region
    if start is not None:
        region = _map_char_span_to_region(start, end, ocr_result, text_span=text_span or "")
        if region is not None:
            return region
    if not text_span:
        return None
    text_norm = _normalize_text_for_match(text_span)
    if not text_norm:
        return None

    candidates: list[tuple[OCRTextBlock, float]] = []
    for block in ocr_result.text_blocks:
        block_norm = _normalize_text_for_match(block.text)
        if text_span in block.text or block.text in text_span or text_norm in block_norm or block_norm in text_norm:
            candidates.append((block, _text_region_quality(text_norm, block)))
    if candidates:
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
        for block, _score in ranked:
            local = _find_local_text_span(block.text, text_span)
            if local is not None:
                region = _region_from_block_char_range(block, local[0], local[1], "text_match")
                if _is_reasonable_text_region(region.bbox, ocr_result, text_span):
                    return region
                continue
            if _is_reasonable_text_region(block.bbox, ocr_result, text_span):
                return _region_from_block(block, "text_match_full_block")

    matched_blocks = [
        block for block in ocr_result.text_blocks
        if _text_similarity(text_norm, _normalize_text_for_match(block.text)) >= 0.56
        and _is_reasonable_text_region(block.bbox, ocr_result, text_span)
    ]
    if matched_blocks:
        if len(matched_blocks) == 1:
            block = matched_blocks[0]
            return _region_from_block(block, "fuzzy_text_match_full_block")
        union = _union_blocks(matched_blocks)
        if _is_reasonable_text_region(union.bbox, ocr_result, text_span):
            return _pad_text_region(union, "fuzzy_text_match_union")

    return None


def _map_text_to_spotting_instance(text_span: str, ocr_result: OCRLayoutResult) -> RegionMask | None:
    text_norm = _normalize_text_for_match(text_span)
    if not text_norm:
        return None
    candidates: list[tuple[OCRTextBlock, float]] = []
    for block in _ocr_text_instance_blocks(ocr_result):
        block_norm = _normalize_text_for_match(block.text)
        if not block_norm:
            continue
        if text_span in block.text or block.text in text_span or text_norm in block_norm or block_norm in text_norm:
            candidates.append((block, _text_region_quality(text_norm, block) + _spotting_source_bonus(block)))
    if not candidates:
        candidates = [
            (block, _text_similarity(text_norm, _normalize_text_for_match(block.text)) + _spotting_source_bonus(block))
            for block in _ocr_text_instance_blocks(ocr_result)
            if _text_similarity(text_norm, _normalize_text_for_match(block.text)) >= 0.72
        ]
    for block, _score in sorted(candidates, key=lambda item: item[1], reverse=True):
        local = _find_local_text_span(block.text, text_span)
        if local is not None:
            return _region_from_block_char_range(block, local[0], local[1], "paddleocr_spotting")
        if _normalize_text_for_match(block.text) == text_norm or text_norm in _normalize_text_for_match(block.text):
            return _region_from_block(block, "paddleocr_spotting_full_instance")
    return None


def _ocr_text_instance_blocks(ocr_result: OCRLayoutResult) -> list[OCRTextBlock]:
    blocks: list[OCRTextBlock] = []
    for item in _ocr_text_instances(ocr_result):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        bbox_value = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        if not text or not bbox_value:
            continue
        try:
            bbox = BBox(
                x=float(bbox_value.get("x", 0)),
                y=float(bbox_value.get("y", 0)),
                w=float(bbox_value.get("w", 0)),
                h=float(bbox_value.get("h", 0)),
            )
        except (TypeError, ValueError):
            continue
        blocks.append(
            OCRTextBlock(
                text=text,
                bbox=bbox,
                polygon=_polygon_from_metadata(item.get("polygon")),
                confidence=float(item.get("confidence") or 0.0),
                metadata={
                    "unit_kind": "ocr_text_instance",
                    "ocr_unit_id": item.get("unit_id"),
                    "source": item.get("source") or "paddleocr_spotting",
                    "quality": item.get("quality") or "high",
                    "char_polys": item.get("char_polys") or [],
                    "reading_order": item.get("reading_order"),
                    "page_index": item.get("page_index"),
                },
            )
        )
    return blocks


def _ocr_text_instances(ocr_result: OCRLayoutResult) -> list[dict[str, Any]]:
    value = (ocr_result.metadata or {}).get("text_instances")
    return value if isinstance(value, list) else []


def _spotting_source_bonus(block: OCRTextBlock) -> float:
    source = str((block.metadata or {}).get("source") or "")
    if "char" in source:
        return 2.0
    if "rec_poly" in source:
        return 1.5
    if "spotting" in source:
        return 1.0
    return 0.0


def _matching_text_instance_source(ocr_result: OCRLayoutResult, region: RegionMask) -> str:
    rb = region.bbox
    for item in _ocr_text_instances(ocr_result):
        if not isinstance(item, dict):
            continue
        bbox_value = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
        try:
            bbox = BBox(
                x=float(bbox_value.get("x", 0)),
                y=float(bbox_value.get("y", 0)),
                w=float(bbox_value.get("w", 0)),
                h=float(bbox_value.get("h", 0)),
            )
        except (TypeError, ValueError):
            continue
        if _bbox_iou(rb, bbox) >= 0.65 or _bbox_containment(rb, bbox) >= 0.85 or _bbox_containment(bbox, rb) >= 0.85:
            return str(item.get("source") or "paddleocr_spotting")
    return ""


def _expand_address_region(
    text_span: str,
    ocr_result: OCRLayoutResult,
    *,
    start: int | None,
    end: int | None,
) -> RegionMask | None:
    units = _ocr_units_for_span(ocr_result, start=start, end=end)
    if not units:
        return None
    all_units = _build_ocr_block_spans(ocr_result)
    selected_ids = {str(unit.get("block_id")) for unit in units}
    anchor_indices = [
        index for index, unit in enumerate(all_units)
        if str(unit.get("block_id")) in selected_ids
    ]
    if not anchor_indices:
        return None
    selected: list[dict[str, Any]] = []
    for anchor_index in anchor_indices:
        anchor = all_units[anchor_index]
        if _looks_like_address_line(str(anchor.get("text") or "")):
            selected.append(anchor)
        for direction in (-1, 1):
            index = anchor_index + direction
            while 0 <= index < len(all_units):
                candidate = all_units[index]
                text = str(candidate.get("text") or "")
                if _address_group_stop_line(text):
                    break
                if not _same_address_column(anchor, candidate):
                    break
                if not _looks_like_address_line(text):
                    break
                selected.append(candidate)
                index += direction
    if not selected:
        return None
    return _region_from_ocr_span_items(_dedupe_span_items(selected), "ocr_address_group")


def _address_recall_findings(
    document: dict[str, Any],
    ocr_result: OCRLayoutResult,
    seen: set[str],
) -> list[PictureFinding]:
    if not _document_mentions_address_risk(document):
        return []
    groups = _address_candidate_groups(_build_ocr_block_spans(ocr_result))
    findings: list[PictureFinding] = []
    for index, group in enumerate(groups):
        text = "\n".join(str(item.get("text") or "").strip() for item in group if str(item.get("text") or "").strip())
        if not text:
            continue
        dedupe_key = f"{FindingType.TEXT_PII.value}:address:{text}:ocr_address_recall"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        region = _region_from_ocr_span_items(group, "ocr_address_recall")
        if region is None:
            continue
        findings.append(
            PictureFinding(
                finding_type=FindingType.TEXT_PII,
                category="address",
                label="地址位置检测",
                score=0.72,
                region=region,
                text_span=text,
                reason_code="OCR_TEXT_PII_ADDRESS",
                provider="ocr_address_recall",
                explanation="文本合规文档级结果提示存在地址风险，OCR 结构化地址召回定位到疑似完整地址块，需要脱敏。",
                metadata={
                    "source_modality": "image_ocr",
                    "source_kind": "ocr_address_recall",
                    "char_start": min(int(item.get("start", 0)) for item in group),
                    "char_end": max(int(item.get("end", 0)) for item in group),
                    "region_source": "ocr_address_recall",
                    "region_missing": False,
                    "requires_manual_region_review": False,
                    "ocr_unit_ids": [str(item.get("block_id")) for item in group],
                    "ocr_unit_texts": [str(item.get("text") or "") for item in group],
                    "ocr_region_source": "ocr_address_recall",
                    "ocr_region_quality": "medium",
                    "recalled_from_document_level_address_assessment": True,
                    "address_group_index": index,
                },
            )
        )
    return findings


def _document_mentions_address_risk(document: dict[str, Any]) -> bool:
    text = json.dumps(document, ensure_ascii=False).lower()
    return any(token in text for token in ("address", "addresses", "地址", "business addresses", "位置"))


def _address_candidate_groups(units: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for unit in units:
        text = str(unit.get("text") or "")
        if _address_group_stop_line(text):
            if len(current) >= 1:
                groups.append(current)
            current = []
            continue
        if _looks_like_address_line(text):
            if current and not _same_address_column(current[-1], unit):
                groups.append(current)
                current = []
            current.append(unit)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return [group for group in groups if _address_group_score(group) >= 2]


def _address_group_score(group: list[dict[str, Any]]) -> int:
    text = " ".join(str(item.get("text") or "") for item in group)
    score = 0
    if re.search(r"\b\d{2,6}\b", text):
        score += 1
    if re.search(r"\b(?:apt|suite|unit|box|street|st\\.?|road|rd\\.?|ave\\.?|avenue|cove|prairie|summit|trafficway|hills)\b", text, re.I):
        score += 1
    if re.search(r"\b[A-Z]{2}\s+\d{4,6}\b", text):
        score += 1
    if any("," in str(item.get("text") or "") for item in group):
        score += 1
    return score


def _looks_like_address_line(text: str) -> bool:
    value = str(text or "").strip()
    if not value or len(value) > 100:
        return False
    if _address_group_stop_line(value):
        return False
    patterns = (
        r"\b\d{2,6}\s+[A-Za-z][A-Za-z0-9 .'-]*(?:street|st\.?|road|rd\.?|avenue|ave\.?|cove|prairie|summit|trafficway|hills|apt\.?|suite|unit|box)\b",
        r"\b(?:apt\.?|suite|unit|box)\s*\d+\b",
        r"\b[A-Za-z][A-Za-z .'-]+,\s*[A-Z]{2}\s+\d{4,6}\b",
        r"\bDPO\s+[A-Z]{2}\s+\d{4,6}\b",
    )
    return any(re.search(pattern, value, re.I) for pattern in patterns)


def _address_group_stop_line(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    return bool(re.match(r"^(tax id|iban|items|summary|seller:|client:|invoice no|date of issue|<table)", value))


def _same_address_column(a: dict[str, Any], b: dict[str, Any]) -> bool:
    a_box = _bbox_from_record(a)
    b_box = _bbox_from_record(b)
    if a_box is None or b_box is None:
        return True
    x_close = abs(float(a_box.x) - float(b_box.x)) <= max(80.0, min(float(a_box.w), float(b_box.w)) * 0.35)
    overlap = _horizontal_overlap_ratio(a_box, b_box) >= 0.45
    y_gap = max(float(b_box.y) - float(a_box.y + a_box.h), float(a_box.y) - float(b_box.y + b_box.h), 0.0)
    return (x_close or overlap) and y_gap <= max(60.0, max(float(a_box.h), float(b_box.h)) * 1.5)


def _region_from_ocr_span_items(items: list[dict[str, Any]], source: str) -> RegionMask | None:
    blocks: list[OCRTextBlock] = []
    for item in items:
        bbox = _bbox_from_record(item)
        if bbox is None:
            continue
        blocks.append(
            OCRTextBlock(
                text=str(item.get("text") or ""),
                bbox=bbox,
                polygon=_polygon_from_metadata(item.get("polygon")),
                confidence=float(item.get("confidence") or 0.0),
                metadata={"source": source},
            )
        )
    if not blocks:
        return None
    if len(blocks) == 1:
        return _region_from_block(blocks[0], source)
    return _pad_text_region(_union_blocks(blocks), source)


def _bbox_from_record(item: dict[str, Any]) -> BBox | None:
    bbox_value = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
    try:
        return BBox(
            x=float(bbox_value.get("x", 0)),
            y=float(bbox_value.get("y", 0)),
            w=float(bbox_value.get("w", 0)),
            h=float(bbox_value.get("h", 0)),
        )
    except (TypeError, ValueError):
        return None


def _dedupe_span_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in sorted(items, key=lambda value: (int(value.get("start", 0)), str(value.get("block_id") or ""))):
        key = str(item.get("block_id") or item.get("text") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _ocr_units_overlapping_region(ocr_result: OCRLayoutResult, region: RegionMask) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for item in _build_ocr_block_spans(ocr_result):
        bbox = _bbox_from_record(item)
        if bbox is None:
            continue
        if _bbox_iou(region.bbox, bbox) > 0.05 or _bbox_containment(bbox, region.bbox) >= 0.75:
            units.append(item)
    return units


def _map_char_span_to_region(
    start: int | None,
    end: int | None,
    ocr_result: OCRLayoutResult,
    *,
    text_span: str = "",
) -> RegionMask | None:
    if start is None:
        return None
    block_spans = _build_ocr_block_spans(ocr_result)
    if not block_spans:
        return None
    span_end = end if end is not None and end >= start else start + 1
    matched: list[tuple[OCRTextBlock, int, int]] = []
    for item in block_spans:
        block_start = int(item["start"])
        block_end = int(item["end"])
        overlap_start = max(start, block_start)
        overlap_end = min(span_end, block_end)
        if overlap_start < overlap_end:
            block_index = _optional_int(item.get("ocr_block_index"))
            if block_index is None:
                block_index = int(str(item["block_id"]).split("_", 2)[-1].split("_", 1)[0]) - 1
            if 0 <= block_index < len(ocr_result.text_blocks):
                source_block = ocr_result.text_blocks[block_index]
                block = _ocr_span_item_to_block(item, source_block)
                matched.append((block, overlap_start - block_start, overlap_end - block_start))
    if matched:
        if len(matched) == 1:
            block, local_start, local_end = matched[0]
            region = _region_from_block_char_range(block, local_start, local_end, "char_span_map")
            if _is_reasonable_text_region(region.bbox, ocr_result, text_span or block.text):
                return region
            if local_start <= 0 and local_end >= len(block.text or ""):
                return None
            return region
        union = _union_blocks([block for block, _, _ in matched])
        if _is_reasonable_text_region(union.bbox, ocr_result, text_span):
            return _pad_text_region(union, "char_span_multi_block_union")
    return None


def _region_from_block(block: OCRTextBlock, source: str) -> RegionMask:
    return _pad_text_region(
        RegionMask(bbox=block.bbox, polygon=block.polygon, confidence=block.confidence),
        source,
    )


def _ocr_span_item_to_block(item: dict[str, Any], source_block: OCRTextBlock) -> OCRTextBlock:
    bbox_value = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
    bbox = BBox(
        x=float(bbox_value.get("x", source_block.bbox.x)),
        y=float(bbox_value.get("y", source_block.bbox.y)),
        w=float(bbox_value.get("w", source_block.bbox.w)),
        h=float(bbox_value.get("h", source_block.bbox.h)),
    )
    polygon = _polygon_from_metadata(item.get("polygon"))
    source_start = _optional_int(item.get("source_text_start")) or 0
    source_end = _optional_int(item.get("source_text_end")) or source_start + len(str(item.get("text") or ""))
    visible_start = _optional_int(item.get("source_visible_start"))
    visible_end = _optional_int(item.get("source_visible_end"))
    char_polys = (source_block.metadata or {}).get("char_polys") or (source_block.metadata or {}).get("char_boxes")
    if isinstance(char_polys, list):
        if visible_start is not None and visible_end is not None:
            char_polys = char_polys[visible_start:visible_end]
        else:
            char_polys = char_polys[source_start:source_end]
    else:
        char_polys = []
    return OCRTextBlock(
        text=str(item.get("text") or ""),
        bbox=bbox,
        polygon=polygon,
        confidence=float(item.get("confidence") or source_block.confidence),
        language=source_block.language,
        metadata={
            **dict(source_block.metadata or {}),
            "char_polys": char_polys,
            "ocr_unit_id": item.get("block_id"),
            "unit_kind": item.get("unit_kind") or "ocr_block",
            "source_text_start": source_start,
            "source_text_end": source_end,
        },
    )


def _region_from_block_char_range(
    block: OCRTextBlock,
    local_start: int,
    local_end: int,
    source: str,
) -> RegionMask:
    text = block.text or ""
    if not text:
        return _region_from_block(block, f"{source}_empty_block")
    local_start = max(0, min(len(text), local_start))
    local_end = max(local_start + 1, min(len(text), local_end))

    char_region = _region_from_explicit_char_polys(block, local_start, local_end)
    if char_region is not None:
        return _pad_text_region(char_region, f"{source}_char_polys")

    if local_start <= 0 and local_end >= len(text):
        return _region_from_block(block, f"{source}_full_block")

    bbox = _estimate_sub_bbox_by_weighted_chars(block.bbox, text, local_start, local_end)
    polygon = _sub_polygon_from_bbox(bbox)
    return _pad_text_region(
        RegionMask(bbox=bbox, polygon=polygon, confidence=max(0.01, block.confidence * 0.82)),
        f"{source}_weighted_estimate",
    )


def _region_from_explicit_char_polys(
    block: OCRTextBlock,
    local_start: int,
    local_end: int,
) -> RegionMask | None:
    char_polys = (block.metadata or {}).get("char_polys") or (block.metadata or {}).get("char_boxes")
    if not isinstance(char_polys, list):
        return None
    selected: list[Polygon] = []
    for item in char_polys[local_start:local_end]:
        polygon = _polygon_from_metadata(item)
        if polygon is not None:
            selected.append(polygon)
    if not selected:
        return None
    points = [point for polygon in selected for point in polygon.points]
    bbox = _bbox_from_points(points)
    return RegionMask(bbox=bbox, polygon=Polygon(points=points), confidence=block.confidence)


def _polygon_from_metadata(value: Any) -> Polygon | None:
    if isinstance(value, dict):
        for key in ("polygon", "poly", "points", "bbox", "box"):
            polygon = _polygon_from_metadata(value.get(key))
            if polygon is not None:
                return polygon
        return None
    if not isinstance(value, list) or not value:
        return None
    points: list[tuple[float, float]] = []
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        x1, y1, x2, y2 = [float(item) for item in value]
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        for point in value:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
    return Polygon(points=points) if len(points) >= 3 else None


def _estimate_sub_bbox_by_weighted_chars(
    bbox: BBox,
    text: str,
    local_start: int,
    local_end: int,
) -> BBox:
    weights = [_char_visual_weight(char) for char in text]
    total = max(0.0001, sum(weights))
    left_ratio = sum(weights[:local_start]) / total
    right_ratio = sum(weights[:local_end]) / total
    x = float(bbox.x) + float(bbox.w) * left_ratio
    right = float(bbox.x) + float(bbox.w) * right_ratio
    return BBox(x=x, y=float(bbox.y), w=max(1.0, right - x), h=max(1.0, float(bbox.h)))


def _char_visual_weight(char: str) -> float:
    if not char:
        return 0.0
    codepoint = ord(char)
    if char.isspace():
        return 0.35
    if char.isdigit():
        return 0.62
    if ("a" <= char.lower() <= "z"):
        return 0.58
    if 0x4E00 <= codepoint <= 0x9FFF:
        return 1.0
    if re.match(r"\W", char, flags=re.UNICODE):
        return 0.36
    return 0.75


def _find_local_text_span(block_text: str, text_span: str) -> tuple[int, int] | None:
    if not block_text or not text_span:
        return None
    direct = block_text.find(text_span)
    if direct >= 0:
        return direct, direct + len(text_span)
    normalized_target = _normalize_text_for_match(text_span)
    if not normalized_target:
        return None
    normalized_chars: list[tuple[str, int]] = []
    for index, char in enumerate(block_text):
        normalized = _normalize_text_for_match(char)
        if normalized:
            normalized_chars.append((normalized, index))
    normalized_block = "".join(char for char, _ in normalized_chars)
    pos = normalized_block.find(normalized_target)
    if pos < 0:
        return None
    start = normalized_chars[pos][1]
    end = normalized_chars[min(len(normalized_chars) - 1, pos + len(normalized_target) - 1)][1] + 1
    return start, end


def _pad_text_region(region: RegionMask, source: str) -> RegionMask:
    bbox = region.bbox
    pad = max(3.0, min(16.0, float(bbox.h) * 0.08))
    padded = BBox(
        x=float(bbox.x) - pad,
        y=float(bbox.y) - pad,
        w=max(1.0, float(bbox.w) + 2.0 * pad),
        h=max(1.0, float(bbox.h) + 2.0 * pad),
    )
    polygon = region.polygon
    if polygon is not None and polygon.points:
        cx = sum(point[0] for point in polygon.points) / len(polygon.points)
        cy = sum(point[1] for point in polygon.points) / len(polygon.points)
        scale_x = (padded.w / max(1.0, float(bbox.w)))
        scale_y = (padded.h / max(1.0, float(bbox.h)))
        polygon = Polygon(
            points=[
                (
                    cx + (float(point[0]) - cx) * scale_x,
                    cy + (float(point[1]) - cy) * scale_y,
                )
                for point in polygon.points
            ]
        )
    return region.model_copy(update={"bbox": padded, "polygon": polygon, "confidence": region.confidence})


def _sub_polygon_from_bbox(bbox: BBox) -> Polygon:
    return Polygon(
        points=[
            (float(bbox.x), float(bbox.y)),
            (float(bbox.x + bbox.w), float(bbox.y)),
            (float(bbox.x + bbox.w), float(bbox.y + bbox.h)),
            (float(bbox.x), float(bbox.y + bbox.h)),
        ]
    )


def _bbox_from_points(points: list[tuple[float, float]]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return BBox(x=min(xs), y=min(ys), w=max(1.0, max(xs) - min(xs)), h=max(1.0, max(ys) - min(ys)))


def _text_region_quality(text_norm: str, block: OCRTextBlock) -> float:
    block_norm = _normalize_text_for_match(block.text)
    if not block_norm:
        return 0.0
    area = max(1.0, float(block.bbox.w) * float(block.bbox.h))
    length_ratio = min(len(text_norm), len(block_norm)) / max(len(text_norm), len(block_norm), 1)
    exact_bonus = 1.0 if text_norm == block_norm else 0.5 if text_norm in block_norm else 0.0
    return exact_bonus + length_ratio + float(block.confidence) - min(area / 1_000_000.0, 1.0)


def _is_reasonable_text_region(
    bbox: BBox,
    ocr_result: OCRLayoutResult,
    text_span: str,
) -> bool:
    image_area = _estimated_image_area(ocr_result)
    if image_area <= 0:
        return True
    area = max(1.0, float(bbox.w) * float(bbox.h))
    ratio = area / image_area
    text_len = len(_normalize_text_for_match(text_span))
    max_ratio = 0.08 if 0 < text_len <= 40 else 0.20
    if ratio > max_ratio:
        return False
    return True


def _estimated_image_area(ocr_result: OCRLayoutResult) -> float:
    metadata = dict(ocr_result.metadata or {})
    width = float(metadata.get("image_width") or metadata.get("width") or 0)
    height = float(metadata.get("image_height") or metadata.get("height") or 0)
    if width > 0 and height > 0:
        return width * height
    right = 0.0
    bottom = 0.0
    for block in ocr_result.text_blocks:
        right = max(right, float(block.bbox.x) + float(block.bbox.w))
        bottom = max(bottom, float(block.bbox.y) + float(block.bbox.h))
    return right * bottom if right > 0 and bottom > 0 else 0.0


def _normalize_privacy_category(value: str, item: dict[str, Any]) -> str:
    raw = _normalize_category(value or "")
    policy_tag = _normalize_category(str(item.get("policy_tag") or ""))
    if raw in {"bank_card", "bank_account", "financial_account"} or "bank_account" in policy_tag:
        return "bank_account"
    if raw in {"id", "id_number", "id_card", "identity", "tax_id"}:
        return "id_card"
    if raw in {"phone", "mobile", "telephone"}:
        return "phone_number"
    if raw in {"pii_presidio_us_itin", "presidio_us_itin"}:
        return "id_card"
    if "address" in policy_tag:
        return "address"
    if "person_name" in policy_tag:
        return "person_name"
    if "date_time" in policy_tag:
        return "date_time"
    return raw or "pii_entity"


def _privacy_label(item: dict[str, Any], category: str) -> str:
    label = _first_text(item, "operator_name_zh", "label_zh")
    if label and not re.fullmatch(r"[A-Za-z0-9_.:-]+", label):
        return label
    mapping = {
        "person_name": "姓名信息",
        "phone_number": "电话号码",
        "email": "邮箱地址",
        "address": "地址位置",
        "id_card": "身份证件/税号信息",
        "bank_account": "金融账户信息",
        "bank_card": "金融账户信息",
        "license_plate": "车牌号码",
        "student_id": "学号信息",
        "date_time": "日期时间信息",
        "combined_identity": "组合可识别风险",
        "pii_entity": "个人信息片段",
    }
    return mapping.get(category, "OCR文字隐私信息")


def _content_label(item: dict[str, Any], category: str) -> str:
    label = _first_text(item, "operator_name_zh", "label_zh")
    if label and not re.fullmatch(r"[A-Za-z0-9_.:-]+", label):
        return label
    mapping = {
        "violence": "暴力危险内容",
        "sexual_content": "色情低俗内容",
        "self_harm": "自伤自杀内容",
        "hate_speech": "仇恨歧视内容",
        "illegal_instruction": "违法违规指引",
        "content_safety": "文字内容安全风险",
    }
    return mapping.get(category, "OCR文字内容安全风险")


def _effective_privacy_score(item: dict[str, Any]) -> float:
    confidence = float(item.get("confidence") or item.get("score") or 0.0)
    sensitivity = str(item.get("sensitivity_level") or "").upper()
    action = str(item.get("privacy_action") or item.get("action") or "").lower()
    route = str(item.get("dataset_route") or "").lower()
    if sensitivity in {"S4", "S5"} or action in {"drop_or_manual_review", "restricted_review"}:
        confidence = max(confidence, 0.92)
    elif sensitivity == "S3" or "redaction" in route or action in {"mask", "generalize"}:
        confidence = max(confidence, 0.78)
    elif confidence <= 0:
        confidence = 0.65
    return min(confidence, 1.0)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dedupe_picture_findings(findings: list[PictureFinding]) -> list[PictureFinding]:
    result: list[PictureFinding] = []
    seen: set[str] = set()
    for finding in findings:
        key = "|".join(
            [
                finding.finding_type.value,
                finding.category,
                finding.text_span or "",
                str((finding.metadata or {}).get("char_start") or ""),
                str((finding.metadata or {}).get("char_end") or ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


_PRIVACY_CATEGORY_HINTS = {
    "name",
    "person_name",
    "phone",
    "phone_number",
    "mobile",
    "email",
    "address",
    "id_card",
    "identity",
    "bank_card",
    "bank_account",
    "account",
    "license_plate",
    "student_id",
}

_CONTENT_CATEGORY_HINTS = {
    "explicit",
    "porn",
    "adult",
    "violence",
    "graphic_violence",
    "dangerous",
    "illegal",
    "hate",
    "self_harm",
    "other_nsfw",
    "content_safety",
}

_FINDING_LIST_KEYS = {
    "findings",
    "safety_findings",
    "privacy_findings",
    "content_safety_findings",
    "evidence_findings",
    "supplemental_findings",
    "redaction_targets",
    "span_annotations",
    "review_tasks",
}


def _collect_text_compliance_findings(document: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            marker = id(value)
            if marker in seen_ids:
                return
            seen_ids.add(marker)
            if _looks_like_finding(value):
                candidates.append(value)
            for key, child in value.items():
                if key in _FINDING_LIST_KEYS or isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(document)
    return candidates


def _looks_like_finding(item: dict[str, Any]) -> bool:
    keys = set(item)
    if keys & {"finding_type", "risk_type", "policy_tag", "matched_text", "evidence_text", "span", "source_tool"}:
        return True
    if (keys & {"category", "label", "operator_id"}) and (keys & {"text", "snippet", "evidence", "confidence", "score"}):
        return True
    return False


def _first_text(mapping: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                return text
    return default


def _normalize_text_for_match(value: str) -> str:
    return re.sub(r"\W+", "", value, flags=re.UNICODE).lower()


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    a_tokens = set(_ngrams(a, 2))
    b_tokens = set(_ngrams(b, 2))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _ngrams(value: str, size: int) -> list[str]:
    if len(value) <= size:
        return [value]
    return [value[index : index + size] for index in range(len(value) - size + 1)]


def _union_blocks(blocks: list[OCRTextBlock]) -> RegionMask:
    left = min(block.bbox.x for block in blocks)
    top = min(block.bbox.y for block in blocks)
    right = max(block.bbox.x + block.bbox.w for block in blocks)
    bottom = max(block.bbox.y + block.bbox.h for block in blocks)
    confidence = sum(float(block.confidence) for block in blocks) / len(blocks)
    return RegionMask(
        bbox=BBox(x=left, y=top, w=max(1.0, right - left), h=max(1.0, bottom - top)),
        confidence=confidence,
    )


def _bbox_iou(a: BBox, b: BBox) -> float:
    left = max(float(a.x), float(b.x))
    top = max(float(a.y), float(b.y))
    right = min(float(a.x + a.w), float(b.x + b.w))
    bottom = min(float(a.y + a.h), float(b.y + b.h))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, float(a.w) * float(a.h))
    area_b = max(1.0, float(b.w) * float(b.h))
    return inter / max(1.0, area_a + area_b - inter)


def _bbox_containment(inner: BBox, outer: BBox) -> float:
    left = max(float(inner.x), float(outer.x))
    top = max(float(inner.y), float(outer.y))
    right = min(float(inner.x + inner.w), float(outer.x + outer.w))
    bottom = min(float(inner.y + inner.h), float(outer.y + outer.h))
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    return inter / max(1.0, float(inner.w) * float(inner.h))


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    left = max(float(a.x), float(b.x))
    right = min(float(a.x + a.w), float(b.x + b.w))
    overlap = max(0.0, right - left)
    return overlap / max(1.0, min(float(a.w), float(b.w)))


def _default_explanation(
    finding_type: FindingType,
    risk_type: str,
    text_span: str,
    has_region: bool,
) -> str:
    target = f"“{text_span}”" if text_span else "OCR 识别文本"
    if finding_type == FindingType.TEXT_PII:
        reason = f"OCR 文本中检测到疑似{risk_type}隐私信息 {target}，需要在图片中进行遮蔽处理。"
    else:
        reason = f"OCR 文本中检测到疑似{risk_type}内容安全风险 {target}，需要按文本合规结果处置。"
    if not has_region:
        reason += " 当前未能自动定位到精确图片区域，需要人工复核补充区域。"
    return reason

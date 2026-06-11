"""Build non-destructive video risk annotations from existing modality engines."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from picture.domain.models import BBox, PictureFinding, PictureJob
from video.application.operator_selection import ResolvedOperatorSelection
from video.domain.models import FrameReference, TimeSpan, VideoRiskAnnotation
from video.domain.operators import CSA_LABELS, PII_TARGETS
from video.domain.taxonomy import (
    map_moderation_category,
    map_picture_finding_category,
    recommended_actions_for_category,
    severity_for_category,
)


def build_frame_risks(
    frames: list[FrameReference],
    frame_jobs: list[PictureJob],
    asset_id: str = "",
    operator_selection: ResolvedOperatorSelection | None = None,
) -> list[VideoRiskAnnotation]:
    """Convert frame-level picture jobs into timeline-anchored risk annotations."""
    risks: list[VideoRiskAnnotation] = []
    for frame, frame_job in zip(frames, frame_jobs):
        span = _frame_span(frame)
        if frame_job.moderation_result and not frame_job.moderation_result.is_safe:
            for reason_code in frame_job.moderation_result.reason_codes or ["VISUAL_SAFETY"]:
                category = map_moderation_category(frame_job.moderation_result, reason_code)
                confidence = max(frame_job.moderation_result.scores.values(), default=0.0)
                source_operator_id = _source_operator_for_category(category)
                risk = VideoRiskAnnotation(
                    asset_id=asset_id,
                    source_modality="safety",
                    category=category,
                    operator_id=_video_operator_id("visual_safety", source_operator_id),
                    source_operator_id=source_operator_id,
                    target_type=category,
                    severity=severity_for_category(category, confidence, frame_job.moderation_result.metadata),
                    confidence=confidence,
                    span=span,
                    frame_ids=[frame.frame_id],
                    evidence_refs=[frame_job.job_id],
                    provider=frame_job.moderation_result.provider,
                    reason_codes=[reason_code],
                    recommended_actions=recommended_actions_for_category(category),
                    metadata={
                        "picture_job_id": frame_job.job_id,
                        "scores": frame_job.moderation_result.scores,
                        **dict(frame_job.moderation_result.metadata or {}),
                    },
                )
                if _risk_allowed(risk, operator_selection):
                    risks.append(risk)
        for finding in frame_job.findings:
            risk = _risk_from_picture_finding(frame, frame_job, finding, asset_id)
            if _should_drop_frame_risk(risk):
                continue
            if _risk_allowed(risk, operator_selection):
                risks.append(risk)
    return risks


def build_audio_risks(
    audio_output_dir: Path | None,
    asset_id: str = "",
    operator_selection: ResolvedOperatorSelection | None = None,
) -> list[VideoRiskAnnotation]:
    """Build coarse audio risks from audio pipeline artifacts when available."""
    if audio_output_dir is None:
        return []
    risks: list[VideoRiskAnnotation] = []
    bridge_report = audio_output_dir / "32_audio_compliance_report.json"
    if bridge_report.exists():
        report = _read_json(bridge_report)
        findings = report.get("display_findings") or report.get("findings") or []
        for index, record in enumerate(findings if isinstance(findings, list) else []):
            if not isinstance(record, dict) or record.get("display_suppressed"):
                continue
            category = _audio_bridge_category(record)
            source_operator_id = _source_operator_for_category(category)
            confidence = _record_confidence(record)
            start_ms, end_ms = _audio_bridge_span(record)
            risk = VideoRiskAnnotation(
                asset_id=asset_id,
                source_modality="audio",
                category=category,
                operator_id=_video_operator_id("audio", source_operator_id),
                source_operator_id=source_operator_id,
                target_type=_target_type_for_category(category),
                severity=str(record.get("risk_level") or record.get("severity") or severity_for_category(category, confidence, record)),
                confidence=confidence,
                span=TimeSpan(start_ms=start_ms, end_ms=end_ms),
                audio_segment={
                    "source": "32_audio_compliance_report.json",
                    "record_index": index,
                    "text": str(record.get("text") or record.get("display_text") or ""),
                    "time_label": str(record.get("time_label") or ""),
                },
                text_span=str(record.get("text") or record.get("display_text") or ""),
                evidence_refs=[str(bridge_report), str(record.get("finding_id") or "")],
                provider=str(record.get("source_tool") or "audio_text_api_bridge"),
                provider_version=str(report.get("execution_engine") or ""),
                reason_codes=_record_reason_codes(record),
                recommended_actions=recommended_actions_for_category(category),
                metadata=record,
            )
            if _risk_allowed(risk, operator_selection):
                risks.append(risk)
        return risks

    for path in (
        audio_output_dir / "08_privacy_detection.jsonl",
        audio_output_dir / "09_content_safety.jsonl",
        audio_output_dir / "09b_hard_case_adjudication.jsonl",
    ):
        if not path.exists():
            continue
        for index, record in enumerate(_read_jsonl(path)):
            category = _audio_category(record, path.name)
            source_operator_id = _source_operator_for_category(category)
            confidence = _record_confidence(record)
            start_ms, end_ms = _record_span(record)
            risk = VideoRiskAnnotation(
                asset_id=asset_id,
                source_modality="audio",
                category=category,
                operator_id=_video_operator_id("audio", source_operator_id),
                source_operator_id=source_operator_id,
                target_type=_target_type_for_category(category),
                severity=severity_for_category(category, confidence, record),
                confidence=confidence,
                span=TimeSpan(start_ms=start_ms, end_ms=end_ms),
                audio_segment={"source": path.name, "record_index": index},
                evidence_refs=[str(path)],
                provider=str(record.get("provider_name") or record.get("provider") or "audio_pipeline"),
                provider_version=str(record.get("provider_version") or ""),
                reason_codes=_record_reason_codes(record),
                recommended_actions=recommended_actions_for_category(category),
                metadata=record,
            )
            if _risk_allowed(risk, operator_selection):
                risks.append(risk)
    return risks


def _audio_bridge_category(record: dict[str, Any]) -> str:
    policy_tag = str(record.get("policy_tag") or "").strip().lower()
    risk_type = str(record.get("risk_type") or record.get("type") or "").strip().lower()
    if policy_tag.startswith("content."):
        return policy_tag
    if policy_tag.startswith("pii."):
        return "privacy." + policy_tag.split(".", 1)[1]
    if risk_type.startswith("content."):
        return risk_type
    if risk_type:
        if risk_type in {"violence", "hate_speech", "harassment", "self_harm", "illegal_instruction", "minor_harmful", "misleading", "values_violation", "jailbreak_attempt"}:
            return "content." + risk_type.replace("hate_speech", "hate")
        return "privacy." + risk_type
    return "audio_pii"


def _audio_bridge_span(record: dict[str, Any]) -> tuple[int, int]:
    start = record.get("start_time")
    end = record.get("end_time")
    try:
        start_ms = int(round(float(start) * 1000))
    except Exception:
        start_ms = 0
    try:
        end_ms = int(round(float(end) * 1000))
    except Exception:
        end_ms = start_ms + 1000
    return start_ms, max(start_ms + 1, end_ms)


def _risk_from_picture_finding(
    frame: FrameReference,
    frame_job: PictureJob,
    finding: PictureFinding,
    asset_id: str,
) -> VideoRiskAnnotation:
    category = map_picture_finding_category(finding)
    metadata = dict(finding.metadata or {})
    source_modality = "ocr_text" if finding.text_span else "visual"
    if str(finding.reason_code or "").startswith("OCR_"):
        source_modality = "ocr_text"
    confidence = float(finding.score or 0.0)
    source_operator_id = str(metadata.get("operator_id") or "").strip().upper()
    if not source_operator_id:
        source_operator_id = _source_operator_for_finding(category, finding)
    target_type = _target_type_for_category(category)
    regions = _regions_from_finding(frame, finding)
    trackable_target_type = _trackable_target_for_finding(category, finding, metadata, regions)
    if trackable_target_type:
        target_type = trackable_target_type
    video_role = _video_role_for_finding(category, source_modality, regions, metadata)
    instance_label_zh = _instance_label_zh(finding, metadata, target_type)
    instance_label_en = str(metadata.get("entity_label_en") or target_type or finding.category or "").strip()
    if trackable_target_type == "conflict_region":
        instance_label_zh = "暴力斗殴区域"
        instance_label_en = "conflict_region"
    return VideoRiskAnnotation(
        asset_id=asset_id,
        source_modality=source_modality,
        category=category,
        operator_id=_video_operator_id(source_modality, source_operator_id),
        source_operator_id=source_operator_id,
        target_type=target_type,
        severity=severity_for_category(category, confidence, metadata),
        confidence=confidence,
        span=_frame_span(frame),
        frame_ids=[frame.frame_id],
        regions=regions,
        text_span=finding.text_span,
        evidence_refs=[finding.finding_id, frame_job.job_id],
        provider=finding.provider,
        provider_version=finding.provider_version,
        reason_codes=[finding.reason_code] if finding.reason_code else [],
        recommended_actions=recommended_actions_for_category(category),
        metadata={
            "picture_job_id": frame_job.job_id,
            "picture_finding_id": finding.finding_id,
            "label": finding.label,
            "explanation": finding.explanation,
            "video_role": video_role,
            "instance_label_zh": instance_label_zh,
            "instance_label_en": instance_label_en,
            "trackable_target_type": trackable_target_type or "",
            "localization_status": str(metadata.get("localization_status") or ""),
            "mask_quality_score": metadata.get("mask_quality_score"),
            **metadata,
        },
    )


def _risk_allowed(risk: VideoRiskAnnotation, operator_selection: ResolvedOperatorSelection | None) -> bool:
    if operator_selection is None:
        return True
    return operator_selection.risk_allowed(
        operator_id=risk.operator_id,
        source_operator_id=risk.source_operator_id,
        target_type=risk.target_type,
        category=risk.category,
    )


def _should_drop_frame_risk(risk: VideoRiskAnnotation) -> bool:
    """Drop frame findings that are not actionable video risk objects."""
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    has_bbox = any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions)
    if risk.category == "visual.dangerous":
        return not _has_concrete_dangerous_object(risk)
    if risk.category == "privacy.face":
        if not has_bbox:
            return True
        decision = str(metadata.get("face_filter_decision") or "").strip().lower()
        if decision == "drop":
            return True
        identifiability = _safe_float(metadata.get("identifiability_score"), 0.0)
        if (decision == "keep" or bool(metadata.get("is_identifiable_face", False))) and identifiability >= 0.70:
            return False
        return True
    if risk.category in {"content.violence", "content.graphic_violence"}:
        return _is_duplicate_unlocalized_frame_content(risk)
    if str(metadata.get("video_role") or "") == "unlocalized_instance":
        return True
    return False


def _has_concrete_dangerous_object(risk: VideoRiskAnnotation) -> bool:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("instance_label_zh", "instance_label_en", "object_name_zh", "risk_subtype_zh", "label")
    ).lower()
    if any(token in text for token in ("肢体冲突", "斗殴", "打架", "暴力行为", "多人冲突", "physical_conflict", "fight")):
        return False
    has_valid_region = False
    for region in risk.regions:
        if not isinstance(region, dict) or not isinstance(region.get("bbox"), dict):
            continue
        if _region_is_invalid_localization(region):
            return False
        has_valid_region = True
    return has_valid_region


def _is_duplicate_unlocalized_frame_content(risk: VideoRiskAnnotation) -> bool:
    metadata = risk.metadata if isinstance(risk.metadata, dict) else {}
    if any(isinstance(region, dict) and isinstance(region.get("bbox"), dict) for region in risk.regions):
        return False
    if str(metadata.get("video_role") or "") == "event":
        return True
    if str(metadata.get("localization_status") or "").strip().lower() in {"", "unlocalized"}:
        return bool(metadata.get("moderation_result", False))
    return False


def _region_is_invalid_localization(region: dict[str, Any]) -> bool:
    source = str(region.get("source") or "").strip().lower()
    status = str(region.get("localization_status") or "").strip().lower()
    if source in {"qwen_rough_bbox_fallback", "rough_bbox_fallback"}:
        return True
    if "coarse_localization" in status or "sam3_rejections" in status:
        return True
    quality = region.get("mask_quality_score")
    if quality is not None:
        try:
            return float(quality) < 0.35
        except (TypeError, ValueError):
            return False
    return False


def _trackable_target_for_finding(
    category: str,
    finding: PictureFinding,
    metadata: dict[str, Any],
    regions: list[dict[str, Any]],
) -> str:
    if not regions:
        return ""
    text = " ".join(
        str(value or "")
        for value in (
            category,
            finding.category,
            finding.label,
            finding.explanation,
            metadata.get("entity_label_en"),
            metadata.get("entity_label_zh"),
            metadata.get("object_name_zh"),
            metadata.get("risk_subtype_zh"),
            metadata.get("risk_reason_zh"),
        )
    ).lower()
    if category in {"content.violence", "content.graphic_violence"} and any(
        token in text
        for token in ("fight", "fighting", "physical_conflict", "conflict", "打斗", "斗殴", "肢体冲突", "推搡", "拉扯")
    ):
        return "conflict_region"
    if any(token in text for token in ("weapon", "gun", "knife", "pistol", "rifle", "枪", "刀", "武器")):
        return "weapon"
    if any(token in text for token in ("blood", "wound", "injury", "血迹", "伤口", "受伤")):
        return "blood_or_wound"
    return ""


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_operator_for_finding(category: str, finding: PictureFinding) -> str:
    raw_category = str(finding.category or "").strip().lower()
    vpi_by_category = {
        "face": "VPI_001",
        "id_card": "VPI_002",
        "badge": "VPI_003",
        "signature": "VPI_004",
        "stamp": "VPI_005",
        "qr_code": "VPI_006",
        "barcode": "VPI_007",
        "license_plate": "VPI_008",
        "avatar": "VPI_009",
        "account_region": "VPI_010",
        "school_class_identifier": "VPI_011",
    }
    if raw_category in vpi_by_category:
        return vpi_by_category[raw_category]
    return _source_operator_for_category(category)


def _source_operator_for_category(category: str) -> str:
    normalized = str(category or "").strip().lower()
    if normalized.startswith("content."):
        for operator_id, label in CSA_LABELS.items():
            if normalized == label or normalized == label.replace("content.pornographic", "content.sexual"):
                return operator_id
        aliases = {
            "content.sexual": "CSA_002",
            "content.violence": "CSA_003",
            "content.graphic_violence": "CSA_003",
            "content.hate": "CSA_004",
            "content.hate_speech": "CSA_004",
            "content.harassment": "CSA_005",
            "content.self_harm": "CSA_006",
            "content.illegal_instruction": "CSA_007",
            "content.illegal_activity": "CSA_007",
            "content.minor_harmful": "CSA_008",
            "content.misleading": "CSA_009",
            "content.scam": "CSA_009",
            "content.fraud": "CSA_009",
        }
        return aliases.get(normalized, "")
    privacy = normalized.replace("privacy.", "")
    for operator_id, targets in PII_TARGETS.items():
        if privacy in {str(item).lower() for item in targets}:
            return operator_id
    aliases = {
        "phone": "PII_002",
        "audio_pii": "PII_002",
        "id_card": "PII_003",
        "address": "PII_004",
        "screen_sensitive": "PII_009",
    }
    return aliases.get(privacy, "")


def _video_operator_id(source_modality: str, source_operator_id: str) -> str:
    if not source_operator_id:
        return ""
    if source_operator_id.startswith("VPI_"):
        return f"VVIS_{source_operator_id}"
    if source_operator_id.startswith("PII_"):
        return f"VAUD_{source_operator_id}" if source_modality == "audio" else f"VTXT_{source_operator_id}"
    if source_operator_id.startswith("CSA_"):
        if source_modality == "audio":
            return f"VAUD_{source_operator_id}"
        if source_modality == "safety":
            return f"VVIS_{source_operator_id}"
        return f"VTXT_{source_operator_id}"
    return source_operator_id


def _target_type_for_category(category: str) -> str:
    category = str(category or "").strip().lower()
    if category.startswith("privacy."):
        return category.split(".", 1)[1]
    return category


def _frame_span(frame: FrameReference) -> TimeSpan:
    duration_ms = int(frame.metadata.get("duration_ms", 0) or 0)
    return TimeSpan(start_ms=frame.pts_ms, end_ms=frame.pts_ms + max(1, duration_ms))


def _regions_from_finding(frame: FrameReference, finding: PictureFinding) -> list[dict[str, Any]]:
    if finding.region is None:
        return []
    region = finding.region
    metadata = dict(finding.metadata or {})
    evidence = _first_evidence_region(metadata)
    payload: dict[str, Any] = {
        "frame_id": frame.frame_id,
        "bbox": region.bbox.model_dump(mode="json"),
        "confidence": region.confidence,
        "source": str(evidence.get("source") or metadata.get("source") or finding.provider or "picture_frame_detection"),
        "localization_status": str(evidence.get("localization_status") or metadata.get("localization_status") or ""),
        "mask_quality_score": evidence.get("mask_quality_score", metadata.get("mask_quality_score")),
        "instance_label_zh": str(evidence.get("entity_label_zh") or metadata.get("entity_label_zh") or metadata.get("object_name_zh") or finding.label or ""),
        "instance_label_en": str(evidence.get("entity_label_en") or metadata.get("entity_label_en") or finding.category or ""),
        "violation_id": str(evidence.get("violation_id") or metadata.get("violation_id") or ""),
    }
    if region.polygon is not None:
        payload["polygon"] = region.polygon.model_dump(mode="json")
    if region.mask_path:
        payload["mask_path"] = region.mask_path
    for key in ("boundary_status", "decision_hint", "risk_subtype", "redaction_target"):
        value = evidence.get(key, metadata.get(key))
        if value not in (None, ""):
            payload[key] = value
    return [payload]


def _video_role_for_finding(
    category: str,
    source_modality: str,
    regions: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    if not regions:
        return "event" if category.startswith("content.") else "unlocalized_instance"
    status = str(metadata.get("localization_status") or "").lower()
    if "coarse_localization" in status or bool(metadata.get("localization_required", False)):
        return "localized_candidate"
    if category.startswith("content.") or category.startswith("privacy.") or source_modality in {"visual", "ocr_text"}:
        return "object_instance"
    return "localized_candidate"


def _first_evidence_region(metadata: dict[str, Any]) -> dict[str, Any]:
    evidence_regions = metadata.get("evidence_regions")
    if isinstance(evidence_regions, list):
        for item in evidence_regions:
            if isinstance(item, dict):
                return item
    return {}


def _instance_label_zh(finding: PictureFinding, metadata: dict[str, Any], target_type: str) -> str:
    return str(
        metadata.get("entity_label_zh")
        or metadata.get("object_name_zh")
        or metadata.get("risk_subtype_zh")
        or finding.label
        or target_type
        or finding.category
        or ""
    ).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _audio_category(record: dict[str, Any], source_name: str) -> str:
    raw = " ".join(
        str(record.get(key) or "")
        for key in ("category", "label", "entity_type", "reason_code", "decision", "risk_type")
    ).lower()
    if "privacy" in source_name or any(token in raw for token in ("phone", "id", "address", "pii", "person")):
        if "phone" in raw:
            return "privacy.phone"
        if "address" in raw:
            return "privacy.address"
        return "privacy.audio_pii"
    if "sexual" in raw:
        return "content.sexual"
    if "violence" in raw:
        return "content.violence"
    if "hate" in raw:
        return "content.hate"
    if "self" in raw or "suicide" in raw:
        return "content.self_harm"
    return "content.audio_safety"


def _record_confidence(record: dict[str, Any]) -> float:
    for key in ("score", "confidence", "risk_score"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.7


def _record_span(record: dict[str, Any]) -> tuple[int, int]:
    start = _first_number(record, ("start_ms", "start_time_ms", "start"))
    end = _first_number(record, ("end_ms", "end_time_ms", "end"))
    if end <= start:
        end = start + 1
    return int(start), int(end)


def _first_number(record: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            if match:
                return float(match.group(0))
    return 0.0


def _record_reason_codes(record: dict[str, Any]) -> list[str]:
    value = record.get("reason_codes") or record.get("reasons") or record.get("reason_code")
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []

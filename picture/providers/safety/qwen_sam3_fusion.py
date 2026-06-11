from __future__ import annotations

from pathlib import Path
from typing import Any

from picture.domain.enums import SafetyCategory
from picture.domain.models import BBox, PictureModerationResult, RegionMask
from picture.providers.base import SafetyModerator
from picture.providers.safety.qwen35_vl import Qwen35VLSafetyModerator
from picture.providers.vision.sam3_api import SAM3APIVisionDetector


SAFETY_SAM3_PROMPTS: dict[str, tuple[str, ...]] = {
    SafetyCategory.EXPLICIT.value: (
        "nudity",
        "naked body",
        "explicit sexual content",
        "sexual body part",
    ),
    SafetyCategory.GRAPHIC_VIOLENCE.value: (
        "blood",
        "wound",
        "injury",
        "violent weapon",
        "graphic violence",
    ),
    SafetyCategory.HATE_SYMBOL.value: (
        "hate symbol",
        "extremist symbol",
        "nazi symbol",
        "racist symbol",
    ),
    SafetyCategory.SELF_HARM.value: (
        "self harm object",
        "suicide scene",
        "wrist wound",
        "dangerous self injury",
    ),
    SafetyCategory.DANGEROUS.value: (
        "gun",
        "pistol",
        "handgun",
        "firearm",
        "rifle",
        "knife",
        "drug package",
    ),
    SafetyCategory.OTHER_NSFW.value: (
        "bare torso",
        "nude torso",
        "chest and abdomen",
        "torso without head",
        "skin area",
        "human body",
        "upper body",
        "unsafe visual content",
        "inappropriate object",
        "nsfw object",
    ),
}

class QwenSAM3SafetyFusionModerator(SafetyModerator):
    """Qwen3.5 semantic safety judgment plus SAM3 API localization."""

    def __init__(
        self,
        sam3_api_url: str = "http://127.0.0.1:8218",
        sam3_timeout_seconds: float = 180.0,
        sam3_confidence: float = 0.35,
        qwen_timeout_seconds: float = 120.0,
        qwen_max_tokens: int = 384,
        image_max_side: int = 1280,
        image_jpeg_quality: int = 85,
    ) -> None:
        self._qwen = Qwen35VLSafetyModerator(
            timeout_seconds=qwen_timeout_seconds,
            max_tokens=qwen_max_tokens,
            image_max_side=image_max_side,
            image_jpeg_quality=image_jpeg_quality,
        )
        self._sam3 = SAM3APIVisionDetector(
            base_url=sam3_api_url,
            confidence_threshold=sam3_confidence,
            timeout_seconds=sam3_timeout_seconds,
            prompts=SAFETY_SAM3_PROMPTS,
        )

    @property
    def name(self) -> str:
        return "Qwen3.5+SAM3SafetyFusionModerator"

    def warmup(self) -> dict[str, Any]:
        qwen_provider = self._qwen._get_provider()
        sam3_status = self._sam3.warmup()
        return {
            "qwen_model": qwen_provider.model,
            "qwen_base_url": qwen_provider.base_url,
            "sam3": sam3_status,
        }

    def moderate(self, image_path: str) -> PictureModerationResult:
        moderation = self._qwen.moderate(image_path)
        if moderation.is_safe:
            return moderation.model_copy(
                update={
                    "provider": self.name,
                    "metadata": {
                        **dict(moderation.metadata or {}),
                        "fusion_provider": self.name,
                        "sam3_localized": False,
                    },
                }
            )

        categories = [
            category.value
            for category in moderation.categories
            if category != SafetyCategory.SAFE
        ] or [SafetyCategory.OTHER_NSFW.value]
        details = dict((moderation.metadata or {}).get("category_details") or {})
        qwen_hints = _qwen_hint_evidence(categories, moderation.metadata or {})
        violations = _safety_violations(categories, moderation.metadata or {})
        if violations:
            prompt_rounds: list[dict[str, list[str]]] = []
            sam3_evidence, rejected_sam3_evidence, reviewed_violations = _run_violation_localization(
                self._sam3,
                self._qwen,
                image_path,
                violations,
                self._sam3._confidence_threshold,
            )
        else:
            reviewed_violations = []
            prompt_rounds = _safety_prompt_rounds(categories, details, moderation.metadata or {})
            thresholds = _safety_thresholds(categories, self._sam3._confidence_threshold)
            sam3_findings = _run_prompt_rounds(
                self._sam3,
                image_path,
                categories,
                prompt_rounds,
                thresholds,
            )
            sam3_evidence, rejected_sam3_evidence = _select_reliable_sam3_evidence(
                sam3_findings,
                qwen_hints,
                image_path,
            )
        localized_categories = {str(item.get("category") or "").lower() for item in sam3_evidence}
        unlocalized_categories = [
            category for category in categories if category.lower() not in localized_categories
        ]
        evidence = sam3_evidence

        metadata = {
            **dict(moderation.metadata or {}),
            "fusion_provider": self.name,
            "qwen_provider": moderation.provider,
            "sam3_provider": self._sam3.name,
            "sam3_localized": bool(sam3_evidence),
            "sam3_localized_count": len(sam3_evidence),
            "sam3_prompt_rounds": prompt_rounds,
            "evidence_regions": evidence,
            "localized_violations": reviewed_violations,
            "violations": violations,
            "qwen_evidence_hints": qwen_hints,
            "rejected_sam3_evidence": rejected_sam3_evidence,
            "unlocalized_safety_categories": unlocalized_categories,
            "localization_status": "localized_by_sam3" if not unlocalized_categories else "partially_localized" if sam3_evidence else "unlocalized",
            "review_required": bool((moderation.metadata or {}).get("review_required", False)) or bool(unlocalized_categories),
        }
        return moderation.model_copy(update={"provider": self.name, "metadata": metadata})


def _safety_prompt_rounds(
    categories: list[str],
    details: dict[str, Any],
    metadata: dict[str, Any],
) -> list[dict[str, list[str]]]:
    qwen_specific: dict[str, list[str]] = {category: [] for category in categories}
    generalized: dict[str, list[str]] = {
        category: list(SAFETY_SAM3_PROMPTS.get(category, ())) for category in categories
    }
    evidence_regions = metadata.get("evidence_regions") if isinstance(metadata.get("evidence_regions"), list) else []
    for category in categories:
        detail = details.get(category) if isinstance(details.get(category), dict) else {}
        for value in (
            detail.get("object_name_zh"),
            detail.get("risk_subtype_zh"),
        ):
            _append_prompt_fragments(qwen_specific[category], str(value or ""), category)
        for item in evidence_regions:
            if not isinstance(item, dict):
                continue
            if str(item.get("category") or "").lower() not in {"", category.lower()}:
                continue
            _append_prompt_fragments(qwen_specific[category], str(item.get("label") or ""), category)
    return [
        {category: values for category, values in qwen_specific.items() if values},
        {category: values for category, values in generalized.items() if values},
    ]


def _run_prompt_rounds(
    sam3: SAM3APIVisionDetector,
    image_path: str,
    categories: list[str],
    prompt_rounds: list[dict[str, list[str]]],
    thresholds: dict[str, float],
) -> list[Any]:
    findings: list[Any] = []
    for round_index, prompts in enumerate(prompt_rounds, start=1):
        active_categories = [
            category for category in categories
            if round_index == 2 or prompts.get(category)
        ]
        if not active_categories:
            continue
        round_findings = sam3.detect_with_prompts(
            image_path,
            target_types=active_categories,
            extra_prompts=prompts,
            confidence_thresholds=thresholds,
        )
        for finding in round_findings:
            metadata = dict(finding.metadata or {})
            metadata["safety_localization_round"] = round_index
            finding = finding.model_copy(update={"metadata": metadata})
            findings.append(finding)
    return findings


def _safety_violations(categories: list[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw = metadata.get("violations")
    if not isinstance(raw, list):
        return []
    allowed_categories = {category.lower() for category in categories}
    violations: list[dict[str, Any]] = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower()
        if category not in allowed_categories:
            continue
        sam_prompt = _canonical_safety_prompt(
            category,
            str(item.get("sam_prompt_text") or item.get("entity_label_en") or item.get("entity_label") or ""),
        )
        sam_prompts = _canonical_prompt_list(category, item.get("sam_prompt_texts") or [])
        if sam_prompt and sam_prompt not in sam_prompts:
            sam_prompts.insert(0, sam_prompt)
        label_en = _canonical_safety_prompt(category, str(item.get("entity_label_en") or item.get("entity_label") or "")) or sam_prompt
        label_zh = str(item.get("entity_label_zh") or item.get("label") or label_en).strip()
        center = item.get("center_point")
        centers = _center_points(item.get("center_points") or [])
        if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
            canonical_center = [float(center[0]), float(center[1])]
            if canonical_center not in centers:
                centers.insert(0, canonical_center)
        if not label_en and not label_zh:
            continue
        violation = {
            **item,
            "violation_id": str(item.get("violation_id") or f"{category}_{index}"),
            "category": category,
            "entity_label_en": label_en,
            "entity_label_zh": label_zh,
            "sam_prompt_text": sam_prompt or label_en,
            "sam_prompt_texts": _expand_category_prompts(category, sam_prompts or [sam_prompt or label_en]),
            "redaction_target": str(item.get("redaction_target") or item.get("target_region") or ""),
            "center_point": centers[0] if centers else None,
            "center_points": centers,
            "rough_bbox": _bbox(item.get("rough_bbox") or item.get("bbox")),
            "confidence": float(item.get("confidence") or 0.0) if _is_number(item.get("confidence")) else 0.0,
        }
        _backfill_exposed_upper_body_policy(violation)
        violations.append(violation)
    return violations


def _run_violation_localization(
    sam3: SAM3APIVisionDetector,
    qwen: Qwen35VLSafetyModerator,
    image_path: str,
    violations: list[dict[str, Any]],
    default_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []
    width, height = _image_size(image_path)
    for raw_violation in violations:
        violation = _normalize_violation_geometry(raw_violation, width, height)
        category = str(violation.get("category") or "").lower()
        prompt = _canonical_safety_prompt(
            category,
            str(violation.get("sam_prompt_text") or violation.get("entity_label_en") or violation.get("entity_label_zh") or ""),
        )
        if not prompt:
            rejected.append({**violation, "rejection_reason": "missing_specific_prompt"})
            reviewed.append({**violation, "localization_status": "unlocalized", "review_required": True})
            continue
        threshold = _safety_thresholds([category], default_threshold).get(category, default_threshold)
        candidates: list[dict[str, Any]] = []
        for attempt in _sam3_localization_attempts(sam3, image_path, violation, prompt, threshold, category):
            attempt_candidates, candidate_rejected = _select_violation_candidates(
                attempt["findings"],
                violation,
                image_path,
                require_center_proximity=attempt["require_center_proximity"],
            )
            for item in attempt_candidates:
                item["localization_attempt"] = attempt["name"]
            rejected.extend({**item, "localization_attempt": attempt["name"]} for item in candidate_rejected)
            candidates.extend(attempt_candidates)
            if candidates:
                break
        if not candidates:
            coarse = _coarse_evidence_from_violation(violation, width, height)
            if coarse is None:
                reviewed.append({**violation, "localization_status": "unlocalized", "review_required": True})
                continue
            coarse["localization_status"] = "coarse_localization_from_qwen_rough_bbox"
            coarse["review_required"] = True
            selected.append(coarse)
            reviewed.append({**violation, "localization_status": coarse["localization_status"], "review_required": True, "bbox": coarse.get("bbox")})
            continue
        accepted = None
        for evidence in candidates[:3]:
            verified = _review_and_refine_candidate(sam3, qwen, image_path, evidence, violation, width, height)
            if verified is None:
                rejected.append({**evidence, "rejection_reason": "qwen_local_review_rejected", "localization_status": "rejected_unreliable"})
                continue
            evidence = verified
            review = dict(evidence.get("local_review") or {})
            evidence.update(
                {
                    "entity_label_en": str(review.get("entity_label_en") or violation.get("entity_label_en") or prompt),
                    "entity_label_zh": str(review.get("entity_label_zh") or violation.get("entity_label_zh") or ""),
                    "boundary_status": str(review.get("boundary_status") or "uncertain"),
                    "localization_status": "localized_by_qwen_point_sam3_refined_mask_verified" if evidence.get("mask_path") else "localized_by_qwen_point_sam3_verified",
                    "review_required": str(review.get("boundary_status") or "") != "complete",
                    "source": "qwen_point_sam3_local_review",
                }
            )
            if evidence.get("global_target_coverage") == "partial":
                evidence["boundary_status"] = "truncated"
                evidence["review_required"] = True
            accepted = evidence
            break
        if accepted is None:
            coarse = _coarse_evidence_from_violation(violation, width, height)
            if coarse is not None:
                coarse["localization_status"] = "coarse_localization_after_sam3_rejections"
                coarse["review_required"] = True
                selected.append(coarse)
                reviewed.append({**violation, "localization_status": coarse["localization_status"], "review_required": True, "bbox": coarse.get("bbox")})
                continue
            reviewed.append({**violation, "localization_status": "unlocalized", "review_required": True})
            continue
        selected.append(accepted)
        reviewed.append({**violation, "localization_status": accepted["localization_status"], "review_required": bool(accepted.get("review_required", False)), "bbox": accepted.get("bbox")})
    return selected, rejected, reviewed


def _sam3_localization_attempts(
    sam3: SAM3APIVisionDetector,
    image_path: str,
    violation: dict[str, Any],
    prompt: str,
    threshold: float,
    category: str,
) -> Any:
    prompts = _expand_category_prompts(category, violation.get("sam_prompt_texts") or [prompt])
    centers = _center_points(violation.get("center_points") or [])
    center = violation.get("center_point")
    if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
        first = [float(center[0]), float(center[1])]
        if first not in centers:
            centers.insert(0, first)
    if centers:
        point_payload = [
            {
                "category": category,
                "text": text,
                "point": point,
                "threshold": threshold,
                "box_size_ratio": _point_box_size_ratio(category),
            }
            for point in centers[:4]
            for text in prompts[:3]
        ]
        if point_payload:
            yield {
                "name": "point_text_prompt",
                "require_center_proximity": False if category == SafetyCategory.EXPLICIT.value else True,
                "findings": sam3.detect_with_points(image_path, point_payload),
            }
    yield {
        "name": "exact_text_prompt",
        "require_center_proximity": False,
        "findings": sam3.detect_exact_prompts(
            image_path,
            [{"category": category, "text": text, "threshold": threshold} for text in prompts[:6]],
        ),
    }
    fallback_prompts = _fallback_prompts_for_violation(category, prompt, violation)
    if fallback_prompts:
        yield {
            "name": "category_prompt_round",
            "require_center_proximity": False,
            "findings": sam3.detect_with_prompts(
                image_path,
                target_types=[category],
                extra_prompts={category: fallback_prompts},
                confidence_thresholds={category: threshold},
            ),
        }


def _review_and_refine_candidate(
    sam3: SAM3APIVisionDetector,
    qwen: Qwen35VLSafetyModerator,
    image_path: str,
    evidence: dict[str, Any],
    violation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    current = dict(evidence)
    original_bbox = list(current.get("bbox") or [])
    original_polygon = current.get("polygon")
    original_mask_path = current.get("mask_path")
    last_review: dict[str, Any] = {}
    if _explicit_body_violation_has_tiny_local_candidate(current, violation, image_width, image_height):
        current["review_failure_reason"] = "explicit_body_candidate_too_local"
        return None
    for round_index in range(2):
        crop_info = _crop_for_local_review(image_path, current["bbox"], image_width, image_height)
        if crop_info is None:
            current["review_failure_reason"] = "crop_failed"
            return None
        review = qwen.verify_local_violation(crop_info["crop_path"], violation)
        current["local_review"] = review
        current["crop_bbox"] = crop_info["crop_bbox"]
        current["crop_path"] = crop_info["crop_path"]
        current["local_review_round"] = round_index + 1
        last_review = review
        if _upper_body_review_rejects_candidate(current, violation) or not bool(review.get("is_authentic_violation")) or float(review.get("confidence") or 0.0) < 0.35:
            current["review_failure_reason"] = "not_same_violation"
            if _upper_body_candidate_can_bypass_local_rejection(current, violation, image_width, image_height):
                current["local_review_bypassed"] = True
                current["local_review_bypass_reason"] = "sam3_upper_body_geometry_preserved"
                break
            return None
        local_bbox = review.get("local_bbox")
        if _local_bbox_is_usable(local_bbox, crop_info, original_bbox, violation):
            suggested_bbox = [
                float(crop_info["crop_bbox"][0]) + float(local_bbox[0]),
                float(crop_info["crop_bbox"][1]) + float(local_bbox[1]),
                float(local_bbox[2]),
                float(local_bbox[3]),
            ]
            current["qwen_local_bbox_applied"] = True
            current["bbox"] = suggested_bbox
        else:
            current["qwen_local_bbox_rejected"] = local_bbox
        if str(review.get("boundary_status") or "").lower() in {"truncated", "uncertain"} and round_index == 0:
            current["bbox"] = _expand_bbox(current["bbox"], image_width, image_height, ratio=0.75)
            current["boundary_expanded_for_review"] = True
            _refine_verified_evidence_with_sam3(sam3, image_path, current, violation, image_width, image_height)
            continue
        break
    current["local_review"] = last_review
    _refine_verified_evidence_with_sam3(sam3, image_path, current, violation, image_width, image_height)
    if current.get("sam3_refine_rejected"):
        current["bbox"] = original_bbox
        if original_polygon is not None:
            current["polygon"] = original_polygon
        if original_mask_path:
            current["mask_path"] = original_mask_path
        current["fallback_to_original_sam3_mask"] = True
    _apply_upper_body_global_coverage(current, violation)
    _annotate_evidence_region_quality(current, image_width, image_height)
    return current


def _refine_verified_evidence_with_sam3(
    sam3: SAM3APIVisionDetector,
    image_path: str,
    evidence: dict[str, Any],
    violation: dict[str, Any] | None = None,
    image_width: int = 0,
    image_height: int = 0,
) -> None:
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    original_bbox = [float(item) for item in bbox]
    try:
        region = RegionMask(
            bbox=BBox(x=float(bbox[0]), y=float(bbox[1]), w=float(bbox[2]), h=float(bbox[3])),
            confidence=float(evidence.get("confidence") or 0.0),
        )
        refined = sam3.refine_regions(image_path, [region])
    except Exception as exc:
        evidence["sam3_refine_error"] = f"{type(exc).__name__}: {exc}"
        return
    if not refined:
        return
    item = refined[0]
    refined_bbox = [float(item.bbox.x), float(item.bbox.y), float(item.bbox.w), float(item.bbox.h)]
    if not _refined_bbox_is_reliable(original_bbox, refined_bbox, violation or {}, image_width, image_height):
        evidence["sam3_refine_rejected"] = True
        evidence["sam3_refine_rejected_bbox"] = refined_bbox
        return
    evidence["bbox"] = refined_bbox
    evidence["sam3_refined"] = True
    if item.mask_path:
        evidence["mask_path"] = item.mask_path
    if item.polygon is not None and item.polygon.points:
        evidence["polygon"] = [[float(x), float(y)] for x, y in item.polygon.points]
        evidence.setdefault("polygons", [evidence["polygon"]])
    evidence["confidence"] = max(float(evidence.get("confidence") or 0.0), float(item.confidence or 0.0))


def _select_violation_candidates(
    findings: list[Any],
    violation: dict[str, Any],
    image_path: str,
    *,
    require_center_proximity: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = _image_size(image_path)
    accepted: list[tuple[float, dict[str, Any]]] = []
    rejected: list[dict[str, Any]] = []
    center = violation.get("center_point")
    for finding in findings:
        if finding.region is None:
            continue
        evidence = _finding_to_evidence(finding, [])
        evidence.update(
            {
                "violation_id": violation.get("violation_id"),
                "entity_label_en": violation.get("entity_label_en"),
                "entity_label_zh": violation.get("entity_label_zh"),
                "risk_subtype": violation.get("risk_subtype"),
                "decision_hint": violation.get("decision_hint"),
                "redaction_target": violation.get("redaction_target"),
                "center_point": center,
            }
        )
        ok, reason, score = _violation_candidate_quality(
            evidence,
            violation,
            width,
            height,
            require_center_proximity=require_center_proximity,
        )
        evidence["localization_quality_score"] = round(score, 4)
        if not ok:
            rejected.append({**evidence, "rejection_reason": reason, "localization_status": "rejected_unreliable"})
            continue
        accepted.append((score, evidence))
    accepted.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in accepted], rejected


def _append_prompt_fragments(items: list[str], value: str, category: str) -> None:
    text = value.strip()
    if not text:
        return
    candidates = [text]
    for sep in ("、", "，", ",", "/", "；", ";", " "):
        if sep in text:
            candidates.extend(part.strip() for part in text.split(sep) if part.strip())
    for candidate in candidates:
        if 1 < len(candidate) <= 40 and candidate not in items and _safety_prompt_allowed(category, candidate):
            items.append(candidate)


def _safety_prompt_allowed(category: str, prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text or _looks_like_scene_sentence(text):
        return False
    if not _looks_like_sam3_short_noun_phrase(text):
        return False
    if category == SafetyCategory.EXPLICIT.value:
        blocked_context = (
            "fishnet",
            "stocking",
            "sock",
            "shoe",
            "boot",
            "heel",
            "hat",
            "cap",
            "uniform",
            "clothing",
            "dress",
            "skirt",
            "pants",
            "shirt",
        )
        if any(term in text for term in blocked_context):
            return False
    if category == SafetyCategory.DANGEROUS.value:
        blocked = (
            "weapon",
            "dangerous object",
            "unsafe object",
            "harmful object",
            "危险物",
            "危险对象",
            "危险行为",
            "人物",
            "人",
            "手持",
            "持有",
            "手臂",
            "身体",
        )
        return not (text in blocked or any(term in text for term in blocked))
    return True


def _canonical_safety_prompt(category: str, prompt: str) -> str:
    text = str(prompt or "").strip().lower()
    mapping = {
        "疑似手枪": "pistol",
        "手枪": "pistol",
        "枪械": "gun",
        "枪支": "gun",
        "疑似步枪": "rifle",
        "步枪": "rifle",
        "管制刀具": "knife",
        "刀具": "knife",
        "刀": "knife",
        "毒品": "drug package",
        "毒品包装": "drug package",
        "血液": "blood",
        "血迹": "blood",
        "伤口": "wound",
        "nude male": "naked body",
        "nude female": "naked body",
        "nude person": "naked body",
        "naked man": "naked body",
        "naked woman": "naked body",
        "naked buttocks": "exposed buttocks",
        "naked chest": "exposed chest",
        "exposed body": "naked body",
        "裸露上身": "bare torso",
        "裸露躯干": "bare torso",
        "裸露胸腹": "bare torso",
        "裸露背部": "bare torso",
        "上半身裸露": "bare torso",
        "bare upper body": "bare torso",
        "shirtless torso": "bare torso",
        "shirtless upper body": "bare torso",
        "body below head": "human body",
        "below head body": "human body",
        "headless body": "human body",
        "body except head": "human body",
        "nude male except head": "naked body",
        "nude body except head": "naked body",
        "色情裸露区域": "exposed body part",
        "裸露区域": "exposed body part",
        "裸露身体": "naked body",
        "裸露身体区域": "naked body",
        "裸露部位": "exposed body part",
        "头部以下身体": "human body",
        "除头以外身体": "human body",
        "生殖器": "genital area",
        "臀部": "buttocks",
        "裸露臀部": "exposed buttocks",
        "胸部": "chest",
        "裸露胸部": "exposed chest",
        "性玩具": "sex toy",
        "成人玩具": "sex toy",
        "震动棒": "vibrator",
    }
    text = mapping.get(text, text)
    if not _safety_prompt_allowed(category, text):
        return ""
    return text


def _fallback_prompts_for_violation(category: str, prompt: str, violation: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    for value in (
        prompt,
        violation.get("entity_label_en"),
        violation.get("entity_label_zh"),
        violation.get("sam_prompt_text"),
    ):
        canonical = _canonical_safety_prompt(category, str(value or ""))
        if canonical and canonical not in prompts:
            prompts.append(canonical)
    for value in violation.get("sam_prompt_texts") or []:
        canonical = _canonical_safety_prompt(category, str(value or ""))
        if canonical and canonical not in prompts:
            prompts.append(canonical)
    for value in SAFETY_SAM3_PROMPTS.get(category, ()):
        if value not in prompts:
            prompts.append(value)
    return prompts[:8]


def _canonical_prompt_list(category: str, prompts: Any) -> list[str]:
    raw_values = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
    values: list[str] = []
    for value in raw_values:
        canonical = _canonical_safety_prompt(category, str(value or ""))
        if canonical and canonical not in values:
            values.append(canonical)
    return values


def _expand_category_prompts(category: str, prompts: list[str] | tuple[str, ...] | Any) -> list[str]:
    values: list[str] = []
    raw_values = list(prompts) if isinstance(prompts, (list, tuple)) else [prompts]
    for value in raw_values:
        canonical = _canonical_safety_prompt(category, str(value or ""))
        if canonical and canonical not in values:
            values.append(canonical)
    if category == SafetyCategory.EXPLICIT.value:
        explicit_prompts = _explicit_prompt_defaults(values)
        for value in explicit_prompts:
            if value not in values:
                values.append(value)
    if category == SafetyCategory.OTHER_NSFW.value and any(_is_exposed_upper_body_prompt(value) for value in values):
        for value in ("bare torso", "nude torso", "chest and abdomen", "torso without head", "skin area", "human body", "upper body"):
            if value not in values:
                values.append(value)
    for value in SAFETY_SAM3_PROMPTS.get(category, ()):
        if value not in values:
            values.append(value)
    return values[:10]


def _explicit_prompt_defaults(existing: list[str]) -> tuple[str, ...]:
    if any(_is_explicit_body_prompt(value) for value in existing):
        return (
            "human body",
            "naked body",
            "nude torso",
            "exposed body part",
            "genital area",
            "exposed buttocks",
            "exposed chest",
            "skin area",
        )
    if any(_is_explicit_local_prompt(value) for value in existing):
        return (
            "genital area",
            "exposed buttocks",
            "exposed chest",
            "exposed body part",
            "skin area",
            "naked body",
            "human body",
        )
    return (
        "exposed body part",
        "genital area",
        "exposed buttocks",
        "exposed chest",
        "naked body",
        "human body",
        "nude torso",
        "skin area",
    )


def _point_box_size_ratio(category: str) -> float:
    if category == SafetyCategory.DANGEROUS.value:
        return 0.05
    if category in {SafetyCategory.EXPLICIT.value, SafetyCategory.OTHER_NSFW.value}:
        return 0.16
    if category in {SafetyCategory.HATE_SYMBOL.value, SafetyCategory.SELF_HARM.value}:
        return 0.06
    return 0.10


def _looks_like_scene_sentence(value: str) -> bool:
    text = str(value or "").strip()
    markers = (
        "存在",
        "画面",
        "场景",
        "行为",
        "不适合",
        "风险",
        "相关",
        "可见",
        "左侧",
        "右侧",
        "上方",
        "下方",
        "区域",
        "附近",
        "手持",
        "持有",
        "person",
        "holding",
    )
    return len(text) > 8 and any(marker in text for marker in markers)


def _looks_like_sam3_short_noun_phrase(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"torso without head", "chest and abdomen"}:
        return True
    blocked_fragments = (
        " except ",
        " without ",
        " below ",
        " above ",
        " excluding ",
        " include ",
        " cover ",
        "redact",
        "unsafe",
        "compliance",
        "education",
        "holding",
        "person holding",
        "区域",
        "以下",
        "以外",
        "除",
        "不要",
        "保留",
        "脱敏",
        "遮盖",
        "违规",
        "色情",
    )
    if any(fragment in text for fragment in blocked_fragments):
        return False
    if any(ord(ch) > 127 for ch in text):
        return False
    words = [part for part in text.replace("-", " ").split() if part]
    if not 1 <= len(words) <= 4:
        return False
    return all(part.replace("'", "").isalnum() for part in words)


def _safety_thresholds(categories: list[str], default_threshold: float) -> dict[str, float]:
    thresholds = {}
    for category in categories:
        if category in {SafetyCategory.DANGEROUS.value, SafetyCategory.EXPLICIT.value}:
            thresholds[category] = min(0.25, default_threshold)
        else:
            thresholds[category] = min(0.30, default_threshold)
    return thresholds


def _finding_to_evidence(finding: Any, qwen_hints: list[dict[str, Any]]) -> dict[str, Any]:
    bbox = finding.region.bbox
    metadata = dict(finding.metadata or {})
    category = str(finding.category or "")
    bbox_list = [float(bbox.x), float(bbox.y), float(bbox.w), float(bbox.h)]
    overlap = _best_hint_iou(bbox_list, category, qwen_hints)
    evidence = {
        "category": category,
        "label": str(metadata.get("prompt") or finding.label or finding.category),
        "bbox": bbox_list,
        "description": finding.explanation or "SAM3 localized visual safety risk.",
        "confidence": float(finding.score or finding.region.confidence or 0.0),
        "source": "sam3_safety_localization",
        "localization_status": "localized_by_sam3",
        "safety_localization_round": metadata.get("safety_localization_round"),
        "qwen_hint_iou": overlap,
    }
    if finding.region.mask_path:
        evidence["mask_path"] = finding.region.mask_path
    if finding.region.polygon is not None:
        evidence["polygon"] = [[float(x), float(y)] for x, y in finding.region.polygon.points]
        evidence.setdefault("polygons", [evidence["polygon"]])
    for key in (
        "polygons",
        "mask_area",
        "mask_area_ratio",
        "mask_bbox_fill_ratio",
        "point_prompts",
        "point_anchor_bboxes",
    ):
        if metadata.get(key) is not None:
            evidence[key] = metadata.get(key)
    if metadata.get("point") is not None:
        evidence["point"] = metadata.get("point")
    if metadata.get("point_anchor_bbox") is not None:
        evidence["point_anchor_bbox"] = metadata.get("point_anchor_bbox")
    _annotate_evidence_region_quality(evidence)
    return evidence


def _violation_candidate_quality(
    evidence: dict[str, Any],
    violation: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    require_center_proximity: bool = True,
) -> tuple[bool, str, float]:
    category = str(evidence.get("category") or violation.get("category") or "").lower()
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "missing_bbox", 0.0
    prompt = str(evidence.get("label") or violation.get("entity_label_en") or "").lower()
    confidence = float(evidence.get("confidence") or 0.0)
    area_ratio = _bbox_image_area_ratio(bbox, image_width, image_height)
    center = violation.get("center_point")
    distance_score = 0.0
    if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
        cx = float(center[0])
        cy = float(center[1])
        if not _point_near_bbox(cx, cy, bbox, expand=0.75):
            if require_center_proximity:
                return False, "far_from_qwen_center_point", confidence
            distance_score = -0.15
        else:
            distance_score = max(0.0, 0.35 - _normalized_point_bbox_distance(cx, cy, bbox, image_width, image_height))
    if category == SafetyCategory.DANGEROUS.value:
        if not _safety_prompt_allowed(category, prompt):
            return False, "generic_dangerous_prompt", confidence
    elif category == SafetyCategory.EXPLICIT.value:
        if _is_explicit_context_prompt(prompt):
            return False, "explicit_context_prompt", confidence
        if area_ratio > 0.60 and not _explicit_allows_large_bbox(evidence, violation):
            return False, "safety_bbox_too_large", confidence
    elif category == SafetyCategory.OTHER_NSFW.value and _is_exposed_upper_body_violation(violation):
        if _is_explicit_context_prompt(prompt):
            return False, "explicit_context_prompt", confidence
        if _upper_body_candidate_is_limb_like(evidence, image_width, image_height):
            return False, "upper_body_candidate_limb_like", confidence
        if area_ratio > 0.75 and not _is_exposed_upper_body_prompt(prompt):
            return False, "safety_bbox_too_large", confidence
    elif area_ratio > 0.60:
        return False, "safety_bbox_too_large", confidence
    quality = confidence + _prompt_specificity_bonus(category, prompt) + distance_score - max(0.0, area_ratio - 0.20)
    if quality < 0.45:
        return False, "low_localization_quality", quality
    return True, "accepted", quality


def _select_reliable_sam3_evidence(
    findings: list[Any],
    qwen_hints: list[dict[str, Any]],
    image_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = _image_size(image_path)
    by_category: dict[str, list[tuple[float, dict[str, Any], str]]] = {}
    rejected: list[dict[str, Any]] = []
    for finding in findings:
        if finding.region is None:
            continue
        evidence = _finding_to_evidence(finding, qwen_hints)
        category = str(evidence.get("category") or "").lower()
        accepted, reason, score = _localization_quality(evidence, qwen_hints, width, height)
        evidence["localization_quality_score"] = round(score, 4)
        if not accepted:
            rejected.append({**evidence, "rejection_reason": reason, "localization_status": "rejected_unreliable"})
            continue
        by_category.setdefault(category, []).append((score, evidence, reason))

    selected: list[dict[str, Any]] = []
    for category, items in by_category.items():
        items.sort(key=lambda item: item[0], reverse=True)
        for _, evidence, _ in items[:2 if category == SafetyCategory.DANGEROUS.value else 3]:
            selected.append(evidence)
    return selected, rejected


def _localization_quality(
    evidence: dict[str, Any],
    qwen_hints: list[dict[str, Any]],
    image_width: int,
    image_height: int,
) -> tuple[bool, str, float]:
    category = str(evidence.get("category") or "").lower()
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "missing_bbox", 0.0
    prompt = str(evidence.get("label") or "").lower()
    confidence = float(evidence.get("confidence") or 0.0)
    area_ratio = _bbox_image_area_ratio(bbox, image_width, image_height)
    hint_iou = _best_hint_iou(bbox, category, qwen_hints)
    center_in_hint = _center_in_expanded_hint(bbox, category, qwen_hints, expand=0.75)
    prompt_bonus = _prompt_specificity_bonus(category, prompt)
    size_penalty = max(0.0, area_ratio - (0.20 if category == SafetyCategory.DANGEROUS.value else 0.35)) * 2.0
    hint_bonus = max(hint_iou * 2.0, 0.35 if center_in_hint else 0.0)
    quality = confidence + prompt_bonus + hint_bonus - size_penalty

    if category == SafetyCategory.DANGEROUS.value:
        if not _safety_prompt_allowed(category, prompt):
            return False, "generic_dangerous_prompt", quality
        has_hint = any(str(item.get("category") or "").lower() in {"", category} and item.get("bbox") for item in qwen_hints)
        if has_hint and hint_iou < 0.03 and not center_in_hint:
            return False, "far_from_qwen_hint", quality
        if quality < 0.55:
            return False, "low_localization_quality", quality
    if category == SafetyCategory.EXPLICIT.value:
        if _is_explicit_context_prompt(prompt):
            return False, "explicit_context_prompt", quality
        if area_ratio > 0.85 and not _is_explicit_body_prompt(prompt):
            return False, "safety_bbox_too_large", quality
        if quality < 0.30:
            return False, "low_localization_quality", quality
    if category == SafetyCategory.OTHER_NSFW.value and _is_exposed_upper_body_evidence(evidence):
        if _is_explicit_context_prompt(prompt):
            return False, "explicit_context_prompt", quality
        if area_ratio > 0.85 and not _is_exposed_upper_body_prompt(prompt):
            return False, "safety_bbox_too_large", quality
        if quality < 0.25:
            return False, "low_localization_quality", quality
    return True, "accepted", quality


def _is_explicit_body_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    return text in {
        "naked body",
        "human body",
        "nude torso",
        "exposed body",
        "nude body",
        "nude person",
        "nude male",
        "nude female",
    }


def _is_explicit_local_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    return text in {
        "genital area",
        "exposed buttocks",
        "buttocks",
        "naked buttocks",
        "exposed chest",
        "chest",
        "breast",
        "breasts",
        "exposed body part",
        "skin area",
        "sex toy",
        "vibrator",
        "adult toy",
    }


def _is_explicit_context_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    context_terms = (
        "fishnet",
        "stocking",
        "sock",
        "shoe",
        "boot",
        "heel",
        "hat",
        "cap",
        "uniform",
        "clothing",
        "dress",
        "skirt",
        "pants",
        "shirt",
    )
    return any(term in text for term in context_terms)


def _is_exposed_upper_body_prompt(prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if text in {
        "bare torso",
        "nude torso",
        "chest and abdomen",
        "torso without head",
        "upper body",
        "shirtless torso",
        "shirtless upper body",
        "exposed upper body",
        "exposed torso",
        "bare upper body",
    }:
        return True
    return any(
        term in text
        for term in (
            "裸露上身",
            "裸露躯干",
            "裸露胸腹",
            "裸露背部",
            "上半身裸露",
            "裸露身体区域",
        )
    )


def _is_exposed_upper_body_violation(violation: dict[str, Any]) -> bool:
    values = [
        str(violation.get("risk_subtype") or ""),
        str(violation.get("entity_label_en") or ""),
        str(violation.get("entity_label_zh") or ""),
        str(violation.get("sam_prompt_text") or ""),
    ]
    values.extend(str(value or "") for value in violation.get("sam_prompt_texts") or [])
    return any(
        _is_exposed_upper_body_prompt(value)
        or str(value).strip().lower() in {"exposed_upper_body", "裸露上身", "裸露躯干", "裸露胸腹", "裸露背部", "上半身裸露", "裸露身体区域"}
        for value in values
    )


def _is_exposed_upper_body_evidence(evidence: dict[str, Any]) -> bool:
    return _is_exposed_upper_body_prompt(str(evidence.get("label") or ""))


def _backfill_exposed_upper_body_policy(violation: dict[str, Any]) -> None:
    if str(violation.get("category") or "").strip().lower() != SafetyCategory.OTHER_NSFW.value:
        return
    if not _is_exposed_upper_body_violation(violation):
        return
    violation["risk_subtype"] = "exposed_upper_body"
    violation["decision_hint"] = "redact_only"
    if not str(violation.get("entity_label_en") or ""):
        violation["entity_label_en"] = "bare torso"
    if not str(violation.get("entity_label_zh") or ""):
        violation["entity_label_zh"] = "裸露上身"
    if not str(violation.get("redaction_target") or ""):
        violation["redaction_target"] = "torso_without_head"
    prompts = _expand_category_prompts(
        SafetyCategory.OTHER_NSFW.value,
        violation.get("sam_prompt_texts") or [violation.get("sam_prompt_text") or violation.get("entity_label_en") or "bare torso"],
    )
    violation["sam_prompt_texts"] = prompts
    if not str(violation.get("sam_prompt_text") or ""):
        violation["sam_prompt_text"] = prompts[0] if prompts else "bare torso"


def _upper_body_candidate_can_bypass_local_rejection(
    evidence: dict[str, Any],
    violation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> bool:
    if not _is_exposed_upper_body_violation(violation) or not _is_exposed_upper_body_evidence(evidence):
        return False
    if _local_review_explicitly_rejects_torso(evidence.get("local_review")):
        return False
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(item) for item in bbox):
        return False
    confidence = float(evidence.get("confidence") or 0.0)
    if confidence < 0.45:
        return False
    area_ratio = _bbox_image_area_ratio([float(item) for item in bbox], image_width, image_height)
    if area_ratio < 0.01 or area_ratio > 0.85:
        return False
    center = violation.get("center_point")
    if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
        return _point_near_bbox(float(center[0]), float(center[1]), [float(item) for item in bbox], expand=1.50)
    rough = _bbox(violation.get("rough_bbox"))
    if rough is not None:
        return _bbox_iou([float(item) for item in bbox], rough) >= 0.03
    return True


def _upper_body_review_rejects_candidate(evidence: dict[str, Any], violation: dict[str, Any]) -> bool:
    if not _is_exposed_upper_body_violation(violation):
        return False
    review = evidence.get("local_review")
    if _local_review_explicitly_rejects_torso(review):
        return True
    if not isinstance(review, dict):
        return False
    if review.get("is_target_region") is False:
        return True
    main_region = str(review.get("main_region_type") or "").strip().lower()
    wrong_region = str(review.get("wrong_region_type") or "").strip().lower()
    negative_types = {"arm", "hand", "shoulder", "head", "face", "cap", "hat", "background", "background_person", "limb", "skin_patch"}
    if main_region in negative_types or wrong_region in negative_types:
        return True
    return False


def _upper_body_candidate_is_limb_like(evidence: dict[str, Any], image_width: int, image_height: int) -> bool:
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(_is_number(item) for item in bbox):
        return False
    x, y, w, h = [float(item) for item in bbox]
    if w <= 1.0 or h <= 1.0:
        return True
    area_ratio = _bbox_image_area_ratio([x, y, w, h], image_width, image_height)
    aspect = max(w / h, h / w)
    prompt = str(evidence.get("label") or "").lower()
    if area_ratio < 0.008:
        return True
    if aspect >= 3.2 and ("torso" not in prompt and "body" not in prompt):
        return True
    if aspect >= 4.0:
        return True
    return False


def _apply_upper_body_global_coverage(evidence: dict[str, Any], violation: dict[str, Any]) -> None:
    if not _is_exposed_upper_body_violation(violation):
        return
    bbox = _bbox(evidence.get("bbox"))
    expected = _bbox(violation.get("rough_bbox"))
    if bbox is None or expected is None:
        return
    expected_area = max(1.0, expected[2] * expected[3])
    actual_area = max(1.0, bbox[2] * bbox[3])
    inter_ratio = _bbox_intersection_area(bbox, expected) / expected_area
    actual_ratio = actual_area / expected_area
    centers = _center_points(violation.get("center_points") or [])
    covered_centers = sum(1 for point in centers if _point_near_bbox(float(point[0]), float(point[1]), bbox, expand=0.12))
    center_ratio = covered_centers / float(len(centers)) if centers else 1.0
    partial = actual_ratio < 0.35 or inter_ratio < 0.35 or center_ratio < 0.50
    evidence["global_target_coverage"] = "partial" if partial else "complete"
    evidence["global_coverage_metrics"] = {
        "actual_area_over_expected": round(actual_ratio, 4),
        "intersection_over_expected": round(inter_ratio, 4),
        "covered_center_points": covered_centers,
        "total_center_points": len(centers),
        "expected_bbox": expected,
        "actual_bbox": bbox,
    }
    if partial:
        evidence["boundary_status"] = "truncated"
        evidence["review_required"] = True
        review = dict(evidence.get("local_review") or {})
        review["global_target_coverage"] = "partial"
        review.setdefault("missing_parts", ["完整裸露躯干覆盖不足"])
        evidence["local_review"] = review


def _bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _local_review_explicitly_rejects_torso(review: Any) -> bool:
    if not isinstance(review, dict):
        return False
    text = " ".join(
        str(review.get(key) or "").lower()
        for key in ("reason_zh", "reason", "entity_label_en", "entity_label_zh", "boundary_status")
    )
    negative_terms = (
        "仅显示一只手臂",
        "只显示一只手臂",
        "仅显示手臂",
        "只显示手臂",
        "未包含躯干",
        "不包含躯干",
        "未包含裸露上身",
        "没有躯干",
        "只有手臂",
        "手臂",
        "arm only",
        "only arm",
        "no torso",
        "not torso",
        "without torso",
    )
    return any(term in text for term in negative_terms)


def _explicit_violation_is_body_scope(violation: dict[str, Any]) -> bool:
    values = [
        str(violation.get("entity_label_en") or ""),
        str(violation.get("sam_prompt_text") or ""),
    ]
    values.extend(str(value or "") for value in violation.get("sam_prompt_texts") or [])
    return any(_is_explicit_body_prompt(value) for value in values)


def _explicit_allows_large_bbox(evidence: dict[str, Any], violation: dict[str, Any]) -> bool:
    prompt = str(evidence.get("label") or violation.get("entity_label_en") or "")
    return _explicit_violation_is_body_scope(violation) and _is_explicit_body_prompt(prompt)


def _explicit_body_violation_has_tiny_local_candidate(
    evidence: dict[str, Any],
    violation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> bool:
    if str(violation.get("category") or "").lower() != SafetyCategory.EXPLICIT.value:
        return False
    if not _explicit_violation_is_body_scope(violation):
        return False
    label = str(evidence.get("label") or "").lower()
    if not _is_explicit_local_prompt(label):
        return False
    bbox = evidence.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    area_ratio = _bbox_image_area_ratio(bbox, image_width, image_height)
    rough = _bbox(violation.get("rough_bbox"))
    if rough is not None:
        rough_area = max(1.0, float(rough[2]) * float(rough[3]))
        candidate_area = max(1.0, float(bbox[2]) * float(bbox[3]))
        if candidate_area < rough_area * 0.30:
            return True
    return area_ratio < 0.12


def _prompt_specificity_bonus(category: str, prompt: str) -> float:
    text = str(prompt or "").lower()
    if category == SafetyCategory.DANGEROUS.value:
        exact_terms = ("pistol", "handgun", "firearm", "rifle", "gun", "knife", "手枪", "枪械", "枪支", "步枪", "刀具")
        return 0.25 if any(term in text for term in exact_terms) else -0.25
    if category == SafetyCategory.EXPLICIT.value:
        if _is_explicit_context_prompt(text):
            return -0.5
        if _is_explicit_body_prompt(text):
            return 0.22
        if _is_explicit_local_prompt(text):
            return 0.16
    if category == SafetyCategory.OTHER_NSFW.value and _is_exposed_upper_body_prompt(text):
        return 0.20
    return 0.0


def _bbox_image_area_ratio(bbox: list[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    return max(0.0, bbox[2]) * max(0.0, bbox[3]) / float(width * height)


def _local_bbox_is_usable(
    local_bbox: Any,
    crop_info: dict[str, Any],
    original_bbox: list[float],
    violation: dict[str, Any],
) -> bool:
    if _is_exposed_upper_body_violation(violation):
        return False
    if not isinstance(local_bbox, list) or len(local_bbox) != 4 or not all(_is_number(item) for item in local_bbox):
        return False
    lx, ly, lw, lh = [float(item) for item in local_bbox]
    crop_bbox = crop_info.get("crop_bbox")
    if not isinstance(crop_bbox, list) or len(crop_bbox) != 4:
        return False
    cw, ch = float(crop_bbox[2]), float(crop_bbox[3])
    if lw <= 1.0 or lh <= 1.0 or cw <= 1.0 or ch <= 1.0:
        return False
    overflow_x = max(0.0, -lx) + max(0.0, lx + lw - cw)
    overflow_y = max(0.0, -ly) + max(0.0, ly + lh - ch)
    if overflow_x > cw * 0.10 or overflow_y > ch * 0.10:
        return False
    global_bbox = [float(crop_bbox[0]) + max(0.0, lx), float(crop_bbox[1]) + max(0.0, ly), min(lw, cw), min(lh, ch)]
    if original_bbox and _bbox_iou(global_bbox, original_bbox) <= 0.02:
        return False
    center = violation.get("center_point")
    if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
        if _normalized_point_bbox_distance(float(center[0]), float(center[1]), global_bbox, int(crop_bbox[2] + crop_bbox[0]), int(crop_bbox[3] + crop_bbox[1])) > 0.55:
            return False
    return True


def _coarse_evidence_from_violation(violation: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    bbox = _bbox(violation.get("rough_bbox"))
    if bbox is None:
        return None
    x, y, w, h = bbox
    x = max(0.0, min(x, float(width)))
    y = max(0.0, min(y, float(height)))
    w = max(1.0, min(w, float(width) - x))
    h = max(1.0, min(h, float(height) - y))
    return {
        "category": str(violation.get("category") or SafetyCategory.OTHER_NSFW.value),
        "label": str(violation.get("entity_label_en") or violation.get("entity_label_zh") or "safety risk"),
        "bbox": [x, y, w, h],
        "description": str(violation.get("risk_reason_zh") or violation.get("visual_attributes_zh") or ""),
        "confidence": float(violation.get("confidence") or 0.5) if _is_number(violation.get("confidence")) else 0.5,
        "source": "qwen_rough_bbox_fallback",
        "entity_label_en": str(violation.get("entity_label_en") or ""),
        "entity_label_zh": str(violation.get("entity_label_zh") or ""),
        "risk_subtype": str(violation.get("risk_subtype") or ""),
        "decision_hint": str(violation.get("decision_hint") or ""),
        "redaction_target": str(violation.get("redaction_target") or ""),
        "center_point": violation.get("center_point"),
        "violation_id": violation.get("violation_id"),
        "localization_status": "coarse_localization_from_qwen_rough_bbox",
        "has_mask": False,
        "has_polygon": False,
        "mask_quality_score": 0.15,
    }


def _annotate_evidence_region_quality(
    evidence: dict[str, Any],
    image_width: int = 0,
    image_height: int = 0,
) -> None:
    bbox = _bbox(evidence.get("bbox"))
    confidence = float(evidence.get("confidence") or 0.0) if _is_number(evidence.get("confidence")) else 0.0
    has_mask = bool(evidence.get("mask_path"))
    has_polygon = bool(evidence.get("polygon") or evidence.get("polygons"))
    score = 0.25 + min(confidence, 1.0) * 0.35
    if has_mask:
        score += 0.25
    elif has_polygon:
        score += 0.15
    source = str(evidence.get("source") or "")
    if source == "qwen_rough_bbox_fallback":
        score = min(score, 0.25)
    boundary = str(evidence.get("boundary_status") or "").lower()
    if boundary == "complete":
        score += 0.10
    elif boundary == "truncated":
        score -= 0.18
    elif boundary == "uncertain":
        score -= 0.10
    if image_width > 0 and image_height > 0 and bbox is not None:
        area_ratio = _bbox_image_area_ratio(bbox, image_width, image_height)
        if area_ratio > 0.85:
            score -= 0.20
        elif area_ratio < 0.0005:
            score -= 0.10
    mask_fill = evidence.get("mask_bbox_fill_ratio")
    if _is_number(mask_fill):
        fill = float(mask_fill)
        if 0.02 <= fill <= 0.95:
            score += 0.05
        elif fill > 1.25:
            score -= 0.10
    evidence["has_mask"] = has_mask
    evidence["has_polygon"] = has_polygon
    evidence["mask_quality_score"] = round(max(0.0, min(1.0, score)), 4)


def _expand_bbox(bbox: list[float], width: int, height: int, ratio: float = 0.75) -> list[float]:
    if len(bbox) != 4:
        return bbox
    x, y, w, h = [float(item) for item in bbox]
    pad_x = max(12.0, w * ratio)
    pad_y = max(12.0, h * ratio)
    x1 = max(0.0, x - pad_x)
    y1 = max(0.0, y - pad_y)
    x2 = min(float(width), x + w + pad_x)
    y2 = min(float(height), y + h + pad_y)
    return [x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)]


def _refined_bbox_is_reliable(
    original_bbox: list[float],
    refined_bbox: list[float],
    violation: dict[str, Any],
    image_width: int,
    image_height: int,
) -> bool:
    if len(original_bbox) != 4 or len(refined_bbox) != 4:
        return False
    if refined_bbox[2] <= 1.0 or refined_bbox[3] <= 1.0:
        return False
    category = str(violation.get("category") or "").lower()
    refined_area = max(1.0, refined_bbox[2] * refined_bbox[3])
    original_area = max(1.0, original_bbox[2] * original_bbox[3])
    if refined_area < original_area * (0.08 if category == SafetyCategory.DANGEROUS.value else 0.04):
        return False
    if _bbox_iou(original_bbox, refined_bbox) <= 0.01 and _normalized_bbox_center_distance(original_bbox, refined_bbox, image_width, image_height) > 0.12:
        return False
    center = violation.get("center_point")
    if isinstance(center, list) and len(center) == 2 and _is_number(center[0]) and _is_number(center[1]):
        if not _point_near_bbox(float(center[0]), float(center[1]), refined_bbox, expand=1.5):
            return False
    if category == SafetyCategory.DANGEROUS.value:
        min_side = min(refined_bbox[2], refined_bbox[3])
        if min_side < 8.0 and refined_area < original_area * 0.35:
            return False
    return True


def _point_near_bbox(x: float, y: float, bbox: list[float], expand: float = 0.75) -> bool:
    bx, by, bw, bh = [float(item) for item in bbox]
    pad_x = max(8.0, bw * expand)
    pad_y = max(8.0, bh * expand)
    return bx - pad_x <= x <= bx + bw + pad_x and by - pad_y <= y <= by + bh + pad_y


def _normalized_bbox_center_distance(a: list[float], b: list[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    ax = float(a[0]) + float(a[2]) / 2.0
    ay = float(a[1]) + float(a[3]) / 2.0
    bx = float(b[0]) + float(b[2]) / 2.0
    by = float(b[1]) + float(b[3]) / 2.0
    dx = abs(ax - bx) / float(width)
    dy = abs(ay - by) / float(height)
    return (dx * dx + dy * dy) ** 0.5


def _normalized_point_bbox_distance(x: float, y: float, bbox: list[float], width: int, height: int) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    cx = float(bbox[0]) + float(bbox[2]) / 2.0
    cy = float(bbox[1]) + float(bbox[3]) / 2.0
    dx = abs(cx - x) / float(width)
    dy = abs(cy - y) / float(height)
    return (dx * dx + dy * dy) ** 0.5


def _crop_for_local_review(
    image_path: str,
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    try:
        from PIL import Image

        x, y, w, h = [float(item) for item in bbox]
        area_ratio = _bbox_image_area_ratio(bbox, image_width, image_height)
        if area_ratio <= 0.02:
            pad = 1.2
        elif area_ratio <= 0.12:
            pad = 0.7
        else:
            pad = 0.35
        x1 = max(0.0, x - w * pad)
        y1 = max(0.0, y - h * pad)
        x2 = min(float(image_width), x + w * (1.0 + pad))
        y2 = min(float(image_height), y + h * (1.0 + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        crop_dir = Path(image_path).parent / "safety_crops"
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"safety_crop_{abs(hash((image_path, tuple(bbox)))):x}.jpg"
        with Image.open(Path(image_path)) as image:
            image.convert("RGB").crop((int(x1), int(y1), int(x2), int(y2))).save(crop_path, format="JPEG", quality=90)
        return {"crop_path": str(crop_path), "crop_bbox": [x1, y1, x2 - x1, y2 - y1]}
    except Exception:
        return None


def _center_in_expanded_hint(
    bbox: list[float],
    category: str,
    hints: list[dict[str, Any]],
    expand: float = 0.75,
) -> bool:
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    for hint in hints:
        if category and str(hint.get("category") or "").lower() not in {"", category.lower()}:
            continue
        hint_bbox = hint.get("bbox")
        if not isinstance(hint_bbox, list) or len(hint_bbox) != 4:
            continue
        hx, hy, hw, hh = [float(item) for item in hint_bbox]
        pad_x = hw * expand
        pad_y = hh * expand
        if hx - pad_x <= cx <= hx + hw + pad_x and hy - pad_y <= cy <= hy + hh + pad_y:
            return True
    return False


def _qwen_hint_evidence(categories: list[str], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_regions = metadata.get("evidence_regions") if isinstance(metadata.get("evidence_regions"), list) else []
    hints: list[dict[str, Any]] = []
    for item in evidence_regions:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").lower()
        if category and category not in {value.lower() for value in categories}:
            continue
        hint = {
            "category": str(item.get("category") or ""),
            "label": str(item.get("label") or ""),
            "description": str(item.get("description") or ""),
            "confidence": item.get("confidence", 0.0),
            "source": "qwen_hint_only",
        }
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                hint["bbox"] = [float(value) for value in bbox]
            except (TypeError, ValueError):
                pass
        hints.append(hint)
    return hints


def _best_hint_iou(bbox: list[float], category: str, hints: list[dict[str, Any]]) -> float:
    best = 0.0
    for hint in hints:
        if category and str(hint.get("category") or "").lower() not in {"", category.lower()}:
            continue
        hint_bbox = hint.get("bbox")
        if not isinstance(hint_bbox, list) or len(hint_bbox) != 4:
            continue
        best = max(best, _bbox_iou(bbox, hint_bbox))
    return round(best, 4)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ar, ab = ax + aw, ay + ah
    br, bb = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ar, br), min(ab, bb)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(Path(image_path)) as image:
            return image.size
    except Exception:
        return 0, 0


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _center_points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        if isinstance(item, list) and len(item) == 2 and _is_number(item[0]) and _is_number(item[1]):
            points.append([float(item[0]), float(item[1])])
    return points


def _normalize_violation_geometry(violation: dict[str, Any], image_width: int, image_height: int) -> dict[str, Any]:
    return violation


def _violation_looks_like_qwen_1000_space(violation: dict[str, Any]) -> bool:
    values: list[float] = []
    center = violation.get("center_point")
    if isinstance(center, list) and len(center) == 2 and all(_is_number(item) for item in center):
        values.extend(float(item) for item in center)
    for point in _center_points(violation.get("center_points") or []):
        values.extend(point)
    bbox = _bbox(violation.get("rough_bbox"))
    if bbox is not None:
        x, y, w, h = bbox
        values.extend([x, y, w, h, x + w, y + h])
    if not values:
        return False
    return min(values) >= -25.0 and max(values) <= 1125.0


def _scale_point(value: Any, scale_x: float, scale_y: float) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(item) for item in value):
        return None
    return [float(value[0]) * scale_x, float(value[1]) * scale_y]


def _scale_bbox(value: Any, scale_x: float, scale_y: float, image_width: int, image_height: int) -> list[float] | None:
    bbox = _bbox(value)
    if bbox is None:
        return None
    x, y, w, h = bbox
    x *= scale_x
    y *= scale_y
    w *= scale_x
    h *= scale_y
    x = max(0.0, min(x, float(image_width)))
    y = max(0.0, min(y, float(image_height)))
    w = max(1.0, min(w, float(image_width) - x))
    h = max(1.0, min(h, float(image_height) - y))
    return [x, y, w, h]


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(_is_number(item) for item in value):
        return None
    return [float(item) for item in value]

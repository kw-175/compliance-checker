from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import httpx

from picture.domain.enums import FindingType
from picture.domain.models import PictureFinding
from picture.providers.base import VisionDetector
from picture.providers.safety.qwen35_vl import _extract_json, _image_data_url, _is_number
from picture.providers.vision.privacy_specialists import VisualPrivacySpecialistDetectors
from picture.providers.vision.sam3 import SAM3SensitiveObjectDetector, _dedupe_findings

logger = logging.getLogger(__name__)

SENSITIVE_OBJECT_SCHEMA = {
    "face": "人脸、证件照人脸、可识别人物面部",
    "id_card": "身份证、学生证、准考证等证件",
    "badge": "胸牌、工牌、校牌、姓名牌",
    "signature": "手写签名、签字",
    "stamp": "公章、学校印章、机构印章",
    "qr_code": "二维码",
    "barcode": "条形码",
    "license_plate": "车牌",
    "avatar": "账号头像、用户头像",
    "account_region": "用户名、账号 ID、个人主页头部",
    "school_class_identifier": "学校名称、班级名称、教室门牌、学校标志",
}

OPERATOR_IDS = {
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

SPECIALIST_FAST_PATH_CATEGORIES = {"qr_code", "barcode"}
SPECIALIST_ANCHORED_CATEGORIES = {"face"}

SYSTEM_PROMPT = """你是教育数据集图片视觉隐私审核员。
只判断图片中是否存在需要脱敏的视觉敏感对象，不判断色情、暴力等内容安全问题。
face 类别只表示“可识别真实人物面部”，不是所有头部或脸形区域。后脑勺、背影、头发、极小远景脸、只露一点侧脸、严重遮挡/模糊且无法识别具体个人的区域，必须 present=false 或不要作为 face 输出。
请只返回 JSON，不要 Markdown。
输出格式：
{
  "sensitive_objects": [
    {
      "category": "face",
      "present": true,
      "confidence": 0.92,
      "requires_redaction": true,
      "object_name_zh": "学生人脸",
      "location_hint": "左上区域",
      "description": "可识别学生人脸",
      "suggested_prompt_for_sam3": "student face"
    }
  ],
  "review_required": false,
  "explanation": "简短说明"
}
先开放式观察图片中所有可能泄露身份、账号、机构、二维码跳转、证件、签名、印章、车牌、屏幕账号信息的视觉对象，再映射到用户给出的候选 category。
category 只能从用户给出的候选类别中选择。无法确定时 present=false 或降低 confidence。
如果存在敏感对象但难以精确定位，也必须 present=true，并给出 location_hint 与 suggested_prompt_for_sam3。
"""


@dataclass(frozen=True)
class SemanticObject:
    category: str
    present: bool
    confidence: float
    requires_redaction: bool
    object_name_zh: str = ""
    location_hint: str = ""
    description: str = ""
    suggested_prompt_for_sam3: str = ""


class QwenSAM3FusionVisionDetector(VisionDetector):
    """Qwen3.5 semantic recall + SAM3 localization for visual sensitive objects."""

    def __init__(
        self,
        model_dir: str,
        confidence_threshold: float = 0.35,
        device: str = "auto",
        semantic_threshold: float = 0.55,
        sam3_keep_without_qwen_threshold: float = 0.75,
        qwen_timeout_seconds: float = 90.0,
        qwen_max_tokens: int = 768,
        image_max_side: int = 1280,
        image_jpeg_quality: int = 85,
        sam3_detector: Any | None = None,
        specialist_detectors: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._sam3 = sam3_detector or SAM3SensitiveObjectDetector(
            model_dir=model_dir,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self._semantic_threshold = semantic_threshold
        self._sam3_keep_without_qwen_threshold = sam3_keep_without_qwen_threshold
        self._qwen_timeout_seconds = qwen_timeout_seconds
        self._qwen_max_tokens = qwen_max_tokens
        self._image_max_side = image_max_side
        self._image_jpeg_quality = image_jpeg_quality
        self._specialists = specialist_detectors or VisualPrivacySpecialistDetectors()
        self._provider: Any | None = None
        self._provider_lock = threading.Lock()
        self._kwargs = kwargs

    @property
    def name(self) -> str:
        return "Qwen3.5+SAM3FusionVisionDetector"

    def _get_predictor(self) -> Any:
        loader = getattr(self._sam3, "_get_predictor", None)
        if loader is not None:
            return loader()
        warmup = getattr(self._sam3, "warmup", None)
        if warmup is not None:
            return warmup()
        return None

    def detect(
        self,
        image_path: str,
        target_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> list[PictureFinding]:
        targets = _normalize_targets(target_types)
        specialist_findings = self._run_specialist_detection(image_path, targets)
        specialist_categories = {
            str(finding.category or "").lower()
            for finding in specialist_findings
            if finding.region is not None
        }
        qwen_targets = [
            category for category in targets
            if category not in SPECIALIST_FAST_PATH_CATEGORIES
        ]
        if qwen_targets:
            semantic_objects, qwen_error = self._semantic_detect(image_path, qwen_targets)
        else:
            semantic_objects, qwen_error = [], ""
        present = {
            item.category: item
            for item in semantic_objects
            if item.present and item.requires_redaction and item.confidence >= self._semantic_threshold
        }
        sam3_targets = sorted(
            category for category in (set(qwen_targets) | set(present))
            if category not in SPECIALIST_FAST_PATH_CATEGORIES
            and category not in SPECIALIST_ANCHORED_CATEGORIES
        )
        extra_prompts = _dynamic_prompts(present)
        confidence_thresholds = {
            category: min(0.20 if category == "face" else 0.25, self._sam3._confidence_threshold)
            for category in present
        }
        sam3_findings = (
            self._run_sam3_detection(
                image_path,
                sam3_targets,
                extra_prompts,
                confidence_thresholds,
            )
            if sam3_targets
            else []
        )

        fused: list[PictureFinding] = [_with_specialist_metadata(finding, self.name) for finding in specialist_findings]
        localized_categories: set[str] = set()
        localized_categories.update(specialist_categories)
        for finding in sam3_findings:
            category = str(finding.category or "").lower()
            if category in SPECIALIST_ANCHORED_CATEGORIES:
                logger.info("Drop SAM3-only specialist-anchored visual finding: category=%s", category)
                continue
            semantic = present.get(category)
            if semantic is not None:
                localized_categories.add(category)
                fused.append(_with_fusion_metadata(finding, self.name, semantic, qwen_error, confirmed=True))
                continue
            if finding.score >= self._sam3_keep_without_qwen_threshold:
                fused.append(_with_fusion_metadata(finding, self.name, None, qwen_error, confirmed=False))
            else:
                logger.info(
                    "Drop unconfirmed SAM3 visual finding below fusion threshold: category=%s score=%.3f",
                    finding.category,
                    finding.score,
                )

        for category, semantic in present.items():
            if category in localized_categories:
                continue
            fused.append(_unlocalized_finding(semantic, self.name, qwen_error))

        return _dedupe_findings(fused)

    def _run_specialist_detection(self, image_path: str, targets: list[str]) -> list[PictureFinding]:
        if not targets:
            return []
        try:
            return self._specialists.detect(image_path, targets)
        except Exception as exc:
            logger.warning("Visual privacy specialist detection failed; continue with Qwen/SAM3: %s", exc)
            return []

    def _get_provider(self) -> Any:
        if self._provider is None:
            with self._provider_lock:
                if self._provider is not None:
                    return self._provider
                from text.api_clients import resolve_provider_config
                from text.config.settings import get_settings as get_text_settings

                self._provider = resolve_provider_config(get_text_settings())
        return self._provider

    def _semantic_detect(self, image_path: str, targets: list[str]) -> tuple[list[SemanticObject], str]:
        if not targets:
            return [], ""
        try:
            provider = self._get_provider()
            url = _chat_completions_url(provider.base_url)
            headers = {"Content-Type": "application/json"}
            if provider.api_key:
                headers["Authorization"] = f"Bearer {provider.api_key}"
            candidates = {key: SENSITIVE_OBJECT_SCHEMA.get(key, key) for key in targets}
            payload = {
                "model": provider.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                            "text": (
                                "请先开放式观察图片中所有可能需要脱敏的视觉敏感对象，再映射到候选类别并返回 JSON。"
                                "如果目标存在但不确定坐标，不要漏报，给出 location_hint 和 suggested_prompt_for_sam3。"
                                "特别注意：face 只输出可识别真实人脸；后脑、背影、头发、极小远景脸、只露一点侧脸、严重遮挡或模糊到无法识别个人的脸形区域不要输出为 face。"
                                f"候选类别说明：{candidates}"
                            ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": _image_data_url(
                                        image_path,
                                        max_side=self._image_max_side,
                                        jpeg_quality=self._image_jpeg_quality,
                                    )
                                },
                            },
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": min(int(provider.max_tokens), self._qwen_max_tokens),
                "response_format": {"type": "json_object"},
            }
            response = httpx.post(url, headers=headers, json=payload, timeout=self._qwen_timeout_seconds)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return _parse_semantic_objects(_extract_json(content), set(targets)), ""
        except Exception as exc:
            logger.warning("Qwen visual sensitive semantic detection failed; fallback to SAM3 only: %s", exc)
            return [], f"{type(exc).__name__}: {exc}"

    def _run_sam3_detection(
        self,
        image_path: str,
        sam3_targets: list[str],
        extra_prompts: dict[str, list[str]],
        confidence_thresholds: dict[str, float],
    ) -> list[PictureFinding]:
        # Tests and some external integrations monkeypatch ``detect`` directly.
        # Honor that override while using the richer dynamic-prompt API in normal runtime.
        if "detect" in getattr(self._sam3, "__dict__", {}):
            return self._sam3.detect(image_path, target_types=sam3_targets)
        return self._sam3.detect_with_prompts(
            image_path,
            target_types=sam3_targets,
            extra_prompts=extra_prompts,
            confidence_thresholds=confidence_thresholds,
        )


def _normalize_targets(target_types: list[str] | set[str] | tuple[str, ...] | None) -> list[str]:
    if not target_types:
        return [
            "face",
            "id_card",
            "badge",
            "qr_code",
            "barcode",
            "signature",
            "stamp",
        ]
    normalized = sorted({str(item).strip().lower().replace(".", "_").replace("-", "_") for item in target_types if str(item).strip()})
    return [item for item in normalized if item in SENSITIVE_OBJECT_SCHEMA]


def _parse_semantic_objects(payload: dict[str, Any], allowed: set[str]) -> list[SemanticObject]:
    raw_items = payload.get("sensitive_objects")
    if not isinstance(raw_items, list):
        raw_items = payload.get("objects")
    if not isinstance(raw_items, list):
        return []
    objects: list[SemanticObject] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip().lower().replace(".", "_").replace("-", "_")
        if category not in allowed:
            continue
        confidence = float(item.get("confidence")) if _is_number(item.get("confidence")) else 0.0
        objects.append(
            SemanticObject(
                category=category,
                present=bool(item.get("present", False)),
                confidence=max(0.0, min(1.0, confidence)),
                requires_redaction=bool(item.get("requires_redaction", item.get("present", False))),
                object_name_zh=str(item.get("object_name_zh") or item.get("object_name") or ""),
                location_hint=str(item.get("location_hint") or ""),
                description=str(item.get("description") or ""),
                suggested_prompt_for_sam3=str(item.get("suggested_prompt_for_sam3") or item.get("sam3_prompt") or ""),
            )
        )
    return objects


def _dynamic_prompts(present: dict[str, SemanticObject]) -> dict[str, list[str]]:
    prompts: dict[str, list[str]] = {}
    for category, semantic in present.items():
        items = prompts.setdefault(category, [])
        for value in (
            semantic.suggested_prompt_for_sam3,
            semantic.object_name_zh,
            SENSITIVE_OBJECT_SCHEMA.get(category, ""),
        ):
            for prompt in _prompt_fragments(value):
                if not _prompt_allowed_for_category(category, prompt):
                    continue
                if prompt not in items:
                    items.append(prompt)
    return prompts


def _prompt_fragments(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    fragments = [text]
    # Keep prompts short for SAM3 token limits while preserving useful object words.
    for sep in ("、", "，", ",", "/", "；", ";"):
        if sep in text:
            fragments.extend(part.strip() for part in text.split(sep) if part.strip())
    return [item for item in fragments if 1 < len(item) <= 32 and not _looks_like_location_or_sentence(item)]


def _prompt_allowed_for_category(category: str, prompt: str) -> bool:
    text = str(prompt or "").strip().lower()
    if not text or _looks_like_location_or_sentence(text):
        return False
    if category == "face":
        allowed_terms = (
            "face",
            "facial",
            "portrait",
            "人脸",
            "面部",
            "脸部",
            "证件照",
            "头像",
        )
        blocked_terms = (
            "person",
            "people",
            "human",
            "head",
            "body",
            "leg",
            "arm",
            "hand",
            "人物",
            "人体",
            "头部",
            "身体",
            "腿",
            "胳膊",
            "手",
        )
        if any(term in text for term in blocked_terms) and not any(term in text for term in allowed_terms):
            return False
        return any(term in text for term in allowed_terms)
    return True


def _looks_like_location_or_sentence(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    location_terms = (
        "左上",
        "右上",
        "左下",
        "右下",
        "上方",
        "下方",
        "左侧",
        "右侧",
        "中间",
        "中央",
        "区域",
        "附近",
        "角落",
    )
    sentence_markers = (
        "属于",
        "存在",
        "可见",
        "可辨识",
        "需要",
        "可能",
        "虽然",
        "但是",
        "但",
        "仍",
        "隐私",
        "身份",
        "对象",
        "部分",
        "轮廓",
        "特征",
    )
    if any(term in text for term in location_terms):
        return True
    if any(term in text for term in sentence_markers) and len(text) > 6:
        return True
    return False


def _with_fusion_metadata(
    finding: PictureFinding,
    provider_name: str,
    semantic: SemanticObject | None,
    qwen_error: str,
    confirmed: bool,
) -> PictureFinding:
    metadata = dict(finding.metadata or {})
    metadata.update(
        {
            "fusion_provider": provider_name,
            "operator_id": metadata.get("operator_id") or OPERATOR_IDS.get(str(finding.category or "").lower(), ""),
            "qwen_semantic_confirmed": confirmed,
            "sam3_localized": finding.region is not None,
        }
    )
    source_detectors = list(metadata.get("source_detectors") or [])
    if "sam3" not in source_detectors:
        source_detectors.append("sam3")
    if semantic is not None and "qwen3.5" not in source_detectors:
        source_detectors.append("qwen3.5")
    metadata["source_detectors"] = source_detectors
    explanation = finding.explanation
    score = finding.score
    if semantic is not None:
        score = max(float(finding.score), semantic.confidence)
        metadata.update(
            {
                "qwen_confidence": semantic.confidence,
                "qwen_object_name_zh": semantic.object_name_zh,
                "qwen_location_hint": semantic.location_hint,
                "qwen_description": semantic.description,
                "qwen_sam3_prompt": semantic.suggested_prompt_for_sam3,
            }
        )
        explanation = semantic.description or finding.explanation
    else:
        metadata["review_required"] = True
    if qwen_error:
        metadata["qwen_semantic_error"] = qwen_error
    return finding.model_copy(
        update={
            "score": score,
            "provider": provider_name,
            "explanation": explanation,
            "metadata": metadata,
        }
    )


def _with_specialist_metadata(finding: PictureFinding, provider_name: str) -> PictureFinding:
    metadata = dict(finding.metadata or {})
    category = str(finding.category or "").lower()
    source_detectors = list(metadata.get("source_detectors") or [])
    metadata.update(
        {
            "fusion_provider": provider_name,
            "operator_id": metadata.get("operator_id") or OPERATOR_IDS.get(category, ""),
            "source_detectors": source_detectors,
            "qwen_semantic_confirmed": category in SPECIALIST_FAST_PATH_CATEGORIES or category == "face",
            "sam3_localized": False,
            "review_required": bool(metadata.get("review_required", False)),
            "localization_status": metadata.get("localization_status") or "localized_by_specialist",
        }
    )
    return finding.model_copy(
        update={
            "provider": provider_name,
            "metadata": metadata,
        }
    )


def _unlocalized_finding(semantic: SemanticObject, provider_name: str, qwen_error: str) -> PictureFinding:
    metadata: dict[str, Any] = {
        "fusion_provider": provider_name,
        "operator_id": OPERATOR_IDS.get(semantic.category, ""),
        "source_detectors": ["qwen3.5"],
        "qwen_semantic_confirmed": True,
        "sam3_localized": False,
        "localization_required": True,
        "review_required": True,
        "qwen_confidence": semantic.confidence,
        "qwen_object_name_zh": semantic.object_name_zh,
        "qwen_location_hint": semantic.location_hint,
        "qwen_description": semantic.description,
        "qwen_sam3_prompt": semantic.suggested_prompt_for_sam3,
    }
    if qwen_error:
        metadata["qwen_semantic_error"] = qwen_error
    return PictureFinding(
        finding_type=FindingType.VISION_OBJECT,
        category=semantic.category,
        label=f"Qwen detected unlocalized {semantic.category}",
        score=semantic.confidence,
        reason_code=f"VISION_UNLOCALIZED_{semantic.category.upper()}",
        provider=provider_name,
        threshold_used=0.55,
        explanation=semantic.description or "Qwen3.5 判断存在视觉敏感对象，但 SAM3 未能定位区域，需要人工框选复核。",
        metadata=metadata,
    )


def _chat_completions_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"

from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import re
import threading
from pathlib import Path
from typing import Any

import httpx

from picture.domain.enums import SafetyCategory
from picture.domain.exceptions import ProviderNotAvailableError
from picture.domain.models import PictureModerationResult
from picture.providers.base import SafetyModerator

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是教育数据集图片合规审核员。
请只返回一个 JSON 对象，不要 Markdown，不要解释性外层文本。
字段：
{
  "is_safe": true,
  "categories": ["safe"],
  "scores": {"safe": 0.99},
  "reason_codes": [],
  "category_details": {
    "dangerous": {
      "risk_subtype_zh": "枪械/管制器具/毒品等更具体类型",
      "object_name_zh": "具体可见风险对象，例如疑似手枪",
      "scene_description_zh": "图片中与风险相关的可见画面",
      "risk_reason_zh": "为什么该画面不适合作为普通教育数据交付"
    }
  },
  "evidence_regions": [
    {"category": "dangerous", "label": "疑似手枪", "bbox": [x, y, w, h], "description": "区域证据说明", "confidence": 0.8}
  ],
  "violations": [
    {
      "category": "dangerous",
      "entity_label_en": "pistol",
      "entity_label_zh": "疑似手枪",
	      "sam_prompt_text": "pistol",
	      "sam_prompt_texts": ["pistol", "handgun", "firearm"],
	      "redaction_target": "pistol",
	      "center_point": [412, 578],
	      "center_points": [[412, 578]],
      "rough_bbox": [360, 540, 130, 60],
      "location_hint_zh": "人物右手附近",
      "visual_attributes_zh": "黑色短枪形物体",
      "risk_reason_zh": "图片中出现疑似手枪，属于危险违禁物品，不适合作为普通教育数据交付。",
      "confidence": 0.91
    }
  ],
    "review_required": false,
    "explanation": "简短说明"
  }
重点审核：色情裸露、未成年人性相关、血腥暴力、自伤自杀、武器危险行为、毒品违禁品、仇恨极端符号、违法犯罪场景、低俗不适内容。
categories 只能使用以下枚举值：safe、explicit、graphic_violence、hate_symbol、self_harm、dangerous、other_nsfw。
色情、裸露、性行为、性暗示、成人内容必须归为 explicit；低俗擦边或无法细分的不适内容归为 other_nsfw。
只要图片存在明显色情裸露或成人性内容，is_safe 必须为 false，categories 必须包含 explicit，scores.explicit 应不低于 0.8。
图片只要存在裸露上身、裸露胸腹、裸露背部或大面积裸露躯干，即使属于体育、健身、医学、舞蹈、艺术、比赛等正常非色情场景，也必须输出一个用于脱敏的 violations 对象；此时 is_safe 必须为 false，categories 使用 other_nsfw，scores.other_nsfw 建议 0.55 到 0.79，reason_codes 包含 exposed_upper_body_redaction_required，violation.decision_hint 必须为 redact_only，risk_subtype 必须为 exposed_upper_body，risk_reason_zh 说明“非色情场景下存在裸露身体区域，按数据交付策略需要局部脱敏”。这类结果只表示需要脱敏，不等同于色情，不应写成成人色情或性暗示。
如果存在武器、枪械、刀具、毒品、违法犯罪工具等危险内容，不要只写 dangerous，必须在 category_details.dangerous.object_name_zh 写出具体对象，例如“疑似手枪”“疑似步枪”“刀具”“毒品包装”。
如果存在色情裸露、血腥暴力、自伤、仇恨符号，也必须写出具体画面对象和自然语言原因。
如果发现违规内容，violations 必须按“一个具体物理违规实体一个对象”输出，entity_label_en 必须是具体英文物理名词，例如 pistol、handgun、rifle、knife、blood、wound、nazi symbol；禁止使用 weapon、dangerous object、unsafe content 这类泛化词。
每个 violation 必须给出 sam_prompt_texts，作为分割模型的英文短文本提示列表；这些提示必须符合 SAM3 的文本提示习惯：英文 short noun phrase，1 到 4 个词，只描述可见物体或可分割区域，不要写句子、动作、合规原因、排除条件或抽象风险词；sam_prompt_text 取列表第一个。
合格示例：pistol、handgun、knife、blood、wound、naked body、genital area、exposed body part、torso、buttocks、human body、hate symbol。
不合格示例：body below head、nude male except head、色情裸露区域、头部以下身体、不适合作为教育数据的区域、person holding weapon、dangerous object。
每个 violation 必须给出 center_points，坐标基于你当前看到的图片尺寸 [x,y]，用于给分割模型打点；center_point 取最核心的一个点。
每个 violation 尽量给出 rough_bbox [x,y,w,h]，坐标同样基于你当前看到的图片尺寸，作为粗空间提示；同时给出 risk_reason_zh，用于向用户展示违规原因。
色情裸露 explicit 不能只输出 nude male 这类抽象提示，必须拆成可分割区域提示，例如 naked body、genital area、exposed body part、nude torso、human body；如果存在生殖器、裸露臀部、裸露胸部等关键区域，应分别给出对应 center_points。
非色情裸露上身的 violations 应使用 category=other_nsfw、risk_subtype=exposed_upper_body、entity_label_en=bare torso 或 nude torso、entity_label_zh=裸露上身，sam_prompt_texts 优先使用 bare torso、nude torso、chest and abdomen、torso without head、skin area、human body，最后才使用 upper body；center_point 应给在裸露胸腹或背部中心，不要给在人脸和手臂。
无法稳定定位时可省略 center_points，但必须将 review_required 设为 true。
教育场景中的正常教材、作业、板书、实验器材、课堂照片，不应误判为不安全。
"""


LOCAL_REVIEW_SYSTEM_PROMPT = """你是视觉内容安全局部复核员。
你会看到一张从原图裁剪出的局部图片，以及前置系统怀疑的违规实体名称。
请只返回严格 JSON，不要 Markdown，不要额外解释。
判断标准：
1. 当前局部图是否确实包含前置系统指出的同一个违规实体，不是相似背景、肢体、衣物、玩具或其他无关物；
2. 是否误把手臂、衣物、背景、打火机、玩具或普通物品当成违规物；
3. 对枪械、刀具、毒品、符号等小目标，判断是否包含同一违规实体主体；
4. 对色情裸露 explicit，只要包含生殖器、裸露臀部、裸露胸部、大面积裸露皮肤等违规裸露区域，就应视为同一违规因素的一部分，不要求裁剪图包含完整人体；
5. 对 risk_subtype=exposed_upper_body 或 redaction_target 为 torso/chest/back 的非色情裸露上身，复核目标不是“是否有皮肤”，而是“当前区域是否主要覆盖裸露胸腹/背部躯干”。如果主要是手臂、肩膀、头部、泳帽、背景人物或局部无关皮肤，必须返回 is_authentic_violation=false，并在 main_region_type/wrong_region_type 中写明；
6. 可选给出局部图内只包裹该违规实体或目标脱敏区域的紧凑 bbox [x,y,w,h]，坐标基于当前裁剪图；该 bbox 只是几何建议，不要臆造；
7. 判断当前区域是否完整包裹目标；如果枪管、刀尖、裸露关键区域、血迹边缘、符号边缘等贴边或被截断，必须返回 truncated。
返回格式：
{
  "is_authentic_violation": true,
  "entity_label_en": "pistol",
  "entity_label_zh": "疑似手枪",
  "is_target_region": true,
  "main_region_type": "target",
  "wrong_region_type": "",
  "boundary_status": "complete",
  "local_bbox": [10, 20, 80, 40],
  "confidence": 0.88,
  "reason_zh": "局部图中可见黑色手枪形物体。"
}
boundary_status 只能是 complete、truncated、uncertain。
如果不是同一个违规实体，或非色情裸露上身候选不是裸露躯干目标区域，is_authentic_violation 必须为 false。
"""


class Qwen35VLSafetyModerator(SafetyModerator):
    """Visual safety moderator that reuses the text-compliance Qwen3.5 endpoint."""

    def __init__(
        self,
        timeout_seconds: float = 120.0,
        max_tokens: int = 384,
        image_max_side: int = 1280,
        image_jpeg_quality: int = 85,
    ) -> None:
        self._provider: Any | None = None
        self._provider_lock = threading.Lock()
        self._timeout_seconds = timeout_seconds
        self._max_tokens = max(max_tokens, 768)
        self._image_max_side = image_max_side
        self._image_jpeg_quality = image_jpeg_quality

    @property
    def name(self) -> str:
        model = self._provider.model if self._provider is not None else "text-compliance-endpoint"
        return f"Qwen3.5-VL(reuse:{model})"

    def _get_provider(self) -> Any:
        if self._provider is None:
            with self._provider_lock:
                if self._provider is not None:
                    return self._provider
                from text.api_clients import resolve_provider_config
                from text.config.settings import get_settings as get_text_settings

                self._provider = resolve_provider_config(get_text_settings())
        return self._provider

    def moderate(self, image_path: str) -> PictureModerationResult:
        provider = self._get_provider()
        url = self._chat_completions_url(provider.base_url)
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        image_payload = _image_data_payload(
            image_path,
            max_side=self._image_max_side,
            jpeg_quality=self._image_jpeg_quality,
        )
        qwen_w, qwen_h = image_payload["qwen_input_size"]

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
                                "请按教育数据集图片合规要求审核这张图片，并返回 JSON。"
                                f"你当前看到的图片尺寸是 {qwen_w} x {qwen_h} 像素。"
                                "所有 center_point、center_points、rough_bbox、evidence_regions.bbox 都必须基于这个当前可见图片尺寸输出，"
                                "不要输出归一化坐标，也不要输出原始大图坐标。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_payload["data_url"]},
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(int(provider.max_tokens), self._max_tokens),
            "response_format": {"type": "json_object"},
        }

        parse_error = ""
        raw_content = ""
        try:
            raw_content = self._request_content(url, headers, payload)
            data = _extract_json(raw_content)
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Qwen3.5-VL visual safety JSON parse failed; retrying once: %s; raw=%s",
                parse_error,
                _short_text(raw_content),
            )
            try:
                retry_payload = _retry_payload(payload)
                raw_content = self._request_content(url, headers, retry_payload)
                data = _extract_json(raw_content)
            except Exception as retry_exc:
                retry_error = f"{type(retry_exc).__name__}: {retry_exc}"
                logger.warning(
                    "Qwen3.5-VL visual safety JSON retry failed; degrading to manual review: first=%s retry=%s raw=%s",
                    parse_error,
                    retry_error,
                    _short_text(raw_content),
                )
                return _degraded_moderation(
                    provider_name=self.name,
                    provider=provider,
                    parse_error=parse_error,
                    retry_error=retry_error,
                    raw_content=raw_content,
                )

        categories = _dedupe_categories(_to_category(item) for item in _string_list(data.get("categories")))
        if not categories:
            categories = [SafetyCategory.SAFE] if bool(data.get("is_safe", True)) else [SafetyCategory.OTHER_NSFW]
        is_safe = bool(data.get("is_safe", SafetyCategory.SAFE in categories))
        if is_safe and any(category != SafetyCategory.SAFE for category in categories):
            is_safe = False
        scores = {str(k): float(v) for k, v in dict(data.get("scores") or {}).items() if _is_number(v)}
        if not is_safe:
            for category in categories:
                if category == SafetyCategory.SAFE:
                    continue
                scores.setdefault(category.value, 1.0)
        evidence_regions = _evidence_regions(data.get("evidence_regions"), image_payload)
        violations = _violations(data.get("violations"), data.get("evidence_regions"), image_payload)
        return PictureModerationResult(
            is_safe=is_safe,
            categories=categories,
            scores=scores,
            reason_codes=_string_list(data.get("reason_codes")) or [
                f"SAFETY_{category.value.upper()}" for category in categories if category != SafetyCategory.SAFE
            ],
            provider=self.name,
            metadata={
                "explanation": str(data.get("explanation") or ""),
                "evidence_regions": evidence_regions,
                "violations": violations,
                "category_details": _category_details(data.get("category_details")),
                "review_required": bool(data.get("review_required", False)),
                "qwen_global_geometry": _geometry_metadata(image_payload),
                "reused_text_compliance_qwen": True,
                "provider_mode": provider.mode,
                "provider_model": provider.model,
            },
        )

    def _request_content(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> str:
        response = httpx.post(url, headers=headers, json=payload, timeout=self._timeout_seconds)
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def verify_local_violation(
        self,
        crop_path: str,
        violation: dict[str, Any],
        *,
        max_side: int = 768,
    ) -> dict[str, Any]:
        provider = self._get_provider()
        url = self._chat_completions_url(provider.base_url)
        headers = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        entity_en = str(violation.get("entity_label_en") or violation.get("entity_label") or "")
        entity_zh = str(violation.get("entity_label_zh") or violation.get("label") or entity_en)
        category = str(violation.get("category") or "")
        redaction_target = str(violation.get("redaction_target") or "")
        risk_subtype = str(violation.get("risk_subtype") or "")
        image_payload = _image_data_payload(crop_path, max_side=max_side, jpeg_quality=self._image_jpeg_quality)
        qwen_w, qwen_h = image_payload["qwen_input_size"]
        payload = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": LOCAL_REVIEW_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请复核当前局部图片是否确实包含视觉内容安全违规实体。"
                                f"你当前看到的局部图片尺寸是 {qwen_w} x {qwen_h} 像素；如果返回 local_bbox，必须基于这个局部图片尺寸输出。"
                                f"类别：{category}；疑似实体英文名：{entity_en}；中文名：{entity_zh}；"
                                f"风险子类型：{risk_subtype}；目标脱敏区域：{redaction_target}。"
                                "如果目标是裸露上身/裸露躯干，请重点判断分割区域是否主要覆盖胸腹或背部躯干，"
                                "不要把手臂、头部、泳帽、背景人物或局部皮肤当作合格目标。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_payload["data_url"]
                            },
                        },
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(int(provider.max_tokens), max(self._max_tokens, 512)),
            "response_format": {"type": "json_object"},
        }
        try:
            data = _extract_json(self._request_content(url, headers, payload))
        except Exception as exc:
            logger.warning("Qwen3.5-VL local safety review failed: %s", exc)
            return {
                "is_authentic_violation": False,
                "boundary_status": "uncertain",
                "confidence": 0.0,
                "reason_zh": f"局部复核失败：{type(exc).__name__}",
                "review_error": str(exc),
            }
        bbox = data.get("local_bbox")
        raw_local_bbox = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 and all(_is_number(v) for v in bbox) else None
        local_bbox = _scale_bbox_from_qwen_input(raw_local_bbox, image_payload) if raw_local_bbox is not None else None
        return {
            "is_authentic_violation": bool(data.get("is_authentic_violation", False)),
            "entity_label_en": str(data.get("entity_label_en") or entity_en),
            "entity_label_zh": str(data.get("entity_label_zh") or entity_zh),
            "is_target_region": bool(data.get("is_target_region", data.get("is_authentic_violation", False))),
            "main_region_type": str(data.get("main_region_type") or ""),
            "wrong_region_type": str(data.get("wrong_region_type") or ""),
            "boundary_status": str(data.get("boundary_status") or "uncertain"),
            "local_bbox": local_bbox,
            "raw_local_bbox": raw_local_bbox,
            "qwen_local_geometry": _geometry_metadata(image_payload),
            "confidence": float(data.get("confidence") or 0.0) if _is_number(data.get("confidence")) else 0.0,
            "reason_zh": str(data.get("reason_zh") or ""),
        }

    @staticmethod
    def _chat_completions_url(base_url: str) -> str:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ProviderNotAvailableError("Qwen3.5 text-compliance endpoint")
        if normalized.endswith("/chat/completions"):
            return normalized
        return f"{normalized}/chat/completions"


def _image_data_url(image_path: str, max_side: int = 1280, jpeg_quality: int = 85) -> str:
    return str(_image_data_payload(image_path, max_side=max_side, jpeg_quality=jpeg_quality)["data_url"])


def _image_data_payload(image_path: str, max_side: int = 1280, jpeg_quality: int = 85) -> dict[str, Any]:
    path = Path(image_path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    if not path.exists():
        return {
            "data_url": "data:image/png;base64,",
            "original_size": [0, 0],
            "qwen_input_size": [0, 0],
            "max_side": max_side,
            "jpeg_quality": jpeg_quality,
        }
    data = path.read_bytes()
    original_size = [0, 0]
    qwen_input_size = [0, 0]
    try:
        from PIL import Image

        with Image.open(path) as image:
            original_size = [int(image.width), int(image.height)]
            image = image.convert("RGB")
            if max_side > 0:
                image.thumbnail((max_side, max_side))
            qwen_input_size = [int(image.width), int(image.height)]
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            data = buffer.getvalue()
            mime = "image/jpeg"
    except Exception:
        qwen_input_size = original_size
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "data_url": f"data:{mime};base64,{encoded}",
        "original_size": original_size,
        "qwen_input_size": qwen_input_size,
        "max_side": max_side,
        "jpeg_quality": jpeg_quality,
    }


def _extract_json(text: str) -> dict[str, Any]:
    candidate = _strip_json_candidate(text)
    for item in _json_candidates(candidate):
        try:
            payload = json.loads(item)
            break
        except json.JSONDecodeError:
            repaired = _repair_json_text(item)
            try:
                payload = json.loads(repaired)
                break
            except json.JSONDecodeError:
                continue
    else:
        payload = json.loads(_repair_json_text(candidate))
    if not isinstance(payload, dict):
        raise ValueError("Qwen3.5-VL response JSON must be an object")
    return payload


def _strip_json_candidate(text: str) -> str:
    candidate = str(text or "").strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    return candidate


def _json_candidates(candidate: str) -> list[str]:
    candidates = [candidate]
    start = candidate.find("{")
    end = candidate.rfind("}")
    if 0 <= start < end:
        sliced = candidate[start : end + 1]
        if sliced not in candidates:
            candidates.append(sliced)
    return candidates


def _repair_json_text(text: str) -> str:
    repaired = _strip_json_candidate(text)
    start = repaired.find("{")
    end = repaired.rfind("}")
    if 0 <= start < end:
        repaired = repaired[start : end + 1]
    repaired = repaired.replace("，", ",").replace("：", ":")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    # 模型偶尔漏掉对象字段之间的逗号："..."\n  "next_key": ...
    repaired = re.sub(r'("(?:[^"\\]|\\.)*")\s*\n\s*("[-A-Za-z0-9_\u4e00-\u9fff]+"\s*:)', r"\1,\n\2", repaired)
    # 数组或对象字段结束后接新字段时补逗号。
    repaired = re.sub(r'([}\]])\s*\n\s*("[-A-Za-z0-9_\u4e00-\u9fff]+"\s*:)', r"\1,\n\2", repaired)
    # 数值、布尔、null 后面接新字段时也补逗号。
    repaired = re.sub(r'(\b(?:true|false|null|-?\d+(?:\.\d+)?)\b)\s*\n\s*("[-A-Za-z0-9_\u4e00-\u9fff]+"\s*:)', r"\1,\n\2", repaired)
    repaired = _balance_json_brackets(repaired)
    return repaired


def _balance_json_brackets(text: str) -> str:
    in_string = False
    escaped = False
    stack: list[str] = []
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((stack[-1] == "{" and ch == "}") or (stack[-1] == "[" and ch == "]")):
                stack.pop()
    closing = {"{": "}", "[": "]"}
    return text + "".join(closing[ch] for ch in reversed(stack))


def _retry_payload(payload: dict[str, Any]) -> dict[str, Any]:
    retry = dict(payload)
    retry["max_tokens"] = max(int(payload.get("max_tokens") or 0), 768)
    retry["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n必须返回严格合法 JSON。所有字符串必须使用双引号，字段之间必须有逗号，不要输出 Markdown。"},
        payload["messages"][1],
    ]
    return retry


def _degraded_moderation(
    provider_name: str,
    provider: Any,
    parse_error: str,
    retry_error: str,
    raw_content: str,
) -> PictureModerationResult:
    return PictureModerationResult(
        is_safe=False,
        categories=[SafetyCategory.OTHER_NSFW],
        scores={SafetyCategory.OTHER_NSFW.value: 1.0},
        reason_codes=["VISUAL_SAFETY_MODEL_JSON_INVALID"],
        provider=provider_name,
        metadata={
            "explanation": "视觉内容安全模型返回格式异常，系统已降级为人工复核，不中断 OCR、视觉定位和脱敏流程。",
            "evidence_regions": [],
            "category_details": {
                SafetyCategory.OTHER_NSFW.value: {
                    "risk_subtype_zh": "模型返回异常",
                    "object_name_zh": "视觉安全审核结果待复核",
                    "scene_description_zh": "视觉安全模型返回内容不是合法 JSON，无法稳定解析具体画面风险。",
                    "risk_reason_zh": "为避免漏放风险，当前图片需要人工复核视觉内容安全结果。",
                }
            },
            "review_required": True,
            "degraded": True,
            "degrade_reason": "qwen_vl_json_parse_failed",
            "parse_error": parse_error,
            "retry_error": retry_error,
            "raw_response_excerpt": _short_text(raw_content),
            "reused_text_compliance_qwen": True,
            "provider_mode": getattr(provider, "mode", ""),
            "provider_model": getattr(provider, "model", ""),
        },
    )


def _short_text(value: str, limit: int = 1200) -> str:
    text = str(value or "").replace("\r", "\\r").replace("\n", "\\n")
    return text[:limit]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _to_category(value: str) -> SafetyCategory:
    normalized = value.strip().lower()
    aliases = {
        "safe": SafetyCategory.SAFE,
        "explicit": SafetyCategory.EXPLICIT,
        "adult": SafetyCategory.EXPLICIT,
        "adult_content": SafetyCategory.EXPLICIT,
        "porn": SafetyCategory.EXPLICIT,
        "porno": SafetyCategory.EXPLICIT,
        "pornography": SafetyCategory.EXPLICIT,
        "pornographic": SafetyCategory.EXPLICIT,
        "sexual_content": SafetyCategory.EXPLICIT,
        "sexually_explicit": SafetyCategory.EXPLICIT,
        "sexual": SafetyCategory.EXPLICIT,
        "nudity": SafetyCategory.EXPLICIT,
        "nude": SafetyCategory.EXPLICIT,
        "naked": SafetyCategory.EXPLICIT,
        "graphic_violence": SafetyCategory.GRAPHIC_VIOLENCE,
        "violence": SafetyCategory.GRAPHIC_VIOLENCE,
        "hate": SafetyCategory.HATE_SYMBOL,
        "hate_symbol": SafetyCategory.HATE_SYMBOL,
        "self_harm": SafetyCategory.SELF_HARM,
        "dangerous": SafetyCategory.DANGEROUS,
        "weapon": SafetyCategory.DANGEROUS,
        "drug": SafetyCategory.DANGEROUS,
        "other_nsfw": SafetyCategory.OTHER_NSFW,
    }
    return aliases.get(normalized, SafetyCategory.OTHER_NSFW)


def _dedupe_categories(categories: Any) -> list[SafetyCategory]:
    deduped: list[SafetyCategory] = []
    seen: set[SafetyCategory] = set()
    for category in categories:
        if category in seen:
            continue
        seen.add(category)
        deduped.append(category)
    if any(category != SafetyCategory.SAFE for category in deduped):
        deduped = [category for category in deduped if category != SafetyCategory.SAFE]
    return deduped


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _evidence_regions(value: Any, geometry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    regions: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(_is_number(v) for v in bbox):
            raw_bbox = [float(v) for v in bbox]
            regions.append(
                {
                    "category": str(item.get("category") or ""),
                    "label": str(item.get("label") or ""),
                    "bbox": _scale_bbox_from_qwen_input(raw_bbox, geometry) if geometry else raw_bbox,
                    "raw_bbox": raw_bbox if geometry else None,
                    "geometry_source_space": "qwen_global_input_space" if geometry else "unspecified",
                    "description": str(item.get("description") or ""),
                    "confidence": float(item.get("confidence") or 0.0),
                }
            )
        else:
            regions.append(
                {
                    "category": str(item.get("category") or ""),
                    "label": str(item.get("label") or ""),
                    "description": str(item.get("description") or ""),
                    "confidence": float(item.get("confidence") or 0.0) if _is_number(item.get("confidence")) else 0.0,
                }
            )
    return regions


def _violations(value: Any, evidence_regions: Any = None, geometry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    raw_items = value if isinstance(value, list) else []
    violations: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        category = _to_category(str(item.get("category") or "")).value
        center = _center_point(item.get("center_point") or item.get("point"))
        center_points = _center_points(item.get("center_points") or item.get("points"))
        if center and center not in center_points:
            center_points.insert(0, center)
        confidence = float(item.get("confidence") or 0.0) if _is_number(item.get("confidence")) else 0.0
        prompt_texts = _string_list(item.get("sam_prompt_texts"))
        prompt_text = str(item.get("sam_prompt_text") or item.get("entity_label_en") or item.get("entity_label") or item.get("label") or "")
        if prompt_text and prompt_text not in prompt_texts:
            prompt_texts.insert(0, prompt_text)
        normalized_prompts = [_normalize_entity_label_en(text) for text in prompt_texts if _normalize_entity_label_en(text)]
        raw_center = center or (center_points[0] if center_points else None)
        raw_bbox = _bbox(item.get("rough_bbox") or item.get("bbox"))
        violation = {
            "category": category,
            "entity_label_en": _normalize_entity_label_en(
                item.get("entity_label_en") or item.get("entity_label") or item.get("label") or item.get("object_name_en")
            ),
            "entity_label_zh": str(item.get("entity_label_zh") or item.get("object_name_zh") or item.get("label") or ""),
            "sam_prompt_text": normalized_prompts[0] if normalized_prompts else _normalize_entity_label_en(prompt_text),
            "sam_prompt_texts": normalized_prompts,
            "redaction_target": str(item.get("redaction_target") or item.get("target_region") or ""),
            "raw_center_point": raw_center,
            "raw_center_points": center_points,
            "raw_rough_bbox": raw_bbox,
            "center_point": _scale_point_from_qwen_input(raw_center, geometry) if geometry else raw_center,
            "center_points": [_scale_point_from_qwen_input(point, geometry) for point in center_points] if geometry else center_points,
            "rough_bbox": _scale_bbox_from_qwen_input(raw_bbox, geometry) if geometry else raw_bbox,
            "geometry_source_space": "qwen_global_input_space" if geometry else "unspecified",
            "qwen_global_geometry": _geometry_metadata(geometry) if geometry else {},
            "location_hint_zh": str(item.get("location_hint_zh") or item.get("location_hint") or ""),
            "visual_attributes_zh": str(item.get("visual_attributes_zh") or item.get("visual_attributes") or ""),
            "risk_reason_zh": str(item.get("risk_reason_zh") or item.get("reason_zh") or ""),
            "risk_subtype": str(item.get("risk_subtype") or ""),
            "decision_hint": str(item.get("decision_hint") or ""),
            "confidence": confidence,
        }
        _backfill_exposed_upper_body_policy(violation)
        if violation["entity_label_en"] or violation["entity_label_zh"]:
            violations.append(violation)
    if violations:
        return violations

    # Backward-compatible bridge for old Qwen responses.
    for item in _evidence_regions(evidence_regions, geometry):
        bbox = item.get("bbox")
        center = None
        if isinstance(bbox, list) and len(bbox) == 4:
            center = [float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0]
        label = str(item.get("label") or "")
        if not label:
            continue
        violation = {
            "category": _to_category(str(item.get("category") or "")).value,
            "entity_label_en": _normalize_entity_label_en(label),
            "entity_label_zh": label,
            "sam_prompt_text": _normalize_entity_label_en(label),
            "sam_prompt_texts": [_normalize_entity_label_en(label)] if _normalize_entity_label_en(label) else [],
            "redaction_target": str(item.get("redaction_target") or ""),
            "center_point": center,
            "center_points": [center] if center else [],
            "rough_bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
            "raw_center_point": item.get("raw_center_point"),
            "raw_rough_bbox": item.get("raw_bbox"),
            "geometry_source_space": item.get("geometry_source_space") or "unspecified",
            "qwen_global_geometry": _geometry_metadata(geometry) if geometry else {},
            "location_hint_zh": "",
            "visual_attributes_zh": str(item.get("description") or ""),
            "risk_reason_zh": str(item.get("description") or ""),
            "risk_subtype": str(item.get("risk_subtype") or ""),
            "decision_hint": str(item.get("decision_hint") or ""),
            "confidence": float(item.get("confidence") or 0.0),
            "source": "evidence_region_bridge",
        }
        _backfill_exposed_upper_body_policy(violation)
        violations.append(violation)
    return violations


def _center_point(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(_is_number(item) for item in value):
        return None
    return [float(value[0]), float(value[1])]


def _geometry_metadata(geometry: dict[str, Any] | None) -> dict[str, Any]:
    if not geometry:
        return {}
    original = list(geometry.get("original_size") or [0, 0])
    qwen_input = list(geometry.get("qwen_input_size") or [0, 0])
    sx = float(original[0]) / float(qwen_input[0]) if len(original) == 2 and len(qwen_input) == 2 and qwen_input[0] else 1.0
    sy = float(original[1]) / float(qwen_input[1]) if len(original) == 2 and len(qwen_input) == 2 and qwen_input[1] else 1.0
    return {
        "original_size": original,
        "qwen_input_size": qwen_input,
        "scale_to_original": {"x": sx, "y": sy},
        "max_side": geometry.get("max_side"),
    }


def _scale_point_from_qwen_input(value: Any, geometry: dict[str, Any] | None) -> list[float] | None:
    point = _center_point(value)
    if point is None:
        return None
    scale = (_geometry_metadata(geometry).get("scale_to_original") or {})
    return [float(point[0]) * float(scale.get("x", 1.0)), float(point[1]) * float(scale.get("y", 1.0))]


def _scale_bbox_from_qwen_input(value: Any, geometry: dict[str, Any] | None) -> list[float] | None:
    bbox = _bbox(value)
    if bbox is None:
        return None
    scale = (_geometry_metadata(geometry).get("scale_to_original") or {})
    sx = float(scale.get("x", 1.0))
    sy = float(scale.get("y", 1.0))
    x, y, w, h = bbox
    return [float(x) * sx, float(y) * sy, float(w) * sx, float(h) * sy]


def _center_points(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        point = _center_point(item)
        if point is not None:
            points.append(point)
    return points


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(_is_number(item) for item in value):
        return None
    return [float(item) for item in value]


def _normalize_entity_label_en(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "疑似手枪": "pistol",
        "手枪": "pistol",
        "枪械": "gun",
        "枪支": "gun",
        "疑似步枪": "rifle",
        "步枪": "rifle",
        "刀具": "knife",
        "刀": "knife",
        "管制刀具": "knife",
        "毒品": "drug package",
        "毒品包装": "drug package",
        "裸露上身": "bare torso",
        "裸露躯干": "bare torso",
        "裸露胸腹": "bare torso",
        "裸露背部": "bare torso",
        "上半身裸露": "bare torso",
        "bare upper body": "bare torso",
        "shirtless torso": "bare torso",
        "shirtless upper body": "bare torso",
    }
    return mapping.get(text, text)


def _backfill_exposed_upper_body_policy(violation: dict[str, Any]) -> None:
    if str(violation.get("category") or "").strip().lower() != SafetyCategory.OTHER_NSFW.value:
        return
    values = [
        str(violation.get("risk_subtype") or ""),
        str(violation.get("entity_label_en") or ""),
        str(violation.get("entity_label_zh") or ""),
        str(violation.get("sam_prompt_text") or ""),
        str(violation.get("risk_reason_zh") or ""),
    ]
    values.extend(str(value or "") for value in violation.get("sam_prompt_texts") or [])
    if not any(_is_exposed_upper_body_text(value) for value in values):
        return
    violation["risk_subtype"] = "exposed_upper_body"
    violation["decision_hint"] = "redact_only"
    if not str(violation.get("entity_label_en") or ""):
        violation["entity_label_en"] = "bare torso"
    if not str(violation.get("entity_label_zh") or ""):
        violation["entity_label_zh"] = "裸露上身"
    if not str(violation.get("redaction_target") or ""):
        violation["redaction_target"] = "torso_without_head"
    prompts = [str(value or "").strip().lower() for value in violation.get("sam_prompt_texts") or [] if str(value or "").strip()]
    for prompt in ("bare torso", "nude torso", "chest and abdomen", "torso without head", "skin area", "human body", "upper body"):
        if prompt not in prompts:
            prompts.append(prompt)
    violation["sam_prompt_texts"] = prompts
    if not str(violation.get("sam_prompt_text") or ""):
        violation["sam_prompt_text"] = prompts[0]


def _is_exposed_upper_body_text(value: str) -> bool:
    text = str(value or "").strip().lower()
    if text in {"exposed_upper_body", "裸露上身", "裸露躯干", "裸露胸腹", "裸露背部", "上半身裸露"}:
        return True
    return any(
        term in text
        for term in (
            "bare torso",
            "nude torso",
            "upper body",
            "shirtless torso",
            "shirtless upper body",
            "exposed torso",
            "bare upper body",
            "裸露上身",
            "裸露躯干",
            "裸露胸腹",
            "裸露背部",
            "上半身裸露",
            "裸露身体区域",
        )
    )


def _category_details(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        return {}
    details: dict[str, dict[str, str]] = {}
    for raw_key, raw_detail in value.items():
        category = _to_category(str(raw_key)).value
        if not isinstance(raw_detail, dict):
            continue
        details[category] = {
            "risk_subtype_zh": str(raw_detail.get("risk_subtype_zh") or raw_detail.get("risk_subtype") or ""),
            "object_name_zh": str(raw_detail.get("object_name_zh") or raw_detail.get("object_name") or ""),
            "scene_description_zh": str(raw_detail.get("scene_description_zh") or raw_detail.get("scene_description") or ""),
            "risk_reason_zh": str(raw_detail.get("risk_reason_zh") or raw_detail.get("risk_reason") or ""),
        }
    return details

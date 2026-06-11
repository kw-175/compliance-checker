"""
Shared GPT-5.2 picture analysis client.

This module centralizes the single external API call used by the
OpenAI-backed OCR / PII / safety / vision providers so they can share
one structured response per image.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from picture.domain.enums import FindingType, PIIEntityType, SafetyCategory, VisionObjectType
from picture.domain.exceptions import ProviderError, ProviderNotAvailableError
from picture.domain.models import (
    BBox,
    OCRLayoutResult,
    OCRTextBlock,
    PictureFinding,
    PictureModerationResult,
    RegionMask,
)
from picture.infra.config import PictureSettings

logger = logging.getLogger(__name__)

_SAFETY_CATEGORY_MAP: dict[str, SafetyCategory] = {
    "safe": SafetyCategory.SAFE,
    "explicit": SafetyCategory.EXPLICIT,
    "graphic_violence": SafetyCategory.GRAPHIC_VIOLENCE,
    "hate_symbol": SafetyCategory.HATE_SYMBOL,
    "self_harm": SafetyCategory.SELF_HARM,
    "dangerous": SafetyCategory.DANGEROUS,
    "other_nsfw": SafetyCategory.OTHER_NSFW,
}

_PII_CATEGORY_MAP: dict[str, PIIEntityType] = {
    "person_name": PIIEntityType.PERSON_NAME,
    "phone_number": PIIEntityType.PHONE_NUMBER,
    "email": PIIEntityType.EMAIL,
    "id_card": PIIEntityType.ID_CARD,
    "bank_card": PIIEntityType.BANK_CARD,
    "address": PIIEntityType.ADDRESS,
    "license_plate": PIIEntityType.LICENSE_PLATE,
    "other": PIIEntityType.OTHER,
}

_VISION_CATEGORY_MAP: dict[str, VisionObjectType] = {
    "face": VisionObjectType.FACE,
    "id_card": VisionObjectType.ID_CARD,
    "badge": VisionObjectType.BADGE,
    "signature": VisionObjectType.SIGNATURE,
    "stamp": VisionObjectType.STAMP,
    "qr_code": VisionObjectType.QR_CODE,
    "barcode": VisionObjectType.BARCODE,
    "license_plate": VisionObjectType.LICENSE_PLATE,
}


@dataclass(slots=True)
class OpenAIPictureAnalysis:
    route_suggestion: str
    image_summary: str
    safety: dict[str, Any]
    text_blocks: list[dict[str, Any]]
    pii_findings: list[dict[str, Any]]
    visual_findings: list[dict[str, Any]]
    recommended_decision: str
    decision_reason_codes: list[str]
    notes: list[str]
    raw_response: dict[str, Any]


class OpenAIPictureAnalyzer:
    """Shared single-request image analyzer backed by the OpenAI Responses API."""

    def __init__(self, settings: PictureSettings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key.strip()
        self._base_url = settings.openai_base_url.rstrip("/")
        self._model = settings.openai_model.strip() or "gpt-5.2"
        self._timeout = settings.openai_timeout_seconds
        self._image_detail = settings.openai_image_detail
        self._cache: dict[str, OpenAIPictureAnalysis] = {}
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model

    def analyze(self, image_path: str) -> OpenAIPictureAnalysis:
        normalized_path = str(Path(image_path).resolve())
        with self._lock:
            cached = self._cache.get(normalized_path)
            if cached is not None:
                return cached
            analysis = self._request_and_parse(normalized_path)
            self._cache[normalized_path] = analysis
        return analysis

    def build_ocr_result(self, image_path: str) -> OCRLayoutResult:
        analysis = self.analyze(image_path)
        width, height = _image_size(image_path)
        text_blocks: list[OCRTextBlock] = []
        for block in analysis.text_blocks:
            bbox = _bbox_from_norm(block.get("bbox_norm"), width, height)
            if bbox is None:
                continue
            text_blocks.append(
                OCRTextBlock(
                    text=_text(block, "text"),
                    bbox=bbox,
                    confidence=_score(block.get("confidence"), default=0.85),
                    language=_text(block, "language", "unknown"),
                )
            )

        pii_findings = self.build_pii_findings(image_path)
        return OCRLayoutResult(
            full_text="\n".join(block.text for block in text_blocks).strip(),
            text_blocks=text_blocks,
            layout_regions=[],
            engine_name="openai_gpt52",
            metadata={
                "route_suggestion": analysis.route_suggestion,
                "image_summary": analysis.image_summary,
                "recommended_decision": analysis.recommended_decision,
                "decision_reason_codes": analysis.decision_reason_codes,
                "notes": analysis.notes,
                "precomputed_pii_findings": pii_findings,
                "openai_model": self._model,
            },
        )

    def build_pii_findings(self, image_path: str) -> list[PictureFinding]:
        analysis = self.analyze(image_path)
        width, height = _image_size(image_path)
        findings: list[PictureFinding] = []
        for item in analysis.pii_findings:
            category = _PII_CATEGORY_MAP.get(_text(item, "category", "other"), PIIEntityType.OTHER)
            region = _region_from_item(item, width, height)
            findings.append(
                PictureFinding(
                    finding_type=FindingType.TEXT_PII,
                    category=category.value,
                    label=_text(item, "label", f"PII: {category.value}"),
                    score=_score(item.get("score")),
                    region=region,
                    text_span=_text(item, "text_span"),
                    reason_code=_text(item, "reason_code", f"PII_{category.value.upper()}"),
                    provider="OpenAIGPT52PII",
                    provider_version=self._model,
                    threshold_used=0.0,
                    explanation=_text(item, "reason"),
                    metadata={
                        "block_ids": _string_list(item.get("block_ids")),
                        "source": "openai_gpt52_shared_analysis",
                    },
                )
            )
        return findings

    def build_safety_result(self, image_path: str) -> PictureModerationResult:
        analysis = self.analyze(image_path)
        safety = analysis.safety or {}
        categories_payload = safety.get("categories", [])
        categories: list[SafetyCategory] = []
        scores: dict[str, float] = {}
        reason_codes: list[str] = []

        for item in categories_payload:
            name = _text(item, "name", "safe")
            score = _score(item.get("score"), default=0.0)
            scores[name] = score
            mapped = _SAFETY_CATEGORY_MAP.get(name)
            if mapped is not None and mapped != SafetyCategory.SAFE and score > 0:
                categories.append(mapped)
                default_reason = f"SAFETY_{mapped.value.upper()}"
                reason_codes.append(_text(item, "reason_code", default_reason))

        declared_safe = bool(safety.get("is_safe", True))
        if not categories:
            if declared_safe:
                categories = [SafetyCategory.SAFE]
                scores.setdefault("safe", 1.0)
            else:
                categories = [SafetyCategory.OTHER_NSFW]
                scores.setdefault("other_nsfw", 0.5)
                reason_codes.append("SAFETY_OTHER_NSFW")

        unique_reason_codes: list[str] = []
        for code in reason_codes:
            if code and code not in unique_reason_codes:
                unique_reason_codes.append(code)

        return PictureModerationResult(
            is_safe=bool(
                declared_safe
                and (not categories or categories == [SafetyCategory.SAFE])
            ),
            categories=categories,
            scores=scores,
            reason_codes=unique_reason_codes,
            provider="OpenAIGPT52Safety",
            metadata={
                "route_suggestion": analysis.route_suggestion,
                "recommended_decision": analysis.recommended_decision,
                "notes": analysis.notes,
                "image_summary": analysis.image_summary,
            },
        )

    def build_vision_findings(self, image_path: str) -> list[PictureFinding]:
        analysis = self.analyze(image_path)
        width, height = _image_size(image_path)
        findings: list[PictureFinding] = []
        for item in analysis.visual_findings:
            category = _VISION_CATEGORY_MAP.get(_text(item, "category", ""), None)
            if category is None:
                continue
            findings.append(
                PictureFinding(
                    finding_type=FindingType.VISION_OBJECT,
                    category=category.value,
                    label=_text(item, "label", category.value),
                    score=_score(item.get("score")),
                    region=_region_from_item(item, width, height),
                    reason_code=_text(item, "reason_code", f"VISION_{category.value.upper()}"),
                    provider="OpenAIGPT52Vision",
                    provider_version=self._model,
                    threshold_used=0.0,
                    explanation=_text(item, "reason"),
                    metadata={"source": "openai_gpt52_shared_analysis"},
                )
            )
        return findings

    def _request_and_parse(self, image_path: str) -> OpenAIPictureAnalysis:
        if not self._api_key:
            raise ProviderNotAvailableError("OpenAI API key (set PICTURE_OPENAI_API_KEY)")

        payload = self._build_payload(image_path)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/responses"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise ProviderError("OpenAIGPT52", f"HTTP {exc.response.status_code}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("OpenAIGPT52", f"request failed: {exc}") from exc

        body = response.json()
        content = _extract_output_json_text(body)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError("OpenAIGPT52", f"invalid JSON response: {exc}") from exc

        logger.info(
            "[OpenAIGPT52] analyzed %s with %d text blocks, %d pii findings, %d visual findings",
            image_path,
            len(parsed.get("text_blocks", [])),
            len(parsed.get("pii_findings", [])),
            len(parsed.get("visual_findings", [])),
        )
        return OpenAIPictureAnalysis(
            route_suggestion=_text(parsed, "route_suggestion", "mixed"),
            image_summary=_text(parsed, "image_summary"),
            safety=parsed.get("safety", {}) if isinstance(parsed.get("safety"), dict) else {},
            text_blocks=parsed.get("text_blocks", []) if isinstance(parsed.get("text_blocks"), list) else [],
            pii_findings=parsed.get("pii_findings", []) if isinstance(parsed.get("pii_findings"), list) else [],
            visual_findings=parsed.get("visual_findings", []) if isinstance(parsed.get("visual_findings"), list) else [],
            recommended_decision=_text(parsed, "recommended_decision", "review"),
            decision_reason_codes=_string_list(parsed.get("decision_reason_codes")),
            notes=_string_list(parsed.get("notes")),
            raw_response=body,
        )

    def _build_payload(self, image_path: str) -> dict[str, Any]:
        _, data_url = _image_data_url(image_path)
        prompt = (
            "You are a picture compliance analysis engine. Analyze the input image and return only the"
            " requested JSON schema. Detect visible text blocks, text PII, sensitive visual objects, and"
            " high-level image safety issues. Use only the allowed category values. Bounding boxes must be"
            " normalized to a 0-1000 scale with x1 <= x2 and y1 <= y2. If no item exists, return an empty"
            " list. Keep scores between 0 and 1. Use concise reason codes such as PII_PHONE, VISION_FACE,"
            " SAFETY_EXPLICIT."
        )
        return {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": data_url,
                            "detail": self._image_detail,
                        },
                    ],
                }
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "picture_compliance_analysis",
                    "strict": True,
                    "schema": _analysis_schema(),
                }
            },
        }


def _analysis_schema() -> dict[str, Any]:
    bbox_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x1": {"type": "number"},
            "y1": {"type": "number"},
            "x2": {"type": "number"},
            "y2": {"type": "number"},
        },
        "required": ["x1", "y1", "x2", "y2"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "route_suggestion": {
                "type": "string",
                "enum": ["document", "natural", "mixed"],
            },
            "image_summary": {"type": "string"},
            "safety": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "is_safe": {"type": "boolean"},
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [
                                        "safe",
                                        "explicit",
                                        "graphic_violence",
                                        "hate_symbol",
                                        "self_harm",
                                        "dangerous",
                                        "other_nsfw",
                                    ],
                                },
                                "score": {"type": "number"},
                                "reason": {"type": "string"},
                                "reason_code": {"type": "string"},
                            },
                            "required": ["name", "score", "reason", "reason_code"],
                        },
                    },
                },
                "required": ["is_safe", "categories"],
            },
            "text_blocks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "language": {"type": "string"},
                        "bbox_norm": bbox_schema,
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "text", "language", "bbox_norm", "confidence"],
                },
            },
            "pii_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "person_name",
                                "phone_number",
                                "email",
                                "id_card",
                                "bank_card",
                                "address",
                                "license_plate",
                                "other",
                            ],
                        },
                        "label": {"type": "string"},
                        "text_span": {"type": "string"},
                        "block_ids": {"type": "array", "items": {"type": "string"}},
                        "bbox_norm": bbox_schema,
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                        "reason_code": {"type": "string"},
                    },
                    "required": [
                        "category",
                        "label",
                        "text_span",
                        "block_ids",
                        "bbox_norm",
                        "score",
                        "reason",
                        "reason_code",
                    ],
                },
            },
            "visual_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "face",
                                "id_card",
                                "badge",
                                "signature",
                                "stamp",
                                "qr_code",
                                "barcode",
                                "license_plate",
                            ],
                        },
                        "label": {"type": "string"},
                        "bbox_norm": bbox_schema,
                        "score": {"type": "number"},
                        "reason": {"type": "string"},
                        "reason_code": {"type": "string"},
                    },
                    "required": [
                        "category",
                        "label",
                        "bbox_norm",
                        "score",
                        "reason",
                        "reason_code",
                    ],
                },
            },
            "recommended_decision": {
                "type": "string",
                "enum": ["pass_raw", "pass_redacted", "drop", "review"],
            },
            "decision_reason_codes": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "route_suggestion",
            "image_summary",
            "safety",
            "text_blocks",
            "pii_findings",
            "visual_findings",
            "recommended_decision",
            "decision_reason_codes",
            "notes",
        ],
    }


def _image_data_url(image_path: str) -> tuple[str, str]:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return mime_type, f"data:{mime_type};base64,{encoded}"


def _image_size(image_path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size
    except Exception:
        pass

    try:
        return _image_size_without_pil(image_path)
    except Exception as exc:
        raise ProviderError("OpenAIGPT52", f"failed to read image size: {exc}") from exc


def _image_size_without_pil(image_path: str) -> tuple[int, int]:
    path = Path(image_path)
    with path.open("rb") as handle:
        header = handle.read(32)

    if header[:8] == b"\x89PNG\r\n\x1a\n":
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("invalid PNG header")
        width, height = struct.unpack(">II", header[16:24])
        return int(width), int(height)

    if header[:6] in {b"GIF87a", b"GIF89a"}:
        width, height = struct.unpack("<HH", header[6:10])
        return int(width), int(height)

    if header[:2] == b"BM":
        width, height = struct.unpack("<II", header[18:26])
        return int(width), int(height)

    if header[:2] == b"\xff\xd8":
        with path.open("rb") as handle:
            handle.seek(0)
            size = 2
            marker = handle.read(2)
            while marker and marker[0] == 0xFF:
                while marker[1] == 0xFF:
                    marker = bytes([marker[0], handle.read(1)[0]])
                if 0xC0 <= marker[1] <= 0xC3:
                    handle.read(3)
                    height, width = struct.unpack(">HH", handle.read(4))
                    return int(width), int(height)
                size_bytes = handle.read(2)
                if len(size_bytes) != 2:
                    break
                size = struct.unpack(">H", size_bytes)[0] - 2
                handle.seek(size, 1)
                marker = handle.read(2)
        raise ValueError("unsupported JPEG structure")

    raise ValueError("unsupported image format")


def _extract_output_json_text(body: dict[str, Any]) -> str:
    if isinstance(body.get("output_text"), str) and body["output_text"].strip():
        return body["output_text"]

    for output_item in body.get("output", []):
        for content_item in output_item.get("content", []):
            if content_item.get("type") in {"output_text", "text"} and isinstance(content_item.get("text"), str):
                return content_item["text"]
    raise ProviderError("OpenAIGPT52", "response did not contain structured text output")


def _region_from_item(item: dict[str, Any], width: int, height: int) -> RegionMask | None:
    bbox = _bbox_from_norm(item.get("bbox_norm"), width, height)
    if bbox is None:
        return None
    return RegionMask(bbox=bbox, confidence=_score(item.get("score")))


def _bbox_from_norm(raw_bbox: Any, width: int, height: int) -> BBox | None:
    if not isinstance(raw_bbox, dict):
        return None
    x1 = _clamp_norm(raw_bbox.get("x1"))
    y1 = _clamp_norm(raw_bbox.get("y1"))
    x2 = _clamp_norm(raw_bbox.get("x2"))
    y2 = _clamp_norm(raw_bbox.get("y2"))
    if None in {x1, y1, x2, y2}:
        return None
    left = min(x1, x2) / 1000.0 * width
    top = min(y1, y2) / 1000.0 * height
    right = max(x1, x2) / 1000.0 * width
    bottom = max(y1, y2) / 1000.0 * height
    bbox_width = max(1.0, right - left)
    bbox_height = max(1.0, bottom - top)
    return BBox(x=left, y=top, w=bbox_width, h=bbox_height)


def _clamp_norm(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(number, 1000.0))


def _score(value: Any, default: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(number, 1.0))


def _text(data: dict[str, Any], key: str, default: str = "") -> str:
    value = data.get(key, default)
    return value if isinstance(value, str) else default


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str) and item:
            result.append(item)
    return result

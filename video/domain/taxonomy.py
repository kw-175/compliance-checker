"""Video risk taxonomy mapped from the existing picture/audio/text engines."""

from __future__ import annotations

import json
from typing import Any

from picture.domain.enums import SafetyCategory
from picture.domain.models import PictureFinding, PictureModerationResult

PRIVACY_CATEGORIES = {
    "privacy.face",
    "privacy.id_card",
    "privacy.badge",
    "privacy.signature",
    "privacy.stamp",
    "privacy.qr_code",
    "privacy.barcode",
    "privacy.license_plate",
    "privacy.account_region",
    "privacy.school_class_identifier",
    "privacy.phone",
    "privacy.address",
    "privacy.screen_sensitive",
}

CONTENT_CATEGORIES = {
    "content.sexual",
    "content.violence",
    "content.graphic_violence",
    "content.self_harm",
    "content.hate",
    "content.harassment",
    "content.illegal_instruction",
    "content.other_nsfw",
}

_PICTURE_CATEGORY_MAP = {
    "face": "privacy.face",
    "id_card": "privacy.id_card",
    "badge": "privacy.badge",
    "signature": "privacy.signature",
    "stamp": "privacy.stamp",
    "qr_code": "privacy.qr_code",
    "barcode": "privacy.barcode",
    "license_plate": "privacy.license_plate",
    "avatar": "privacy.face",
    "account_region": "privacy.account_region",
    "school_class_identifier": "privacy.school_class_identifier",
    "phone": "privacy.phone",
    "phone_number": "privacy.phone",
    "address": "privacy.address",
    "screen_sensitive": "privacy.screen_sensitive",
}

_SAFETY_CATEGORY_MAP = {
    SafetyCategory.EXPLICIT.value: "content.sexual",
    SafetyCategory.OTHER_NSFW.value: "content.other_nsfw",
    SafetyCategory.GRAPHIC_VIOLENCE.value: "content.graphic_violence",
    SafetyCategory.HATE_SYMBOL.value: "content.hate",
    SafetyCategory.SELF_HARM.value: "content.self_harm",
    SafetyCategory.DANGEROUS.value: "content.illegal_instruction",
}


def map_picture_finding_category(finding: PictureFinding) -> str:
    """Map a picture finding into the stable video risk taxonomy."""
    category = str(finding.category or "").strip().lower()
    if category in _PICTURE_CATEGORY_MAP:
        return _PICTURE_CATEGORY_MAP[category]
    reason = str(finding.reason_code or "").strip().lower()
    if "sexual" in reason or "porn" in reason:
        return "content.sexual"
    if "violence" in reason or "violent" in reason:
        return "content.violence"
    if "hate" in reason:
        return "content.hate"
    if "self_harm" in reason or "suicide" in reason:
        return "content.self_harm"
    if "phone" in reason:
        return "privacy.phone"
    if "address" in reason:
        return "privacy.address"
    return f"visual.{category or 'unknown'}"


def map_moderation_category(moderation: PictureModerationResult, reason_code: str = "") -> str:
    """Map picture safety moderation categories into video taxonomy."""
    metadata_text = json.dumps(moderation.metadata or {}, ensure_ascii=False).lower()
    if any(token in metadata_text for token in ("violence", "violent", "fight", "assault", "physical_conflict", "肢体冲突", "斗殴", "打架", "暴力")):
        return "content.graphic_violence"
    for category in moderation.categories:
        mapped = _SAFETY_CATEGORY_MAP.get(category.value)
        if mapped:
            return mapped
    reason = str(reason_code or " ".join(moderation.reason_codes)).lower()
    if "explicit" in reason or "sexual" in reason:
        return "content.sexual"
    if "violence" in reason:
        return "content.graphic_violence"
    if "hate" in reason:
        return "content.hate"
    if "self" in reason:
        return "content.self_harm"
    return "content.other_nsfw"


def severity_for_category(category: str, confidence: float = 0.0, metadata: dict[str, Any] | None = None) -> str:
    """Return a conservative severity for a normalized category."""
    metadata = metadata or {}
    if category in {"content.sexual_minor", "content.graphic_violence"}:
        return "critical"
    if category.startswith("content."):
        return "high" if confidence >= 0.7 else "medium"
    if category in {"privacy.id_card", "privacy.phone", "privacy.address"}:
        return "high" if confidence >= 0.7 else "medium"
    if category in {"privacy.face", "privacy.license_plate", "privacy.qr_code"}:
        return "medium" if confidence >= 0.5 else "low"
    if bool(metadata.get("review_required", False)):
        return "medium"
    return "low"


def recommended_actions_for_category(category: str) -> list[str]:
    if category.startswith("content."):
        return ["review", "restrict_or_reject"]
    if category.startswith("privacy."):
        return ["label", "restrict", "redact_if_public_release"]
    return ["label", "review"]

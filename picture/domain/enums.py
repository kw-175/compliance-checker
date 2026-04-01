"""
Core enumerations for the picture compliance engine.
"""
# 中文说明：该文件集中定义 picture 模块里会反复出现的枚举常量。
# 把这些值收敛到一起，可以避免魔法字符串散落在各层代码中。
from __future__ import annotations

from enum import Enum


class RouteType(str, Enum):
    """Image routing classification."""

    # 中文说明：文档类图片，通常文本密度高，优先走 OCR/PII 链路。
    DOCUMENT = "document"

    # 中文说明：自然图像，通常是实拍照片或监控画面，优先走安全审核和视觉检测。
    NATURAL = "natural"

    # 中文说明：混合截图，同时包含文字和视觉对象，适合 OCR 与视觉并行处理。
    MIXED = "mixed"


class JobStatus(str, Enum):
    """Lifecycle states of a picture compliance job."""

    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    ROUTED = "ROUTED"
    DETECTING = "DETECTING"
    SEGMENTING = "SEGMENTING"
    REDACTING = "REDACTING"
    POLICY_EVALUATING = "POLICY_EVALUATING"
    DONE = "DONE"
    DROPPED = "DROPPED"
    FAILED = "FAILED"


class DecisionType(str, Enum):
    """Final compliance decision for an image."""

    # 中文说明：原图可直接通过，不需要脱敏。
    PASS_RAW = "pass_raw"

    # 中文说明：原图存在敏感信息，需要脱敏后才能交付。
    PASS_REDACTED = "pass_redacted"

    # 中文说明：图片内容不允许输出。
    DROP = "drop"


class FindingType(str, Enum):
    """Category of a detected finding."""

    TEXT_PII = "text_pii"
    SAFETY = "safety"
    VISION_OBJECT = "vision_object"


class RedactionMode(str, Enum):
    """Visual redaction rendering strategy."""

    BLACK_BOX = "black_box"
    GAUSSIAN_BLUR = "gaussian_blur"
    PIXELATE = "pixelate"
    SOLID_FILL = "solid_fill"


class SafetyCategory(str, Enum):
    """Safety moderation categories."""

    SAFE = "safe"
    EXPLICIT = "explicit"
    GRAPHIC_VIOLENCE = "graphic_violence"
    HATE_SYMBOL = "hate_symbol"
    SELF_HARM = "self_harm"
    DANGEROUS = "dangerous"
    OTHER_NSFW = "other_nsfw"


class PIIEntityType(str, Enum):
    """PII entity types detected in text."""

    PERSON_NAME = "person_name"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    ID_CARD = "id_card"
    BANK_CARD = "bank_card"
    ADDRESS = "address"
    LICENSE_PLATE = "license_plate"
    OTHER = "other"


class VisionObjectType(str, Enum):
    """Vision-detected object types."""

    FACE = "face"
    ID_CARD = "id_card"
    BADGE = "badge"
    SIGNATURE = "signature"
    STAMP = "stamp"
    QR_CODE = "qr_code"
    BARCODE = "barcode"
    LICENSE_PLATE = "license_plate"


class FailurePolicy(str, Enum):
    """How to handle provider failures."""

    # 中文说明：失败时尽量放行，适合低阻断场景，但合规风险更高。
    FAIL_OPEN = "fail_open"

    # 中文说明：失败时趋向保守处理，适合合规优先场景。
    FAIL_CLOSED = "fail_closed"

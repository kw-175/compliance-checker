"""
Core enumerations for the picture compliance engine.
"""

from __future__ import annotations

from enum import Enum


class RouteType(str, Enum):
    """Image routing classification."""
    DOCUMENT = "document"
    NATURAL = "natural"
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
    PASS_RAW = "pass_raw"
    PASS_REDACTED = "pass_redacted"
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
    FAIL_OPEN = "fail_open"
    FAIL_CLOSED = "fail_closed"

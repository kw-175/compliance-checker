"""Video compliance operators inherited from text, audio, and picture modules."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VideoComplianceOperator(BaseModel):
    """Catalog entry for a video-level operator inherited from another modality."""

    operator_id: str
    source_operator_id: str
    source_modality: str
    target_type: str
    target_labels: list[str] = Field(default_factory=list)
    display_name: str = ""
    description: str = ""


VPI_OPERATORS: dict[str, VideoComplianceOperator] = {
    "VVIS_VPI_001": VideoComplianceOperator(operator_id="VVIS_VPI_001", source_operator_id="VPI_001", source_modality="picture", target_type="face", display_name="视频帧人脸"),
    "VVIS_VPI_002": VideoComplianceOperator(operator_id="VVIS_VPI_002", source_operator_id="VPI_002", source_modality="picture", target_type="id_card", display_name="视频帧证件"),
    "VVIS_VPI_003": VideoComplianceOperator(operator_id="VVIS_VPI_003", source_operator_id="VPI_003", source_modality="picture", target_type="badge", display_name="视频帧胸牌/工牌"),
    "VVIS_VPI_004": VideoComplianceOperator(operator_id="VVIS_VPI_004", source_operator_id="VPI_004", source_modality="picture", target_type="signature", display_name="视频帧签名"),
    "VVIS_VPI_005": VideoComplianceOperator(operator_id="VVIS_VPI_005", source_operator_id="VPI_005", source_modality="picture", target_type="stamp", display_name="视频帧印章"),
    "VVIS_VPI_006": VideoComplianceOperator(operator_id="VVIS_VPI_006", source_operator_id="VPI_006", source_modality="picture", target_type="qr_code", display_name="视频帧二维码"),
    "VVIS_VPI_007": VideoComplianceOperator(operator_id="VVIS_VPI_007", source_operator_id="VPI_007", source_modality="picture", target_type="barcode", display_name="视频帧条形码"),
    "VVIS_VPI_008": VideoComplianceOperator(operator_id="VVIS_VPI_008", source_operator_id="VPI_008", source_modality="picture", target_type="license_plate", display_name="视频帧车牌"),
    "VVIS_VPI_009": VideoComplianceOperator(operator_id="VVIS_VPI_009", source_operator_id="VPI_009", source_modality="picture", target_type="avatar", display_name="视频帧头像"),
    "VVIS_VPI_010": VideoComplianceOperator(operator_id="VVIS_VPI_010", source_operator_id="VPI_010", source_modality="picture", target_type="account_region", display_name="视频帧账号区域"),
    "VVIS_VPI_011": VideoComplianceOperator(operator_id="VVIS_VPI_011", source_operator_id="VPI_011", source_modality="picture", target_type="school_class_identifier", display_name="视频帧学校/班级标识"),
}

PII_TARGETS: dict[str, set[str]] = {
    "PII_001": {"person_name"},
    "PII_002": {"phone", "phone_number", "email", "social_account"},
    "PII_003": {"id_card", "id_number", "passport"},
    "PII_004": {"address", "location"},
    "PII_005": {"student_id", "education_record", "score_record", "disciplinary_record"},
    "PII_006": {"parent_contact", "guardian_contact", "family_contact"},
    "PII_007": {"bank_card", "bank_account", "payment_account"},
    "PII_008": {"medical_record", "psychological_record", "health_record"},
    "PII_009": {"secret", "api_key", "token", "password", "credential"},
    "PII_010": {"combined_identity"},
    "PII_011": {"minor_info", "student_id", "education_record", "parent_contact"},
}

CSA_LABELS: dict[str, str] = {
    "CSA_001": "content.political",
    "CSA_002": "content.pornographic",
    "CSA_003": "content.violent",
    "CSA_004": "content.hate",
    "CSA_005": "content.harassment",
    "CSA_006": "content.self_harm",
    "CSA_007": "content.illegal_instruction",
    "CSA_008": "content.minor_harmful",
    "CSA_009": "content.misleading",
    "CSA_010": "content.values_violation",
    "CSA_011": "content.jailbreak",
}

VIDEO_OPERATOR_CATALOG: dict[str, VideoComplianceOperator] = dict(VPI_OPERATORS)
for source_id, targets in PII_TARGETS.items():
    suffix = source_id.split("_", 1)[1]
    for prefix, modality, label in (
        ("VTXT", "ocr_text", "视频 OCR/字幕文本隐私"),
        ("VAUD", "audio", "视频音轨文本隐私"),
    ):
        operator_id = f"{prefix}_{source_id}"
        VIDEO_OPERATOR_CATALOG[operator_id] = VideoComplianceOperator(
            operator_id=operator_id,
            source_operator_id=source_id,
            source_modality=modality,
            target_type=sorted(targets)[0],
            target_labels=sorted(targets),
            display_name=f"{label} {suffix}",
        )

for source_id, label in CSA_LABELS.items():
    suffix = source_id.split("_", 1)[1]
    for prefix, modality, name in (
        ("VTXT", "ocr_text", "视频 OCR/字幕内容安全"),
        ("VAUD", "audio", "视频音轨内容安全"),
        ("VVIS", "visual_safety", "视频画面视觉安全"),
    ):
        operator_id = f"{prefix}_{source_id}"
        VIDEO_OPERATOR_CATALOG[operator_id] = VideoComplianceOperator(
            operator_id=operator_id,
            source_operator_id=source_id,
            source_modality=modality,
            target_type=label,
            target_labels=[label],
            display_name=f"{name} {suffix}",
        )


def catalog_snapshot() -> list[dict[str, Any]]:
    """Return a stable JSON-ready operator catalog snapshot."""
    return [
        item.model_dump(mode="json")
        for _, item in sorted(VIDEO_OPERATOR_CATALOG.items())
    ]

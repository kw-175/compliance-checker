"""Resolve audio compliance operator selection before detector execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PII_TARGETS: dict[str, set[str]] = {
    "PII_001": {"person_name"},
    "PII_002": {"phone", "phone_number", "email", "social_account"},
    "PII_003": {"id_card", "id_number", "passport", "driver_license"},
    "PII_004": {"address", "location"},
    "PII_005": {"student_id", "education_record", "score_record", "disciplinary_record"},
    "PII_006": {"parent_contact", "guardian_contact", "family_contact"},
    "PII_007": {"bank_card", "bank_account", "payment_account"},
    "PII_008": {"medical_record", "psychological_record", "health_record"},
    "PII_009": {"secret", "api_key", "token", "password", "credential", "ip_address", "url"},
    "PII_010": {"combined_identity"},
    "PII_011": {"minor_info", "student_id", "education_record", "parent_contact"},
}

PII_ENTITY_TYPES: dict[str, set[str]] = {
    "PII_001": {"PERSON"},
    "PII_002": {"PHONE_NUMBER", "CN_PHONE_NUMBER", "EMAIL_ADDRESS", "WECHAT_ID", "QQ_NUMBER"},
    "PII_003": {"ID_CARD", "CN_ID_CARD", "PASSPORT", "DRIVER_LICENSE", "US_SSN"},
    "PII_004": {"LOCATION"},
    "PII_005": {"STUDENT_ID"},
    "PII_006": {"PARENT_CONTACT"},
    "PII_007": {"CREDIT_CARD", "BANK_ACCOUNT"},
    "PII_008": set(),
    "PII_009": {"IP_ADDRESS", "URL"},
    "PII_010": set(),
    "PII_011": {"STUDENT_ID", "PARENT_CONTACT"},
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


@dataclass
class AudioOperatorSelection:
    """Resolved audio detector selection."""

    privacy_operator_ids: list[str] = field(default_factory=list)
    privacy_target_types: list[str] = field(default_factory=list)
    privacy_entity_types: list[str] = field(default_factory=list)
    content_safety_operator_ids: list[str] = field(default_factory=list)
    content_safety_target_labels: list[str] = field(default_factory=list)
    disabled_operator_ids: list[str] = field(default_factory=list)
    disabled_target_types: list[str] = field(default_factory=list)
    preserved_training_targets: list[str] = field(default_factory=list)
    privacy_enabled: bool = True
    content_safety_enabled: bool = True
    explicit: bool = False

    def model_dump(self) -> dict[str, Any]:
        return {
            "privacy_operator_ids": self.privacy_operator_ids,
            "privacy_target_types": self.privacy_target_types,
            "privacy_entity_types": self.privacy_entity_types,
            "content_safety_operator_ids": self.content_safety_operator_ids,
            "content_safety_target_labels": self.content_safety_target_labels,
            "disabled_operator_ids": self.disabled_operator_ids,
            "disabled_target_types": self.disabled_target_types,
            "preserved_training_targets": self.preserved_training_targets,
            "privacy_enabled": self.privacy_enabled,
            "content_safety_enabled": self.content_safety_enabled,
            "explicit": self.explicit,
        }


def resolve_audio_operator_selection(settings: Any, overrides: dict[str, Any] | None = None) -> AudioOperatorSelection:
    payload = _settings_payload(settings)
    payload.update({key: value for key, value in dict(overrides or {}).items() if value not in (None, "")})

    raw_privacy_ids = _normalize_ids(_list_value(payload, "privacy_operator_ids") + _list_value(payload, "audio_privacy_operator_ids"))
    raw_privacy_targets = _normalize_targets(_list_value(payload, "privacy_target_types") + _list_value(payload, "audio_privacy_target_types"))
    raw_content_ids = _normalize_ids(_list_value(payload, "content_safety_operator_ids") + _list_value(payload, "audio_content_safety_operator_ids"))
    raw_content_labels = _normalize_targets(_list_value(payload, "content_safety_target_labels") + _list_value(payload, "audio_content_safety_target_labels"))
    disabled_ids = _normalize_ids(_list_value(payload, "disabled_operator_ids"))
    disabled_targets = _normalize_targets(_list_value(payload, "disabled_target_types") + _list_value(payload, "preserved_training_targets"))

    disabled_source_ids = {_source_operator_id(item) for item in disabled_ids}
    disabled_targets.update(_target_for_source_operator(item) for item in disabled_source_ids if _target_for_source_operator(item))
    enabled_fields_present = bool(raw_privacy_ids or raw_privacy_targets or raw_content_ids or raw_content_labels)
    explicit = enabled_fields_present or bool(disabled_ids or disabled_targets)

    privacy_enabled_flag = _optional_bool(payload.get("enable_audio_privacy_detection"))
    content_enabled_flag = _optional_bool(payload.get("enable_audio_content_detection"))

    privacy_ids = _resolve_privacy_operator_ids(raw_privacy_ids, raw_privacy_targets, enabled_fields_present, disabled_source_ids)
    privacy_targets = _resolve_privacy_targets(raw_privacy_targets, privacy_ids, disabled_targets)
    privacy_entities = _resolve_privacy_entities(privacy_ids, privacy_targets)

    content_labels = _resolve_content_labels(raw_content_ids, raw_content_labels, enabled_fields_present, disabled_targets)
    content_ids = _resolve_content_operator_ids(content_labels, disabled_source_ids)

    privacy_enabled = bool(privacy_ids or privacy_targets or privacy_entities)
    content_enabled = bool(content_labels)
    if privacy_enabled_flag is not None:
        privacy_enabled = privacy_enabled_flag
    if content_enabled_flag is not None:
        content_enabled = content_enabled_flag

    if not privacy_enabled:
        privacy_ids = []
        privacy_targets = []
        privacy_entities = []
    if not content_enabled:
        content_ids = []
        content_labels = []

    return AudioOperatorSelection(
        privacy_operator_ids=sorted(privacy_ids),
        privacy_target_types=sorted(privacy_targets),
        privacy_entity_types=sorted(privacy_entities),
        content_safety_operator_ids=sorted(content_ids),
        content_safety_target_labels=sorted(content_labels),
        disabled_operator_ids=sorted(disabled_ids),
        disabled_target_types=sorted(disabled_targets),
        preserved_training_targets=sorted(_normalize_targets(_list_value(payload, "preserved_training_targets"))),
        privacy_enabled=privacy_enabled,
        content_safety_enabled=content_enabled,
        explicit=explicit,
    )


def _settings_payload(settings: Any) -> dict[str, Any]:
    if hasattr(settings, "model_dump"):
        return dict(settings.model_dump())
    return dict(getattr(settings, "__dict__", {}) or {})


def _list_value(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _normalize_ids(values: list[str]) -> set[str]:
    return {str(item).strip().upper() for item in values if str(item).strip()}


def _normalize_targets(values: list[str]) -> set[str]:
    return {str(item).strip().lower() for item in values if str(item).strip()}


def _source_operator_id(operator_id: str) -> str:
    normalized = str(operator_id or "").strip().upper()
    parts = normalized.split("_")
    if len(parts) >= 3 and parts[-2] in {"PII", "CSA"}:
        return f"{parts[-2]}_{parts[-1]}"
    return normalized


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _target_for_source_operator(operator_id: str) -> str:
    source_id = _source_operator_id(operator_id)
    if source_id in PII_TARGETS:
        return sorted(PII_TARGETS[source_id])[0]
    if source_id in CSA_LABELS:
        return CSA_LABELS[source_id]
    return ""


def _resolve_privacy_operator_ids(
    raw_ids: set[str],
    raw_targets: set[str],
    enabled_fields_present: bool,
    disabled_source_ids: set[str],
) -> set[str]:
    ids = {_source_operator_id(item) for item in raw_ids if _source_operator_id(item).startswith("PII_")}
    if raw_targets:
        for operator_id, targets in PII_TARGETS.items():
            if raw_targets & {target.lower() for target in targets}:
                ids.add(operator_id)
    if not ids and not enabled_fields_present:
        ids = set(PII_TARGETS)
    return ids - disabled_source_ids


def _resolve_privacy_targets(raw_targets: set[str], operator_ids: set[str], disabled_targets: set[str]) -> set[str]:
    targets = set(raw_targets)
    for operator_id in operator_ids:
        targets.update(target.lower() for target in PII_TARGETS.get(operator_id, set()))
    return targets - disabled_targets


def _resolve_privacy_entities(operator_ids: set[str], targets: set[str]) -> set[str]:
    entities: set[str] = set()
    for operator_id in operator_ids:
        entities.update(PII_ENTITY_TYPES.get(operator_id, set()))
    normalized_targets = {target.lower() for target in targets}
    for operator_id, operator_targets in PII_TARGETS.items():
        if normalized_targets & {target.lower() for target in operator_targets}:
            entities.update(PII_ENTITY_TYPES.get(operator_id, set()))
    return entities


def _resolve_content_labels(
    raw_ids: set[str],
    raw_labels: set[str],
    enabled_fields_present: bool,
    disabled_targets: set[str],
) -> set[str]:
    labels = set(raw_labels)
    for operator_id in raw_ids:
        source_id = _source_operator_id(operator_id)
        if source_id in CSA_LABELS:
            labels.add(CSA_LABELS[source_id])
    if not labels and not enabled_fields_present:
        labels = set(CSA_LABELS.values())
    return labels - disabled_targets


def _resolve_content_operator_ids(labels: set[str], disabled_source_ids: set[str]) -> set[str]:
    inverse = {label: operator_id for operator_id, label in CSA_LABELS.items()}
    ids = {inverse[label] for label in labels if label in inverse}
    return ids - disabled_source_ids

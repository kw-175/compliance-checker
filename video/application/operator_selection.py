"""Resolve video operator selection into existing picture/audio/text options."""

from __future__ import annotations

from typing import Any

from video.domain.models import ComplianceOperatorSelection, PreservedTrainingTarget
from video.domain.operators import CSA_LABELS, PII_TARGETS, VIDEO_OPERATOR_CATALOG, VPI_OPERATORS

_VPI_TARGET_BY_SOURCE = {item.source_operator_id: item.target_type for item in VPI_OPERATORS.values()}
_VPI_SOURCE_BY_TARGET = {item.target_type: item.source_operator_id for item in VPI_OPERATORS.values()}
_VISUAL_LABELS_BY_CSA = {
    operator_id: label.replace("content.", "visual.", 1)
    for operator_id, label in CSA_LABELS.items()
}
_CSA_BY_VISUAL_LABEL = {label: operator_id for operator_id, label in _VISUAL_LABELS_BY_CSA.items()}


class ResolvedOperatorSelection:
    """Resolved operator selection with downstream modality options."""

    def __init__(self, selection: ComplianceOperatorSelection, explicit: bool = False) -> None:
        self.selection = selection
        self.explicit = explicit
        self.disabled_operator_ids = _normalize_ids(selection.disabled_operator_ids)
        self.disabled_source_operator_ids = {
            _source_operator_id(item)
            for item in self.disabled_operator_ids
            if _source_operator_id(item)
        }
        self.disabled_target_types = _normalize_targets(selection.disabled_target_types + selection.preserved_training_targets)
        self.disabled_target_types.update(
            _normalize_targets([_target_for_source_operator(item) for item in self.disabled_source_operator_ids])
        )
        self._has_visual_sensitive_selection = bool(selection.visual_sensitive_object_operator_ids or selection.visual_sensitive_object_types)
        self._has_visual_safety_selection = bool(selection.visual_safety_operator_ids or selection.visual_safety_target_labels)
        self._has_privacy_selection = bool(
            selection.privacy_operator_ids
            or selection.privacy_target_types
        )
        self._has_audio_privacy_selection = bool(selection.audio_privacy_operator_ids)
        self._has_content_selection = bool(
            selection.content_safety_operator_ids
            or selection.content_safety_target_labels
        )
        self._has_audio_content_selection = bool(selection.audio_content_safety_operator_ids)
        self._has_any_enabled_selection = bool(
            self._has_visual_sensitive_selection
            or self._has_visual_safety_selection
            or self._has_privacy_selection
            or self._has_audio_privacy_selection
            or self._has_content_selection
            or self._has_audio_content_selection
        )
        self.visual_sensitive_object_types = self._resolve_visual_sensitive_object_types()
        self.visual_sensitive_object_operator_ids = self._source_vpi_ids(self.visual_sensitive_object_types)
        self.visual_safety_target_labels = self._resolve_csa_labels(
            selection.visual_safety_operator_ids,
            selection.visual_safety_target_labels,
            family_prefix="VVIS",
        )
        self.visual_safety_operator_ids = self._source_csa_ids(self.visual_safety_target_labels)
        self.privacy_operator_ids = self._resolve_pii_operator_ids(
            selection.privacy_operator_ids,
            family_selected=self._has_privacy_selection,
        )
        self.privacy_target_types = self._resolve_pii_target_types(selection.privacy_target_types, self.privacy_operator_ids)
        self.audio_privacy_operator_ids = self._resolve_pii_operator_ids(
            selection.audio_privacy_operator_ids,
            family_selected=self._has_audio_privacy_selection,
        )
        self.audio_privacy_target_types = self._resolve_pii_target_types([], self.audio_privacy_operator_ids)
        self.content_safety_target_labels = self._resolve_csa_labels(
            selection.content_safety_operator_ids,
            selection.content_safety_target_labels,
            family_prefix="VTXT",
        )
        self.content_safety_operator_ids = self._source_csa_ids(self.content_safety_target_labels)
        self.audio_content_safety_target_labels = self._resolve_csa_labels(
            selection.audio_content_safety_operator_ids,
            [],
            family_prefix="VAUD",
        )
        self.audio_content_safety_operator_ids = self._source_csa_ids(self.audio_content_safety_target_labels)

    def picture_options(self) -> dict[str, Any]:
        """Return options understood by the existing picture orchestrator."""
        options: dict[str, Any] = {
            "visual_sensitive_object_operator_ids": self.visual_sensitive_object_operator_ids,
            "visual_sensitive_object_types": self.visual_sensitive_object_types,
            "visual_safety_operator_ids": self.visual_safety_operator_ids,
            "visual_safety_target_labels": self.visual_safety_target_labels,
            "privacy_operator_ids": self.privacy_operator_ids,
            "privacy_target_types": self.privacy_target_types,
            "content_safety_operator_ids": self.content_safety_operator_ids,
            "content_safety_target_labels": self.content_safety_target_labels,
        }
        if self.explicit:
            options["enable_visual_sensitive_object_detection"] = bool(self.visual_sensitive_object_types)
            options["enable_visual_safety_detection"] = bool(self.visual_safety_target_labels)
            options["enable_text_privacy_detection"] = bool(self.privacy_operator_ids or self.privacy_target_types)
            options["enable_text_content_detection"] = bool(self.content_safety_target_labels)
        return options

    def audio_config_overrides(self) -> dict[str, Any]:
        """Return operator selection fields compatible with the audio text bridge."""
        return {
            "privacy_operator_ids": self.audio_privacy_operator_ids,
            "privacy_target_types": self.audio_privacy_target_types,
            "content_safety_operator_ids": self.audio_content_safety_operator_ids,
            "content_safety_target_labels": self.audio_content_safety_target_labels,
        }

    def preserved_targets(self, task_type: str = "") -> list[PreservedTrainingTarget]:
        preserved: dict[str, PreservedTrainingTarget] = {}
        for target in sorted(self.disabled_target_types):
            source_operator_id = _source_operator_for_target(target)
            video_operator_id = _video_operator_for_source(source_operator_id, target)
            preserved[target] = PreservedTrainingTarget(
                target_type=target,
                source_operator_id=source_operator_id,
                video_operator_id=video_operator_id,
                source_modality=_operator_modality(video_operator_id),
                preserved_for_task=task_type,
                metadata={"disabled_by": "target_type"},
            )
        for operator_id in sorted(self.disabled_operator_ids):
            source_operator_id = _source_operator_id(operator_id)
            target = _target_for_source_operator(source_operator_id)
            if not target:
                continue
            preserved.setdefault(
                target,
                PreservedTrainingTarget(
                    target_type=target,
                    source_operator_id=source_operator_id,
                    video_operator_id=operator_id if operator_id in VIDEO_OPERATOR_CATALOG else _video_operator_for_source(source_operator_id, target),
                    source_modality=_operator_modality(operator_id),
                    preserved_for_task=task_type,
                    metadata={"disabled_by": "operator_id"},
                ),
            )
        return list(preserved.values())

    def risk_allowed(self, *, operator_id: str = "", source_operator_id: str = "", target_type: str = "", category: str = "") -> bool:
        candidates = {
            _normalize_id(operator_id),
            _normalize_id(source_operator_id),
            _normalize_id(_source_operator_id(operator_id)),
            _normalize_id(_source_operator_id(source_operator_id)),
        }
        if candidates & self.disabled_operator_ids:
            return False
        target_candidates = _normalize_targets([target_type, category, _target_for_source_operator(source_operator_id), _target_for_source_operator(operator_id)])
        return not bool(target_candidates & self.disabled_target_types)

    def _resolve_visual_sensitive_object_types(self) -> list[str]:
        selected = _normalize_targets(self.selection.visual_sensitive_object_types)
        for operator_id in self.selection.visual_sensitive_object_operator_ids:
            source_id = _source_operator_id(operator_id)
            target = _VPI_TARGET_BY_SOURCE.get(source_id)
            if target:
                selected.add(target)
        if not selected and (not self._has_any_enabled_selection or self._has_visual_sensitive_selection):
            selected = {item.target_type for item in VPI_OPERATORS.values()}
        return sorted(selected - self.disabled_target_types)

    def _source_vpi_ids(self, targets: list[str]) -> list[str]:
        ids = {_VPI_SOURCE_BY_TARGET[target] for target in targets if target in _VPI_SOURCE_BY_TARGET}
        ids -= self.disabled_operator_ids
        ids -= self.disabled_source_operator_ids
        return sorted(ids)

    def _resolve_pii_operator_ids(self, raw_ids: list[str], family_selected: bool) -> list[str]:
        ids = {_source_operator_id(item) for item in raw_ids if _source_operator_id(item).startswith("PII_")}
        if not ids and (not self._has_any_enabled_selection or family_selected):
            ids = set(PII_TARGETS)
        ids -= self.disabled_operator_ids
        ids -= self.disabled_source_operator_ids
        return sorted(ids)

    def _resolve_pii_target_types(self, raw_targets: list[str], operator_ids: list[str]) -> list[str]:
        targets = _normalize_targets(raw_targets)
        for operator_id in operator_ids:
            targets.update(_normalize_targets(PII_TARGETS.get(operator_id, set())))
        return sorted(targets - self.disabled_target_types)

    def _resolve_csa_labels(self, raw_ids: list[str], raw_labels: list[str], family_prefix: str) -> list[str]:
        labels = _normalize_targets(raw_labels)
        if family_prefix == "VVIS":
            labels = {_to_visual_safety_label(label) for label in labels}
        for operator_id in raw_ids:
            source_id = _source_operator_id(operator_id)
            label = CSA_LABELS.get(source_id)
            if label:
                if family_prefix == "VVIS":
                    label = _to_visual_safety_label(label)
                labels.add(label)
        if family_prefix == "VVIS":
            family_selected = self._has_visual_safety_selection
        elif family_prefix == "VAUD":
            family_selected = self._has_audio_content_selection
        else:
            family_selected = self._has_content_selection
        if not labels and (not self._has_any_enabled_selection or family_selected):
            labels = set(_VISUAL_LABELS_BY_CSA.values()) if family_prefix == "VVIS" else set(CSA_LABELS.values())
        return sorted(labels - self.disabled_target_types)

    def _source_csa_ids(self, labels: list[str]) -> list[str]:
        inverse = {label: operator_id for operator_id, label in CSA_LABELS.items()}
        inverse.update(_CSA_BY_VISUAL_LABEL)
        ids = {inverse[label] for label in labels if label in inverse}
        ids -= self.disabled_operator_ids
        ids -= self.disabled_source_operator_ids
        return sorted(ids)


def resolve_operator_selection(raw: ComplianceOperatorSelection | dict[str, Any] | None, options: dict[str, object] | None = None) -> ResolvedOperatorSelection:
    options = dict(options or {})
    explicit = raw is not None or isinstance(options.get("operator_selection"), dict)
    payload: dict[str, Any] = {}
    if isinstance(raw, ComplianceOperatorSelection):
        payload = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        payload = dict(raw)
    nested = options.get("operator_selection")
    if isinstance(nested, dict):
        payload.update(nested)
    for key in ComplianceOperatorSelection.model_fields:
        if key in options and key not in payload:
            payload[key] = options[key]
    return ResolvedOperatorSelection(ComplianceOperatorSelection(**payload), explicit=explicit)


def _normalize_id(value: str) -> str:
    return str(value or "").strip().upper()


def _normalize_ids(values: list[str]) -> set[str]:
    return {_normalize_id(item) for item in values if str(item).strip()}


def _normalize_targets(values: list[str]) -> set[str]:
    return {str(item or "").strip().lower() for item in values if str(item or "").strip()}


def _to_visual_safety_label(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized.startswith("content."):
        return normalized.replace("content.", "visual.", 1)
    return normalized


def _source_operator_id(operator_id: str) -> str:
    normalized = _normalize_id(operator_id)
    if normalized in VIDEO_OPERATOR_CATALOG:
        return VIDEO_OPERATOR_CATALOG[normalized].source_operator_id
    parts = normalized.split("_")
    if len(parts) >= 3 and parts[-2] in {"PII", "CSA", "VPI"}:
        return f"{parts[-2]}_{parts[-1]}"
    return normalized


def _target_for_source_operator(operator_id: str) -> str:
    source_id = _source_operator_id(operator_id)
    if source_id in _VPI_TARGET_BY_SOURCE:
        return _VPI_TARGET_BY_SOURCE[source_id]
    if source_id in PII_TARGETS:
        return sorted(PII_TARGETS[source_id])[0]
    if source_id in CSA_LABELS:
        return CSA_LABELS[source_id]
    return ""


def _source_operator_for_target(target: str) -> str:
    target = str(target or "").strip().lower()
    if target in _VPI_SOURCE_BY_TARGET:
        return _VPI_SOURCE_BY_TARGET[target]
    for operator_id, targets in PII_TARGETS.items():
        if target in _normalize_targets(list(targets)):
            return operator_id
    for operator_id, label in CSA_LABELS.items():
        visual_label = _VISUAL_LABELS_BY_CSA.get(operator_id, "")
        if target in {label, visual_label, label.replace("content.", ""), visual_label.replace("visual.", "")}:
            return operator_id
    return ""


def _video_operator_for_source(source_operator_id: str, target: str) -> str:
    for operator in VIDEO_OPERATOR_CATALOG.values():
        if operator.source_operator_id == source_operator_id and (operator.target_type == target or target in operator.target_labels):
            return operator.operator_id
    return ""


def _operator_modality(operator_id: str) -> str:
    normalized = _normalize_id(operator_id)
    operator = VIDEO_OPERATOR_CATALOG.get(normalized)
    if operator:
        return operator.source_modality
    if normalized.startswith("PII_") or normalized.startswith("CSA_"):
        return "text"
    if normalized.startswith("VPI_"):
        return "picture"
    return ""

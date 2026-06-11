from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from text.config.settings import Settings


class ContentSafetySubOperator(BaseModel):
    sub_operator_id: str
    display_name: str
    target_labels: list[str] = Field(default_factory=list)
    description: str = ""
    prompt_profile: str = "default"
    policy_version: str = ""
    decision_source: str = "gpt52_content_safety"
    enabled: bool = True
    aliases: list[str] = Field(default_factory=list)


@lru_cache(maxsize=8)
def load_content_safety_sub_operators(operator_dir: str) -> dict[str, ContentSafetySubOperator]:
    directory = Path(operator_dir)
    operators: dict[str, ContentSafetySubOperator] = {}
    if not directory.exists():
        return operators
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            continue
        operator = ContentSafetySubOperator.model_validate(data)
        operators[operator.sub_operator_id] = operator
    return operators


def resolve_selected_sub_operators(
    settings: Settings,
    *,
    label_catalog: dict[str, dict],
) -> list[ContentSafetySubOperator]:
    registry = load_content_safety_sub_operators(str(settings.content_safety_operator_dir))
    enabled = {key: value for key, value in registry.items() if value.enabled}
    if not enabled:
        return []

    requested_ids = [item.strip() for item in settings.content_safety_operator_ids if str(item).strip()]
    if requested_ids:
        selected = [enabled[item] for item in requested_ids if item in enabled]
        if selected:
            return selected

    if settings.content_safety_target_labels:
        selected: list[ContentSafetySubOperator] = []
        for operator in enabled.values():
            operator_aliases = {alias.lower() for alias in operator.aliases}
            operator_aliases.update(label.lower() for label in operator.target_labels)
            for label in settings.content_safety_target_labels:
                normalized = str(label).strip().lower()
                if normalized in operator_aliases:
                    selected.append(operator)
                    break
                spec = label_catalog.get(label, {})
                if str(spec.get("risk_type", "")).lower() in operator_aliases:
                    selected.append(operator)
                    break
        if selected:
            unique: dict[str, ContentSafetySubOperator] = {}
            for operator in selected:
                unique[operator.sub_operator_id] = operator
            return list(unique.values())

    return list(enabled.values())

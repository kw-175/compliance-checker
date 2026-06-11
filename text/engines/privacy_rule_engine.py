from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from text.models.schemas import DetectionFinding, Severity

logger = logging.getLogger(__name__)

SENSITIVITY_RANK = {"S1": 1, "S2": 2, "S3": 3, "S4": 4}
TRAINING_RANK = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

SEVERITY_TO_SENSITIVITY = {
    Severity.LOW: "S1",
    Severity.MEDIUM: "S2",
    Severity.HIGH: "S3",
    Severity.CRITICAL: "S4",
}


def load_privacy_entity_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Privacy entity catalog could not be loaded: %s", exc)
        return {}
    entities = data.get("entities", data)
    if not isinstance(entities, dict):
        return {}
    return {str(key): value for key, value in entities.items() if isinstance(value, dict)}


def entity_spec_for(finding: DetectionFinding, catalog: dict[str, dict[str, Any]]) -> dict[str, Any]:
    risk_type = finding.risk_type
    if risk_type in catalog:
        return catalog[risk_type]
    for spec in catalog.values():
        aliases = {str(item) for item in spec.get("aliases", [])}
        if risk_type in aliases:
            return spec
    return {}


def normalize_privacy_type(risk_type: str, catalog: dict[str, dict[str, Any]]) -> str:
    if risk_type in catalog:
        return risk_type
    for entity_type, spec in catalog.items():
        aliases = {str(item) for item in spec.get("aliases", [])}
        if risk_type in aliases:
            return entity_type
    alias_map = {
        "phone": "phone_number",
        "bank_account": "bank_card",
        "payment_account": "bank_card",
        "psychological_record": "medical_record",
        "api_key": "secret",
        "token": "secret",
        "password": "secret",
    }
    return alias_map.get(risk_type, risk_type)


def privacy_rule_hit(
    finding: DetectionFinding,
    catalog: dict[str, dict[str, Any]],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = context or {}
    normalized_type = normalize_privacy_type(finding.risk_type, catalog)
    spec = entity_spec_for(finding, catalog)
    sensitivity = str(spec.get("default_sensitivity") or SEVERITY_TO_SENSITIVITY.get(finding.severity, "S2"))
    training = str(spec.get("default_training_admissibility") or _training_for_sensitivity(sensitivity))
    action = str(spec.get("default_action") or "mask")

    if _minor_context(context) and normalized_type in {
        "person_name",
        "phone_number",
        "email",
        "address",
        "student_id",
        "education_record",
        "parent_contact",
    }:
        sensitivity = _max_code(sensitivity, "S3", SENSITIVITY_RANK)
        training = _max_code(training, "T2", TRAINING_RANK)

    if normalized_type in {"id_card", "bank_card", "medical_record", "secret", "combined_identity"}:
        sensitivity = "S4"
        training = "T3"

    return {
        "rule_id": f"privacy.entity.{normalized_type}.default",
        "entity_type": normalized_type,
        "operator_id": str(spec.get("operator_id") or ""),
        "operator_name_zh": str(spec.get("name_zh") or normalized_type),
        "entity_name_zh": str(spec.get("name_zh") or normalized_type),
        "description_zh": str(spec.get("description_zh") or ""),
        "sensitivity_level": sensitivity,
        "training_admissibility": training,
        "action": action,
        "replacement": str(spec.get("replacement") or finding.redaction_suggestion or "<PII>"),
        "reason_zh": _rule_reason(normalized_type, sensitivity, training),
    }


def _minor_context(context: dict[str, Any]) -> bool:
    tokens = {
        str(context.get("audience") or "").lower(),
        str(context.get("subject_type") or "").lower(),
        str(context.get("scene") or "").lower(),
    }
    return bool(tokens.intersection({"minor", "minors", "student", "students", "child", "children", "education"}))


def _training_for_sensitivity(sensitivity: str) -> str:
    return {
        "S1": "T0",
        "S2": "T1",
        "S3": "T2",
        "S4": "T3",
    }.get(sensitivity, "T1")


def _max_code(left: str, right: str, rank: dict[str, int]) -> str:
    return right if rank.get(right, -1) > rank.get(left, -1) else left


def _rule_reason(entity_type: str, sensitivity: str, training: str) -> str:
    reasons = {
        "person_name": "姓名可直接或间接指向自然人，进入训练或标注前需要结合场景决定是否脱敏。",
        "phone_number": "手机号属于可直接联系个人的联系方式，默认需要遮蔽。",
        "email": "邮箱属于可联系或识别个人的信息，训练前应脱敏。",
        "social_account": "社交账号属于可联系或识别个人的信息，训练前应脱敏。",
        "id_card": "身份证件属于强身份标识，禁止明文进入普通训练链路。",
        "address": "精确地址会暴露个人位置，需要泛化或遮蔽。",
        "student_id": "学号等教育身份标识与学生主体强绑定，需要受限流转。",
        "education_record": "成绩、处分、成长档案等教育记录具有敏感性，需要复核或受限使用。",
        "parent_contact": "监护人联系方式与未成年人场景强相关，需脱敏后使用。",
        "bank_card": "金融账户信息属于极高敏感信息，默认禁止进入训练链路。",
        "payment_account": "支付账户信息属于极高敏感信息，默认禁止进入训练链路。",
        "medical_record": "医疗或心理记录属于高度敏感个人信息，默认禁止普通训练。",
        "secret": "密钥、令牌或密码泄露会产生安全风险，默认禁止流转。",
        "combined_identity": "多个身份属性组合会显著提高重识别风险，需要受限复核。",
    }
    base = reasons.get(entity_type, "该片段包含个人信息，需要按隐私策略进行治理。")
    return f"{base} 当前规则等级为{sensitivity}，训练准入为{training}。"

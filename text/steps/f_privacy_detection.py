from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    DetectionFinding,
    IngestUnit,
    PrivacyDetectionResult,
    Severity,
    TextSpan,
)

logger = logging.getLogger(__name__)

SEVERITY_WEIGHTS = {
    Severity.LOW: 0.30,
    Severity.MEDIUM: 0.55,
    Severity.HIGH: 0.80,
    Severity.CRITICAL: 1.00,
}
REGEX_FLAGS = {
    "IGNORECASE": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
}


@lru_cache(maxsize=4)
def _load_rules(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_flags(flag_names: list[str]) -> int:
    value = 0
    for name in flag_names:
        value |= REGEX_FLAGS.get(name, 0)
    return value


def _context(text: str, start: int, end: int, window: int = 40) -> tuple[str, str]:
    before = text[max(0, start - window):start]
    after = text[end:min(len(text), end + window)]
    return before, after


def _build_finding(
    unit: IngestUnit,
    *,
    policy_tag: str,
    risk_type: str,
    severity: Severity,
    replacement: str,
    match_text: str,
    start: int,
    end: int,
    source_tool: str,
    needs_adjudication: bool,
    hard_case_reason: str,
    confidence_override: float | None = None,
    explanation: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> DetectionFinding:
    before, after = _context(unit.text, start, end)
    confidence = confidence_override if confidence_override is not None else min(0.99, SEVERITY_WEIGHTS[severity] + 0.08)
    if needs_adjudication:
        confidence = max(0.4, confidence - 0.22)

    return DetectionFinding(
        doc_id=unit.doc_id,
        finding_type="privacy",
        risk_type=risk_type,
        policy_tag=policy_tag,
        severity=severity,
        confidence=round(confidence, 4),
        explanation=explanation or f"Matched privacy pattern for {risk_type}.",
        source_tool=source_tool,
        remediation_suggestion="redact" if replacement else "manual_review",
        redaction_suggestion=replacement,
        needs_adjudication=needs_adjudication,
        hard_case_reason=hard_case_reason,
        span=TextSpan(
            start=start,
            end=end,
            text=match_text,
            context_before=before,
            context_after=after,
        ),
        attributes=attributes or {},
    )


PRESIDIO_ENTITY_MAP = {
    "EMAIL_ADDRESS": ("email", "pii.email", Severity.LOW, "<EMAIL>"),
    "PHONE_NUMBER": ("phone", "pii.phone", Severity.MEDIUM, "<PHONE>"),
    "PERSON": ("person_name", "pii.person_name", Severity.LOW, "<PERSON>"),
    "ORGANIZATION": ("organization", "pii.organization", Severity.LOW, "<ORGANIZATION>"),
    "LOCATION": ("address", "pii.address", Severity.MEDIUM, "<ADDRESS>"),
    "CREDIT_CARD": ("bank_card", "pii.bank_card", Severity.HIGH, "<BANK_CARD>"),
    "CRYPTO": ("crypto_wallet", "pii.crypto_wallet", Severity.HIGH, "<CRYPTO_WALLET>"),
    "IBAN_CODE": ("bank_account", "pii.bank_account", Severity.HIGH, "<BANK_ACCOUNT>"),
    "BANK_ACCOUNT": ("bank_account", "pii.bank_account", Severity.HIGH, "<BANK_ACCOUNT>"),
    "IP_ADDRESS": ("ip_address", "pii.ip_address", Severity.LOW, "<IP_ADDRESS>"),
    "URL": ("url", "pii.url", Severity.LOW, "<URL>"),
    "ID_CARD": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "PASSPORT": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "DRIVER_LICENSE": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "US_SSN": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "US_DRIVER_LICENSE": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "US_PASSPORT": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "CN_ID_CARD": ("id_card", "pii.id_card", Severity.HIGH, "<ID_CARD>"),
    "CN_PHONE_NUMBER": ("phone", "pii.phone", Severity.MEDIUM, "<PHONE>"),
    "STUDENT_ID": ("student_id", "pii.student_id", Severity.MEDIUM, "<STUDENT_ID>"),
    "PARENT_CONTACT": ("parent_contact", "pii.parent_contact", Severity.HIGH, "<PARENT_CONTACT>"),
    "EDUCATION_RECORD": ("education_record", "pii.education_record", Severity.MEDIUM, "<EDU_RECORD>"),
    "WECHAT_ID": ("social_account", "pii.social_account.wechat", Severity.MEDIUM, "<WECHAT_ID>"),
    "QQ_NUMBER": ("social_account", "pii.social_account.qq", Severity.MEDIUM, "<QQ_NUMBER>"),
    "ALIPAY_ID": ("payment_account", "pii.payment_account.alipay", Severity.HIGH, "<ALIPAY_ID>"),
    "LICENSE_PLATE": ("vehicle_identifier", "pii.vehicle.license_plate", Severity.MEDIUM, "<LICENSE_PLATE>"),
}

COMBINATION_RISK_ALIASES = {
    "phone": "phone_number",
    "bank_account": "bank_card",
    "social_account": "parent_contact",
    "payment_account": "bank_card",
}

_BAD_PERSON_NAME_TEXT = {
    "本人手机号",
    "联系电话",
    "联系方式",
    "家庭住址",
    "家庭住址登记",
    "家庭住址登记为",
    "家长联系",
    "常用邮箱",
}
_CHINESE_SURNAME_CHARS = set("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉龚程邢裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温庄晏柴瞿阎连习容向古易廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公")
_NAME_STOP_PREFIXES = (
    "由", "负责", "本学期", "学号", "同学", "老师", "监护", "家长", "学生", "今年",
    "手机号", "电话", "邮箱", "地址", "身份证", "成绩", "，", "。", "；", ";", "\n", " "
)
_ID_CARD_RE = re.compile(r"^[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]$")
_CN_MOBILE_RE = re.compile(r"^(?:\+?86[-\s]?)?1[3-9]\d{9}$")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SPOKEN_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "○": "0",
    "洞": "0",
    "O": "0",
    "o": "0",
    "幺": "1",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}
_PUBLIC_ORGANIZATION_SUFFIXES = (
    "大学",
    "学院",
    "学校",
    "中学",
    "小学",
    "幼儿园",
    "教育局",
    "研究中心",
    "研究院",
    "出版社",
    "平台",
    "联盟",
    "基地",
    "公司",
    "集团",
    "医院",
)
_PUBLIC_ORGANIZATION_EXACT = {
    "清华大学",
    "北京大学",
    "北大",
    "复旦大学",
    "中国科技大学",
    "麻省理工",
    "海南医科大学",
    "学堂在线",
    "教育部在线教育研究中心",
    "世界慕课联盟",
}


def _call_presidio_analyzer(unit: IngestUnit, settings: Settings) -> tuple[list[dict[str, Any]], str]:
    if not settings.enable_presidio or not settings.presidio_analyzer_endpoint:
        return [], ""

    try:
        import httpx

        language = _presidio_language_for(unit, settings)
        response = httpx.post(
            settings.presidio_analyzer_endpoint,
            json={
                "text": unit.text,
                "language": language,
                "score_threshold": settings.presidio_score_threshold,
            },
            timeout=settings.presidio_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)], ""
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)], ""
        return [], "presidio_unexpected_response"
    except Exception as exc:
        logger.warning("Presidio analyzer failed for %s: %s", unit.doc_id, exc)
        return [], "presidio_service_unavailable"


def _presidio_language_for(unit: IngestUnit, settings: Settings) -> str:
    def normalize(language: str) -> str:
        normalized = (language or "").strip().lower().replace("_", "-")
        if normalized in {"zh", "zh-cn", "zh-hans", "chinese", "cn"}:
            return "zh"
        if normalized in {"en", "en-us", "en-gb", "english"}:
            return "en"
        return normalized

    configured = normalize(getattr(settings, "presidio_language", "auto"))
    supported_raw = getattr(settings, "presidio_supported_languages", "en,zh")
    supported = {
        normalize(item)
        for item in supported_raw.split(",")
        if item.strip()
    }
    fallback = normalize(getattr(settings, "presidio_language_fallback", "en")) or "en"

    if configured and configured != "auto":
        return configured if configured in supported else fallback

    inferred = normalize(unit.language or "")
    return inferred if inferred in supported else fallback


def _presidio_finding(unit: IngestUnit, item: dict[str, Any], settings: Settings) -> DetectionFinding | None:
    entity_type = str(item.get("entity_type") or item.get("type") or "").upper()
    start = item.get("start")
    end = item.get("end")
    score = item.get("score", 0.0)
    try:
        start_i = int(start)
        end_i = int(end)
        score_f = float(score)
    except (TypeError, ValueError):
        return None
    if start_i < 0 or end_i <= start_i or end_i > len(unit.text):
        return None
    if score_f < settings.presidio_score_threshold:
        return None

    risk_type, policy_tag, severity, replacement = PRESIDIO_ENTITY_MAP.get(
        entity_type,
        ("pii_entity", f"pii.presidio.{entity_type.lower() or 'unknown'}", Severity.MEDIUM, "<PII>"),
    )
    needs_adjudication = risk_type in {
        "person_name",
        "address",
        "bank_card",
        "bank_account",
        "id_card",
        "student_id",
        "parent_contact",
        "social_account",
        "payment_account",
        "vehicle_identifier",
    }
    hard_case_reason = "context_dependent_pii" if needs_adjudication else ""
    return _build_finding(
        unit,
        policy_tag=policy_tag,
        risk_type=risk_type,
        severity=severity,
        replacement=replacement,
        match_text=unit.text[start_i:end_i],
        start=start_i,
        end=end_i,
        source_tool="presidio_analyzer",
        needs_adjudication=needs_adjudication,
        hard_case_reason=hard_case_reason,
        confidence_override=score_f,
        explanation=f"Presidio detected {entity_type or 'PII'} with score {score_f:.3f}.",
        attributes={"presidio_entity_type": entity_type, "presidio_result": item},
    )


def _deduplicate(findings: list[DetectionFinding]) -> list[DetectionFinding]:
    deduped: dict[tuple[int, int, str], DetectionFinding] = {}
    for finding in findings:
        if finding.span is None:
            continue
        key = (finding.span.start, finding.span.end, finding.policy_tag)
        existing = deduped.get(key)
        if existing is None or finding.confidence > existing.confidence:
            deduped[key] = finding
    return list(deduped.values())


def _is_cjk_char(value: str) -> bool:
    return len(value) == 1 and "\u4e00" <= value <= "\u9fff"


def _context_contains(text: str, start: int, end: int, markers: tuple[str, ...], window: int = 12) -> bool:
    context = text[max(0, start - window):min(len(text), end + window)].lower()
    return any(marker.lower() in context for marker in markers)


def _spoken_digits(value: str) -> str:
    digits: list[str] = []
    for char in str(value or ""):
        if char.isdigit():
            digits.append(char)
        elif char in _SPOKEN_DIGIT_MAP:
            digits.append(_SPOKEN_DIGIT_MAP[char])
        elif char in {"X", "x"}:
            digits.append("X")
    return "".join(digits)


def _looks_public_organization(value: str) -> bool:
    text = re.sub(r"\s+", "", str(value or "").strip())
    if not text:
        return False
    if text in _PUBLIC_ORGANIZATION_EXACT:
        return True
    if any(name in text for name in _PUBLIC_ORGANIZATION_EXACT):
        return True
    return any(text.endswith(suffix) for suffix in _PUBLIC_ORGANIZATION_SUFFIXES)


def _with_span(unit: IngestUnit, finding: DetectionFinding, start: int, end: int, *, explanation: str | None = None) -> DetectionFinding:
    before, after = _context(unit.text, start, end)
    return finding.model_copy(
        deep=True,
        update={
            "span": TextSpan(start=start, end=end, text=unit.text[start:end], context_before=before, context_after=after),
            "explanation": explanation or finding.explanation,
        },
    )


def _expand_single_char_cn_name(unit: IngestUnit, finding: DetectionFinding) -> DetectionFinding:
    span = finding.span
    if span is None or finding.risk_type != "person_name" or len(span.text) != 1:
        return finding
    if span.text not in _CHINESE_SURNAME_CHARS:
        return finding
    end = span.end
    while end < len(unit.text) and end - span.start < 4:
        tail = unit.text[end:end + 4]
        if any(tail.startswith(prefix) for prefix in _NAME_STOP_PREFIXES):
            break
        ch = unit.text[end]
        if not _is_cjk_char(ch):
            break
        end += 1
    if end - span.start < 2:
        return finding
    return _with_span(
        unit,
        finding,
        span.start,
        end,
        explanation=f"识别到中文姓名“{unit.text[span.start:end]}”，位于教育记录上下文中，需要进入隐私治理。",
    )


def _trim_address_span(unit: IngestUnit, finding: DetectionFinding) -> DetectionFinding | None:
    span = finding.span
    if span is None or finding.risk_type != "address":
        return finding
    text = span.text.strip()
    if not text or "共同出现" in text or text.startswith(("、", "，", ",")):
        return None
    start = span.start + (len(span.text) - len(span.text.lstrip()))
    end = span.end - (len(span.text) - len(span.text.rstrip()))
    value = unit.text[start:end]
    for prefix in ("登记为", "为", "是"):
        if value.startswith(prefix):
            start += len(prefix)
            value = unit.text[start:end]
            break
    stop_markers = ("，监护人", "，家长", "，联系电话", "，家校", "，邮箱", "。", "；", ";", "\n")
    stop_positions = [value.find(marker) for marker in stop_markers if value.find(marker) >= 0]
    if stop_positions:
        end = start + min(stop_positions)
    if end <= start or end - start < 4:
        return None
    return _with_span(
        unit,
        finding,
        start,
        end,
        explanation=f"识别到具体地址“{unit.text[start:end]}”，需要泛化或遮蔽。",
    )


def _inside_any(span: TextSpan, containers: list[DetectionFinding], *, same_type: bool = False, risk_type: str = "") -> bool:
    for other in containers:
        other_span = other.span
        if other_span is None:
            continue
        if same_type and other.risk_type != risk_type:
            continue
        if other_span.start <= span.start and span.end <= other_span.end and (other_span.start, other_span.end) != (span.start, span.end):
            return True
    return False


def _post_process_findings(unit: IngestUnit, findings: list[DetectionFinding]) -> list[DetectionFinding]:
    expanded = [_expand_single_char_cn_name(unit, finding) for finding in findings]
    normalized: list[DetectionFinding] = []
    for finding in expanded:
        span = finding.span
        if span is None:
            normalized.append(finding)
            continue
        source = finding.source_tool.lower()
        value = span.text.strip()
        if finding.risk_type == "organization" and _looks_public_organization(value):
            continue
        if finding.risk_type == "person_name" and "person_name_cn" in source:
            if value in _BAD_PERSON_NAME_TEXT or any(token in value for token in ("手机号", "联系电话", "联系方式", "住址", "邮箱")):
                continue
            if finding.confidence <= 0.45 and (len(value) > 4 or not all(_is_cjk_char(ch) or ch == "·" for ch in value)):
                continue
        if finding.risk_type == "phone_number" and "phone_generic" in source:
            digits = re.sub(r"\D", "", value)
            if _ID_CARD_RE.match(digits):
                continue
            if not _CN_MOBILE_RE.match(value) and not re.search(r"[-\s()]", value):
                if not _context_contains(unit.text, span.start, span.end, ("电话", "手机号", "联系方式", "联系电话", "tel", "phone")):
                    continue
        if finding.risk_type == "phone_number" and "phone_spoken" in source:
            digits = _spoken_digits(value)
            if len(digits) != 11 or not _CN_MOBILE_RE.match(digits):
                continue
            finding.attributes["speech_normalized_value"] = digits
        if finding.risk_type == "id_card" and "id_card_spoken" in source:
            digits = _spoken_digits(value)
            if not _ID_CARD_RE.match(digits):
                continue
            finding.attributes["speech_normalized_value"] = digits
        if finding.risk_type == "student_id" and "student_id_spoken" in source:
            digits = _spoken_digits(value)
            if len(digits) < 5:
                continue
            finding.attributes["speech_normalized_value"] = digits
        if finding.risk_type == "bank_card" and _ID_CARD_RE.match(re.sub(r"\D", "", value)):
            continue
        if finding.risk_type == "address":
            trimmed = _trim_address_span(unit, finding)
            if trimmed is None:
                continue
            finding = trimmed
        normalized.append(finding)

    strong_spans = [
        finding for finding in normalized
        if finding.span is not None and finding.risk_type in {"id_card", "bank_card", "email", "student_id", "secret"}
    ]
    filtered: list[DetectionFinding] = []
    for finding in normalized:
        span = finding.span
        if span is None:
            filtered.append(finding)
            continue
        if finding.risk_type == "phone_number" and _inside_any(span, strong_spans):
            continue
        if finding.risk_type == "parent_contact" and any(
            other.span is not None and other.risk_type in {"email", "phone", "phone_number"} and other.span.start <= span.start and span.end <= other.span.end
            for other in normalized
        ):
            continue
        if finding.risk_type == "email" and not _EMAIL_RE.fullmatch(span.text):
            continue
        filtered.append(finding)
    return _deduplicate(filtered)


def run(
    ingest_units: list[IngestUnit],
    settings: Settings | None = None,
) -> list[PrivacyDetectionResult]:
    settings = settings or get_settings()
    rules = _load_rules(str(settings.pii_rules_path))
    pattern_rules = rules.get("patterns", {})
    combination_rules = rules.get("combination_rules", {})

    results: list[PrivacyDetectionResult] = []
    for unit in ingest_units:
        findings: list[DetectionFinding] = []
        hard_case_reasons: list[str] = []
        distinct_types: set[str] = set()
        provider_name = "rule_pii_detector"
        is_degraded = False

        presidio_items, presidio_error = _call_presidio_analyzer(unit, settings)
        if presidio_error:
            is_degraded = True
            hard_case_reasons.append(presidio_error)
        if presidio_items:
            provider_name = "presidio+rule_pii_detector"
            for item in presidio_items:
                finding = _presidio_finding(unit, item, settings)
                if finding is None:
                    continue
                findings.append(finding)
                distinct_types.add(finding.risk_type)
                if finding.needs_adjudication and finding.hard_case_reason:
                    hard_case_reasons.append(finding.hard_case_reason)

        for rule_name, rule in pattern_rules.items():
            pattern = str(rule["regex"])
            flags = _resolve_flags(list(rule.get("flags", [])))
            compiled = re.compile(pattern, flags)
            capture_group = int(rule.get("capture_group", 0))
            severity = Severity(rule["severity"])
            policy_tag = str(rule["policy_tag"])
            risk_type = str(rule["risk_type"])
            replacement = str(rule.get("redaction", ""))

            for match in compiled.finditer(unit.text):
                start, end = match.span(capture_group)
                match_text = match.group(capture_group)
                needs_adjudication = risk_type in {"person_name", "education_record", "bank_card"}
                hard_case_reason = "context_dependent_pii" if needs_adjudication else ""
                if needs_adjudication and "context_dependent_pii" not in hard_case_reasons:
                    hard_case_reasons.append("context_dependent_pii")

                findings.append(
                    _build_finding(
                        unit,
                        policy_tag=policy_tag,
                        risk_type=risk_type,
                        severity=severity,
                        replacement=replacement,
                        match_text=match_text,
                        start=start,
                        end=end,
                        source_tool=f"privacy_rule_engine.{rule_name}",
                        needs_adjudication=needs_adjudication,
                        hard_case_reason=hard_case_reason,
                    )
                )
                distinct_types.add(risk_type)

        findings = _post_process_findings(unit, _deduplicate(findings))

        combined_rule = combination_rules.get("combined_identity")
        if combined_rule:
            base_types = set(combined_rule.get("base_types", []))
            matched_base_types = {
                COMBINATION_RISK_ALIASES.get(finding.risk_type, finding.risk_type)
                for finding in findings
                if finding.risk_type in base_types or COMBINATION_RISK_ALIASES.get(finding.risk_type) in base_types
            }
            if len(matched_base_types) >= settings.privacy_combination_threshold:
                severity = Severity(combined_rule["severity"])
                needs_adjudication = True
                hard_case_reasons.append("combined_identity")
                findings.append(
                    DetectionFinding(
                        doc_id=unit.doc_id,
                        finding_type="privacy",
                        risk_type=str(combined_rule["risk_type"]),
                        policy_tag=str(combined_rule["policy_tag"]),
                        severity=severity,
                        confidence=0.92,
                        explanation=(
                            f"Detected {len(matched_base_types)} distinct identity attributes "
                            "that can combine into a stronger personal profile."
                        ),
                        source_tool="privacy_rule_engine.combined_identity",
                        remediation_suggestion="restrict_and_review",
                        redaction_suggestion=str(combined_rule.get("redaction", "")),
                        needs_adjudication=needs_adjudication,
                        hard_case_reason="combined_identity",
                        span=None,
                        attributes={"matched_types": sorted(matched_base_types)},
                    )
                )

        risk_score = max(
            (SEVERITY_WEIGHTS[finding.severity] * finding.confidence for finding in findings),
            default=0.0,
        )
        needs_adjudication = any(finding.needs_adjudication for finding in findings)
        if needs_adjudication and "manual_resolution_needed" not in hard_case_reasons:
            hard_case_reasons.append("manual_resolution_needed")

        summary = "No privacy risks detected."
        if findings:
            summary = f"Detected {len(findings)} privacy findings."

        results.append(
            PrivacyDetectionResult(
                run_id=unit.run_id,
                doc_id=unit.doc_id,
                text_hash=unit.text_hash,
                pii_count=len(findings),
                risk_score=round(risk_score, 4),
                summary=summary,
                findings=findings,
                needs_adjudication=needs_adjudication,
                hard_case_reasons=sorted(set(hard_case_reasons)),
                provider_name=provider_name,
                provider_version="presidio-analyzer+rules" if provider_name.startswith("presidio") else "builtin-2026.04",
                is_degraded=is_degraded,
            )
        )

    logger.info("Privacy detection completed: %d documents", len(results))
    return results

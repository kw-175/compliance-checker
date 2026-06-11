from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentRuleHit:
    rule_id: str
    policy_tag: str
    risk_type: str
    severity: str
    score: float
    evidence: str
    start: int
    end: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "policy_tag": self.policy_tag,
            "risk_type": self.risk_type,
            "severity": self.severity,
            "score": self.score,
            "evidence": self.evidence,
            "start": self.start,
            "end": self.end,
            "reason": self.reason,
        }


FALLBACK_RULES: dict[str, dict[str, Any]] = {
    "violence": {
        "policy_tag": "content.violent",
        "risk_type": "violence",
        "severity": "high",
        "keywords": ["bomb", "terrorism", "kill", "weapon", "shooting"],
    },
    "sexual": {
        "policy_tag": "content.pornographic",
        "risk_type": "pornographic_content",
        "severity": "high",
        "keywords": ["porn", "explicit sex", "sexual service"],
    },
    "hate": {
        "policy_tag": "content.hate",
        "risk_type": "hate_speech",
        "severity": "high",
        "keywords": ["racial hatred", "genocide", "exterminate"],
    },
    "self_harm": {
        "policy_tag": "content.self_harm",
        "risk_type": "self_harm",
        "severity": "critical",
        "keywords": ["suicide", "self harm", "end my life"],
    },
    "jailbreak": {
        "policy_tag": "content.jailbreak",
        "risk_type": "jailbreak_attempt",
        "severity": "high",
        "keywords": ["ignore previous instructions", "bypass safety", "jailbreak"],
    },
}


def _load_rule_categories(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return FALLBACK_RULES
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Content rule file could not be loaded, using fallback rules: %s", exc)
        return FALLBACK_RULES
    categories = data.get("categories", data)
    if not isinstance(categories, dict):
        return FALLBACK_RULES
    parsed = {str(rule_id): spec for rule_id, spec in categories.items() if isinstance(spec, dict)}
    return parsed or FALLBACK_RULES


def _find_keyword_matches(text: str, keyword: str) -> list[tuple[int, int]]:
    if not keyword:
        return []
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(keyword), text, flags=re.IGNORECASE)
    ]


def recall_content_rules(
    text: str,
    rules_path: Path,
    selected_labels: list[str],
) -> list[ContentRuleHit]:
    selected = {item.lower() for item in selected_labels if item}
    hits: list[ContentRuleHit] = []
    for rule_id, spec in _load_rule_categories(rules_path).items():
        policy_tag = str(spec.get("policy_tag") or f"content.{rule_id}")
        risk_type = str(spec.get("risk_type") or rule_id)
        if selected and policy_tag.lower() not in selected and risk_type.lower() not in selected:
            aliases = {str(item).lower() for item in spec.get("aliases", []) if item}
            if not selected.intersection(aliases):
                continue
        severity = str(spec.get("severity") or "medium").lower()
        keywords = [str(item) for item in spec.get("keywords", []) if str(item).strip()]
        for keyword in keywords:
            for start, end in _find_keyword_matches(text, keyword):
                hits.append(
                    ContentRuleHit(
                        rule_id=rule_id,
                        policy_tag=policy_tag,
                        risk_type=risk_type,
                        severity=severity,
                        score=0.85,
                        evidence=text[start:end],
                        start=start,
                        end=end,
                        reason=f"Keyword matched content-safety rule {rule_id}.",
                    )
                )
    return hits


def summarize_rule_hits(rule_hits: list[ContentRuleHit]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in rule_hits]

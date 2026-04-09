# ──────────────────────────────────────────────────────────────
# 统一策略模型
# ──────────────────────────────────────────────────────────────
#
# 将硬编码阈值、静态规则替换为 Profile 驱动、可审计、可解释的
# 策略评估框架。支持按场景（法律文书、社媒评论、预训练语料等）
# 差异化配置，并产出完整的评分分解和规则命中轨迹。
# ──────────────────────────────────────────────────────────────

"""跨模态统一策略模型：Profile / ScoreBreakdown / RuleTrace / PolicyResult。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from common.enums import TrustLevel, UnifiedDecision

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── 规则命中轨迹 ─────────────────────────────────────────


class RuleTrace(BaseModel):
    """
    单条规则的命中轨迹。

    记录策略评估过程中每条规则的：
    - 是否命中
    - 命中时使用的阈值和实际值
    - 来源 provider 信息
    - 是否因降级影响了判定
    """
    rule_name: str                   # 规则名称，如 "secret_leak_check"
    rule_description: str = ""       # 规则描述
    matched: bool = False            # 是否命中
    threshold: float = 0.0           # 规则阈值
    actual_value: float = 0.0        # 实际检测值
    provider: str = ""               # 产出该信号的 provider
    degraded: bool = False           # 该规则的信号是否来自降级 provider
    contribution: str = ""           # 对最终决策的贡献描述
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── 评分维度分解 ─────────────────────────────────────────


class ScoreDimension(BaseModel):
    """单个评分维度。"""
    dimension_name: str              # 维度名称，如 "secrets" / "safety" / "privacy"
    score: float = 1.0               # 维度评分 [0.0, 1.0]
    weight: float = 1.0              # 维度权重
    rule_traces: list[RuleTrace] = Field(default_factory=list)
    explanation: str = ""            # 维度评分解释


class ScoreBreakdown(BaseModel):
    """
    多维评分分解。

    将策略决策过程拆解为多个可独立审计的评分维度，
    每个维度有独立的分数、权重和规则命中轨迹。
    """
    dimensions: list[ScoreDimension] = Field(default_factory=list)
    min_score: float = 1.0           # 最低维度分 → 决定最终决策
    weighted_score: float = 1.0      # 加权总分（可选使用）

    def compute_min(self) -> float:
        """计算所有维度的最低分。"""
        if not self.dimensions:
            return 1.0
        self.min_score = min(d.score for d in self.dimensions)
        return self.min_score

    def compute_weighted(self) -> float:
        """计算加权总分。"""
        if not self.dimensions:
            return 1.0
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return 1.0
        self.weighted_score = sum(d.score * d.weight for d in self.dimensions) / total_weight
        return self.weighted_score


# ── 策略结果 ─────────────────────────────────────────────


class PolicyResult(BaseModel):
    """
    统一策略评估结果。

    相比现有系统仅输出 decision + reasons 的模式，新增了：
    - score_breakdown: 完整的多维评分分解
    - rule_traces: 所有规则的命中轨迹
    - trust_level: 总体可信等级
    - review_suggestions: 人类复核建议
    - degrade_summary: 降级事件摘要
    """
    decision: UnifiedDecision = UnifiedDecision.REVIEW
    reasons: list[str] = Field(default_factory=list)
    score_breakdown: Optional[ScoreBreakdown] = None
    trust_level: TrustLevel = TrustLevel.FULL
    review_suggestions: list[str] = Field(default_factory=list)
    degrade_summary: str = ""
    profile_name: str = "default"
    evaluated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Profile 配置 ─────────────────────────────────────────


class PolicyProfileConfig(BaseModel):
    """
    策略 Profile 配置。

    从 YAML 加载，定义特定场景下的阈值、权重和规则集合。
    例如：法律文书场景的 PII 阈值可能比预训练语料更严格。
    """
    profile_name: str = "default"
    description: str = ""

    # 各维度阈值（可被 YAML 覆盖）
    secret_threshold: float = 0.0    # 有 secret → 直接 0 分
    safety_threshold_unsafe: float = 0.0
    safety_threshold_controversial: float = 0.5
    privacy_high_threshold: int = 5  # PII > N → 低分
    privacy_low_threshold: int = 0
    compliance_copyleft_threshold: int = 0
    text_scan_high_threshold: int = 20
    text_scan_low_threshold: int = 5

    # 决策映射阈值
    reject_below: float = 0.0
    quarantine_below: float = 0.3
    review_below: float = 0.6

    # 维度权重
    dimension_weights: dict[str, float] = Field(default_factory=lambda: {
        "secrets": 1.0,
        "safety": 1.0,
        "privacy": 1.0,
        "compliance": 1.0,
        "text_scan": 1.0,
    })

    # 失败处理策略
    failure_policy: str = "fail_closed"

    metadata: dict[str, Any] = Field(default_factory=dict)


def load_policy_profile(
    profile_name: str,
    profiles_dir: Path | None = None,
) -> PolicyProfileConfig:
    """
    从 YAML 文件加载策略 Profile。

    Args:
        profile_name: Profile 名称（对应 YAML 文件名）
        profiles_dir: Profile 目录路径（默认为 common/profiles/）

    Returns:
        PolicyProfileConfig 实例
    """
    if profiles_dir is None:
        profiles_dir = Path(__file__).parent / "profiles"

    config_path = profiles_dir / f"{profile_name}.yaml"
    if not config_path.exists():
        logger.warning(
            "策略 Profile '%s' 未找到（路径: %s），使用默认配置",
            profile_name, config_path,
        )
        return PolicyProfileConfig(profile_name=profile_name)

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # 移除 YAML 中的 profile_name 以避免与显式参数冲突
        raw.pop("profile_name", None)
        return PolicyProfileConfig(profile_name=profile_name, **raw)
    except Exception as e:
        logger.warning("加载策略 Profile '%s' 失败: %s，使用默认配置", profile_name, e)
        return PolicyProfileConfig(profile_name=profile_name)


# ── 统一策略评估 ─────────────────────────────────────────


def evaluate_with_profile(
    evidence_units: list,
    profile: PolicyProfileConfig | None = None,
    degrade_events: list | None = None,
) -> PolicyResult:
    """
    基于统一证据单元和策略 Profile 进行多维评估。

    五维评分体系：
    - secrets（密钥泄露）
    - safety（内容安全）
    - privacy（隐私 PII）
    - compliance（许可证合规）
    - text_scan（文本扫描命中）

    Args:
        evidence_units: EvidenceUnit 列表
        profile: 策略 Profile 配置（None 使用默认）
        degrade_events: 降级事件列表（影响可信等级）

    Returns:
        PolicyResult 包含完整的评分分解和规则命中轨迹
    """
    if profile is None:
        profile = load_policy_profile("default")

    degrade_events = degrade_events or []
    dimensions: list[ScoreDimension] = []

    # ── 维度 1：密钥泄露 ──────────────────────────────────
    secret_units = [u for u in evidence_units if u.category == "secret"]
    secret_traces: list[RuleTrace] = []
    secret_score = 1.0
    if secret_units:
        secret_score = profile.secret_threshold
        for u in secret_units:
            secret_traces.append(RuleTrace(
                rule_name="secret_leak_check",
                rule_description=f"检测到密钥泄露: {u.sub_category}",
                matched=True,
                threshold=profile.secret_threshold,
                actual_value=0.0,
                provider=u.provider,
                degraded=any(e.step_name.startswith("step_b2a") for e in degrade_events),
                contribution="有密钥泄露 → 直接最低分",
            ))
    dimensions.append(ScoreDimension(
        dimension_name="secrets",
        score=secret_score,
        weight=profile.dimension_weights.get("secrets", 1.0),
        rule_traces=secret_traces,
        explanation=f"检测到 {len(secret_units)} 个密钥泄露" if secret_units else "未检测到密钥泄露",
    ))

    # ── 维度 2：内容安全 ──────────────────────────────────
    safety_units = [u for u in evidence_units if u.category == "safety"]
    safety_traces: list[RuleTrace] = []
    safety_score = 1.0
    for u in safety_units:
        if u.sub_category == "unsafe":
            safety_score = min(safety_score, profile.safety_threshold_unsafe)
            safety_traces.append(RuleTrace(
                rule_name="safety_unsafe_check",
                rule_description=f"内容被判为不安全: {u.explanation}",
                matched=True, threshold=profile.safety_threshold_unsafe,
                actual_value=0.0, provider=u.provider,
                contribution="unsafe → 0 分",
            ))
        elif u.sub_category == "controversial":
            safety_score = min(safety_score, profile.safety_threshold_controversial)
            safety_traces.append(RuleTrace(
                rule_name="safety_controversial_check",
                rule_description=f"内容存在争议: {u.explanation}",
                matched=True, threshold=profile.safety_threshold_controversial,
                actual_value=0.5, provider=u.provider,
                contribution="controversial → 0.5 分",
            ))
    dimensions.append(ScoreDimension(
        dimension_name="safety",
        score=safety_score,
        weight=profile.dimension_weights.get("safety", 1.0),
        rule_traces=safety_traces,
        explanation=f"检测到 {len(safety_units)} 条安全风险" if safety_units else "内容安全",
    ))

    # ── 维度 3：隐私 (PII) ────────────────────────────────
    pii_units = [u for u in evidence_units if u.category == "pii"]
    pii_traces: list[RuleTrace] = []
    pii_score = 1.0
    pii_count = len(pii_units)
    if pii_count > profile.privacy_high_threshold:
        pii_score = 0.3
        pii_traces.append(RuleTrace(
            rule_name="pii_density_high",
            rule_description=f"PII 密度过高: {pii_count} 个实体 > {profile.privacy_high_threshold}",
            matched=True, threshold=float(profile.privacy_high_threshold),
            actual_value=float(pii_count),
            contribution="PII 密度高 → 0.3 分",
        ))
    elif pii_count > profile.privacy_low_threshold:
        pii_score = 0.7
        pii_traces.append(RuleTrace(
            rule_name="pii_density_low",
            rule_description=f"检测到 PII: {pii_count} 个实体",
            matched=True, threshold=float(profile.privacy_low_threshold),
            actual_value=float(pii_count),
            contribution="存在 PII → 0.7 分",
        ))
    dimensions.append(ScoreDimension(
        dimension_name="privacy",
        score=pii_score,
        weight=profile.dimension_weights.get("privacy", 1.0),
        rule_traces=pii_traces,
        explanation=f"检测到 {pii_count} 个 PII 实体" if pii_count else "未检测到 PII",
    ))

    # ── 维度 4：许可证合规 ────────────────────────────────
    license_units = [u for u in evidence_units if u.category == "license"]
    compliance_traces: list[RuleTrace] = []
    compliance_score = 1.0
    copyleft_count = sum(
        1 for u in license_units
        if any(k in (u.sub_category or "").lower() for k in ["gpl", "agpl", "copyleft"])
    )
    if copyleft_count > profile.compliance_copyleft_threshold:
        compliance_score = 0.2
        compliance_traces.append(RuleTrace(
            rule_name="copyleft_license_check",
            rule_description=f"检测到 {copyleft_count} 个 copyleft 许可证",
            matched=True, threshold=float(profile.compliance_copyleft_threshold),
            actual_value=float(copyleft_count),
            contribution="copyleft 许可证 → 0.2 分",
        ))
    dimensions.append(ScoreDimension(
        dimension_name="compliance",
        score=compliance_score,
        weight=profile.dimension_weights.get("compliance", 1.0),
        rule_traces=compliance_traces,
        explanation=f"检测到 {copyleft_count} 个 copyleft 许可证" if copyleft_count else "许可证合规",
    ))

    # ── 维度 5：文本扫描 ──────────────────────────────────
    text_scan_units = [u for u in evidence_units if u.category in ("keyword", "regex")]
    ts_traces: list[RuleTrace] = []
    ts_score = 1.0
    total_hits = len(text_scan_units)
    if total_hits > profile.text_scan_high_threshold:
        ts_score = 0.2
        ts_traces.append(RuleTrace(
            rule_name="text_scan_high_density",
            rule_description=f"{total_hits} 个命中 > {profile.text_scan_high_threshold}",
            matched=True, threshold=float(profile.text_scan_high_threshold),
            actual_value=float(total_hits),
            contribution="高密度命中 → 0.2 分",
        ))
    elif total_hits > profile.text_scan_low_threshold:
        ts_score = 0.6
        ts_traces.append(RuleTrace(
            rule_name="text_scan_low_density",
            rule_description=f"{total_hits} 个命中 > {profile.text_scan_low_threshold}",
            matched=True, threshold=float(profile.text_scan_low_threshold),
            actual_value=float(total_hits),
            contribution="中等密度命中 → 0.6 分",
        ))
    dimensions.append(ScoreDimension(
        dimension_name="text_scan",
        score=ts_score,
        weight=profile.dimension_weights.get("text_scan", 1.0),
        rule_traces=ts_traces,
        explanation=f"检测到 {total_hits} 个关键词/正则命中" if total_hits else "文本扫描无命中",
    ))

    # ── 计算总分与决策 ────────────────────────────────────
    breakdown = ScoreBreakdown(dimensions=dimensions)
    breakdown.compute_min()
    breakdown.compute_weighted()

    min_score = breakdown.min_score

    # fail-closed：降级事件存在时，上抬风险
    from common.enums import FailurePolicy
    if profile.failure_policy == FailurePolicy.FAIL_CLOSED.value and degrade_events:
        # 降级步骤数越多，风险上抬越大
        penalty = min(0.3, len(degrade_events) * 0.1)
        min_score = max(0.0, min_score - penalty)
        logger.info("fail-closed 策略：降级惩罚 %.2f，调整后最低分 %.2f", penalty, min_score)

    # 阈值 → 决策
    if min_score <= profile.reject_below:
        decision = UnifiedDecision.REJECT
    elif min_score <= profile.quarantine_below:
        decision = UnifiedDecision.QUARANTINE
    elif min_score <= profile.review_below:
        decision = UnifiedDecision.REVIEW
    else:
        decision = UnifiedDecision.ALLOW

    # 可信等级
    trust_level = TrustLevel.FULL
    if any(e.is_mock for e in degrade_events):
        trust_level = TrustLevel.UNTRUSTED
    elif degrade_events:
        trust_level = TrustLevel.DEGRADED

    # 复核建议
    review_suggestions: list[str] = []
    for dim in dimensions:
        for trace in dim.rule_traces:
            if trace.matched:
                review_suggestions.append(trace.rule_description)

    # 降级摘要
    degrade_summary = ""
    if degrade_events:
        steps = [e.step_name for e in degrade_events]
        degrade_summary = f"以下步骤发生降级：{', '.join(steps)}"

    all_reasons = [t.rule_description for d in dimensions for t in d.rule_traces if t.matched]

    return PolicyResult(
        decision=decision,
        reasons=all_reasons,
        score_breakdown=breakdown,
        trust_level=trust_level,
        review_suggestions=review_suggestions,
        degrade_summary=degrade_summary,
        profile_name=profile.profile_name,
    )

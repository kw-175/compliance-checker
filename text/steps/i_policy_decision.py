# ──────────────────────────────────────────────────────────────
# 步骤 I – 策略决策 (Policy Decision)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   基于步骤 H 聚合的证据包进行最终的合规决策。
#   支持两种决策引擎：
#
# 1. OPA (Open Policy Agent) REST API（主路径）：
#    - 将证据包发送至 OPA 的 /v1/data/compliance/decision 端点
#    - OPA 使用 Rego 策略文件（policies/compliance.rego）进行评估
#    - 返回结构化的决策结果
#
# 2. 本地 Python 规则引擎（Fallback）：
#    - 五维评分体系：
#      · secrets（密钥泄露）: 有泄露=0, 无=1
#      · safety（内容安全）: unsafe=0, controversial=0.5, safe=1
#      · privacy（隐私）: PII>5=0.3, PII>0=0.7, 无=1
#      · compliance（许可证）: copyleft=0.2, 正常=1
#      · text_scan（文本扫描）: 命中>20=0.2, 命中>5=0.6, 正常=1
#    - 决策映射：min_score<=0→REJECT, <=0.3→QUARANTINE,
#                            <=0.6→REVIEW, >0.6→ALLOW
#
# 在流水线中的位置：
#   H(证据聚合) → I(本步骤) → J(血缘审计)
#
# 输出产物：
#   decision.json
# ──────────────────────────────────────────────────────────────

"""
步骤 I – 策略决策。

主路径：OPA REST API
Fallback：本地 Python 规则引擎（五维评分）

输出 → decision.json
"""

from __future__ import annotations

import logging
from typing import Any

from text.config.settings import Settings
from text.models.schemas import (
    Decision,
    DocumentDecision,
    DocumentEvidence,
    EvidenceBundle,
    PolicyDecision,
    SafetyLevel,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# OPA REST API 调用
# ────────────────────────────────────────────────────────────

def _query_opa(
    evidence_bundle: EvidenceBundle,
    settings: Settings,
) -> PolicyDecision | None:
    """
    将证据包发送至 OPA 进行策略评估。

    构造 OPA 标准输入格式（input.documents），POST 到 OPA REST API，
    解析响应中的决策结果。

    Args:
        evidence_bundle: 证据包
        settings: 配置对象（包含 OPA URL 和策略路径）

    Returns:
        PolicyDecision 或 None（OPA 不可用时）
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx 未安装；无法调用 OPA")
        return None

    # 构造 OPA API URL
    url = f"{settings.opa_url}/{settings.opa_policy_path}"

    # 构造 OPA 输入 payload
    # OPA 要求输入放在 "input" 字段中
    payload = {
        "input": {
            "pipeline_run_id": evidence_bundle.pipeline_run_id,
            "summary": evidence_bundle.summary,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "source_id": doc.source_id,
                    "is_duplicate": doc.is_duplicate,
                    "secret_count": len(doc.secret_hits),
                    "compliance_count": len(doc.compliance_hits),
                    "keyword_count": len(doc.keyword_hits),
                    "regex_count": len(doc.regex_hits),
                    "pii_count": doc.privacy.pii_count if doc.privacy else 0,
                    "safety_level": doc.safety.safety_level.value if doc.safety else "safe",
                    "harm_categories": doc.safety.harm_categories if doc.safety else [],
                }
                for doc in evidence_bundle.documents
            ],
        }
    }

    try:
        resp = httpx.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json().get("result", {})

        # 解析 OPA 响应为 PolicyDecision
        doc_decisions = []
        for dd in result.get("document_decisions", []):
            doc_decisions.append(
                DocumentDecision(
                    doc_id=dd.get("doc_id", ""),
                    decision=Decision(dd.get("decision", "review")),
                    reasons=dd.get("reasons", []),
                    scores=dd.get("scores", {}),
                )
            )

        return PolicyDecision(
            pipeline_run_id=evidence_bundle.pipeline_run_id,
            overall_decision=Decision(result.get("overall_decision", "review")),
            document_decisions=doc_decisions,
        )
    except Exception as e:
        logger.warning("OPA 查询失败: %s", e)
        return None


# ────────────────────────────────────────────────────────────
# 本地规则引擎 fallback
# 五维评分体系 + 阈值决策
# ────────────────────────────────────────────────────────────

def _evaluate_document(doc: DocumentEvidence) -> DocumentDecision:
    """
    使用本地规则对单个文档的证据进行评估。

    五维评分：
    - secrets:    有密钥泄露=0, 无=1
    - safety:     UNSAFE=0, CONTROVERSIAL=0.5, SAFE=1
    - privacy:    PII>5=0.3, PII>0=0.7, 无=1
    - compliance: copyleft 许可证=0.2, 正常=1
    - text_scan:  命中>20=0.2, 命中>5=0.6, 正常=1

    决策逻辑（取最低分）：
    - min ≤ 0.0 → REJECT（拒绝）
    - min ≤ 0.3 → QUARANTINE（隔离）
    - min ≤ 0.6 → REVIEW（人工审核）
    - min > 0.6 → ALLOW（允许）

    Args:
        doc: 单文档证据

    Returns:
        DocumentDecision 决策对象
    """
    reasons: list[str] = []
    scores: dict[str, float] = {}

    # ── 维度 1：密钥泄露 ─────────────────────────────────
    secret_count = len(doc.secret_hits)
    if secret_count > 0:
        reasons.append(f"发现 {secret_count} 个泄露密钥")
        scores["secrets"] = 0.0  # 有泄露直接 0 分 → REJECT
    else:
        scores["secrets"] = 1.0

    # ── 维度 2：内容安全 ─────────────────────────────────
    safety_score = 1.0
    if doc.safety:
        if doc.safety.safety_level == SafetyLevel.UNSAFE:
            reasons.append(f"内容被分类为 UNSAFE ({doc.safety.harm_categories})")
            safety_score = 0.0
        elif doc.safety.safety_level == SafetyLevel.CONTROVERSIAL:
            reasons.append(f"内容被分类为 CONTROVERSIAL ({doc.safety.harm_categories})")
            safety_score = 0.5
    scores["safety"] = safety_score

    # ── 维度 3：隐私 (PII) ───────────────────────────────
    pii_score = 1.0
    if doc.privacy and doc.privacy.pii_count > 5:
        reasons.append(f"PII 密度过高: {doc.privacy.pii_count} 个实体")
        pii_score = 0.3
    elif doc.privacy and doc.privacy.pii_count > 0:
        pii_score = 0.7
    scores["privacy"] = pii_score

    # ── 维度 4：许可证合规 ───────────────────────────────
    compliance_score = 1.0
    copyleft_count = 0
    for ch in doc.compliance_hits:
        for lic in ch.licenses:
            expr_lower = lic.license_expression.lower()
            # 检测 copyleft 风格许可证（GPL、AGPL 等）
            if any(k in expr_lower for k in ["gpl", "agpl", "copyleft"]):
                copyleft_count += 1
    if copyleft_count > 0:
        reasons.append(f"发现 {copyleft_count} 个 copyleft 许可证")
        compliance_score = 0.2
    scores["compliance"] = compliance_score

    # ── 维度 5：文本扫描命中 ─────────────────────────────
    text_scan_score = 1.0
    total_text_hits = len(doc.keyword_hits) + len(doc.regex_hits)
    if total_text_hits > 20:
        reasons.append(f"{total_text_hits} 个关键词/正则命中（高密度）")
        text_scan_score = 0.2
    elif total_text_hits > 5:
        reasons.append(f"{total_text_hits} 个关键词/正则命中")
        text_scan_score = 0.6
    scores["text_scan"] = text_scan_score

    # ── 决策逻辑：取所有维度的最低分 ────────────────────
    min_score = min(scores.values()) if scores else 1.0

    if min_score <= 0.0:
        decision = Decision.REJECT       # 存在严重风险
    elif min_score <= 0.3:
        decision = Decision.QUARANTINE   # 需要隔离审查
    elif min_score <= 0.6:
        decision = Decision.REVIEW       # 需要人工审核
    else:
        decision = Decision.ALLOW        # 合规通过

    return DocumentDecision(
        doc_id=doc.doc_id,
        decision=decision,
        reasons=reasons,
        scores=scores,
    )


def _local_evaluate(evidence_bundle: EvidenceBundle) -> PolicyDecision:
    """
    使用本地规则引擎评估所有文档。

    总体决策取所有文档中最严格（最差）的决策。
    优先级：REJECT > QUARANTINE > REVIEW > ALLOW

    Args:
        evidence_bundle: 证据包

    Returns:
        PolicyDecision 包含总体决策和各文档决策
    """
    doc_decisions = [_evaluate_document(doc) for doc in evidence_bundle.documents]

    # 总体决策 = 所有文档中最严格的决策
    priority = {
        Decision.REJECT: 0,      # 最严格
        Decision.QUARANTINE: 1,
        Decision.REVIEW: 2,
        Decision.ALLOW: 3,       # 最宽松
    }
    if doc_decisions:
        worst = min(doc_decisions, key=lambda d: priority[d.decision])
        overall = worst.decision
    else:
        overall = Decision.ALLOW  # 无文档时默认允许

    return PolicyDecision(
        pipeline_run_id=evidence_bundle.pipeline_run_id,
        overall_decision=overall,
        document_decisions=doc_decisions,
    )


# ────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────

def run(
    evidence_bundle: EvidenceBundle,
    settings: Settings | None = None,
) -> PolicyDecision:
    """
    执行策略决策。

    优先使用 OPA REST API，若不可用或 OPA 被禁用，
    则回退到本地 Python 规则引擎。

    Args:
        evidence_bundle: 证据包（步骤 H 输出）
        settings: 配置对象（可选）

    Returns:
        PolicyDecision 包含最终的合规决策
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # 尝试 OPA
    if settings.opa_enabled:
        opa_result = _query_opa(evidence_bundle, settings)
        if opa_result is not None:
            logger.info(
                "OPA 决策: %s（%d 个文档决策）",
                opa_result.overall_decision.value,
                len(opa_result.document_decisions),
            )
            return opa_result
        logger.info("OPA 不可用；回退到本地规则引擎")

    # Fallback 到本地规则引擎
    result = _local_evaluate(evidence_bundle)

    # 统计各决策的分布
    decision_counts: dict[str, int] = {}
    for dd in result.document_decisions:
        decision_counts[dd.decision.value] = decision_counts.get(dd.decision.value, 0) + 1

    logger.info(
        "策略决策（本地引擎）: 总体=%s, 分布=%s",
        result.overall_decision.value, decision_counts,
    )
    return result

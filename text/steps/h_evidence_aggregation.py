# ──────────────────────────────────────────────────────────────
# 步骤 H – 证据聚合 (Evidence Aggregation)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   汇总步骤 B2（密钥+许可证）、D（去重）、E1（关键词+正则）、
#   F（隐私）、G（安全）的所有检测结果，按文档维度聚合为
#   统一的 EvidenceBundle 结构。
#
# 聚合策略：
#   - 密钥和合规记录按 source_id 关联（因为 B2 步骤操作的是原始来源）
#   - 关键词、正则、隐私、安全记录按 doc_id 关联
#   - 生成统计摘要（总文档数、各类检测数量等）
#
# 在流水线中的位置：
#   B2a/B2b + E1a/E1b + F + G → H(本步骤) → I(策略决策)
#
# 输出产物：
#   evidence_bundle.json
# ──────────────────────────────────────────────────────────────

"""
步骤 H – 证据聚合。

按文档维度汇总 B2/D/E1/F/G 各步骤的检测结果为 EvidenceBundle。
输出 → evidence_bundle.json
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from text.models.schemas import (
    ComplianceHit,
    DedupDocument,
    DocumentEvidence,
    EvidenceBundle,
    KeywordHit,
    PrivacyResult,
    RegexHit,
    SafetyResult,
    SecretHit,
)

logger = logging.getLogger(__name__)


def run(
    dedup_docs: list[DedupDocument],
    secret_hits: list[SecretHit],
    compliance_hits: list[ComplianceHit],
    keyword_hits: list[KeywordHit],
    regex_hits: list[RegexHit],
    privacy_results: list[PrivacyResult],
    safety_results: list[SafetyResult],
    pipeline_run_id: str = "",
) -> EvidenceBundle:
    """
    聚合所有检测结果为统一的证据包。

    将各步骤的检测结果按文档维度进行关联和汇总，
    生成包含全局统计信息的 EvidenceBundle。

    关联方式：
    - secret_hits / compliance_hits 通过 source_id 关联（来源级别）
    - keyword_hits / regex_hits 通过 doc_id 关联（文档级别）
    - privacy_results / safety_results 通过 doc_id 关联（文档级别）

    Args:
        dedup_docs: 去重后的文档列表（步骤 D 输出）
        secret_hits: 密钥泄露记录（步骤 B2a 输出）
        compliance_hits: 许可证合规记录（步骤 B2b 输出）
        keyword_hits: 关键词命中记录（步骤 E1a 输出）
        regex_hits: 正则匹配记录（步骤 E1b 输出）
        privacy_results: 隐私检测结果（步骤 F 输出）
        safety_results: 安全审核结果（步骤 G 输出）
        pipeline_run_id: 流水线运行 ID

    Returns:
        EvidenceBundle 证据包
    """
    # ── 构建查找索引 ─────────────────────────────────────
    # 按 source_id 索引密钥和合规记录（来源级别）
    secrets_by_source: dict[str, list[SecretHit]] = defaultdict(list)
    for h in secret_hits:
        secrets_by_source[h.source_id].append(h)

    compliance_by_source: dict[str, list[ComplianceHit]] = defaultdict(list)
    for h in compliance_hits:
        compliance_by_source[h.source_id].append(h)

    # 按 doc_id 索引关键词、正则、隐私和安全记录（文档级别）
    kw_by_doc: dict[str, list[KeywordHit]] = defaultdict(list)
    for h in keyword_hits:
        kw_by_doc[h.doc_id].append(h)

    regex_by_doc: dict[str, list[RegexHit]] = defaultdict(list)
    for h in regex_hits:
        regex_by_doc[h.doc_id].append(h)

    privacy_by_doc: dict[str, PrivacyResult] = {r.doc_id: r for r in privacy_results}
    safety_by_doc: dict[str, SafetyResult] = {r.doc_id: r for r in safety_results}

    # ── 构建每个文档的证据记录 ───────────────────────────
    doc_evidences: list[DocumentEvidence] = []
    for doc in dedup_docs:
        evidence = DocumentEvidence(
            doc_id=doc.doc_id,
            source_id=doc.source_id,
            # 通过 source_id 关联来源级别的检测结果
            secret_hits=secrets_by_source.get(doc.source_id, []),
            compliance_hits=compliance_by_source.get(doc.source_id, []),
            is_duplicate=doc.is_duplicate,
            # 通过 doc_id 关联文档级别的检测结果
            keyword_hits=kw_by_doc.get(doc.doc_id, []),
            regex_hits=regex_by_doc.get(doc.doc_id, []),
            privacy=privacy_by_doc.get(doc.doc_id),
            safety=safety_by_doc.get(doc.doc_id),
        )
        doc_evidences.append(evidence)

    # ── 计算统计摘要 ─────────────────────────────────────
    summary = {
        "total_documents": len(dedup_docs),                                    # 总文档数
        "unique_documents": sum(1 for d in dedup_docs if not d.is_duplicate),  # 唯一文档数
        "duplicate_documents": sum(1 for d in dedup_docs if d.is_duplicate),   # 重复文档数
        "total_secret_hits": len(secret_hits),                                 # 密钥泄露总数
        "total_compliance_hits": len(compliance_hits),                          # 合规问题总数
        "total_keyword_hits": len(keyword_hits),                               # 关键词命中总数
        "total_regex_hits": len(regex_hits),                                   # 正则匹配总数
        "total_pii_entities": sum(r.pii_count for r in privacy_results),       # PII 实体总数
        "unsafe_documents": sum(                                                # 不安全文档数
            1 for r in safety_results if r.safety_level.value == "unsafe"
        ),
        "controversial_documents": sum(                                         # 争议文档数
            1 for r in safety_results if r.safety_level.value == "controversial"
        ),
    }

    # ── 构建最终证据包 ───────────────────────────────────
    bundle = EvidenceBundle(
        pipeline_run_id=pipeline_run_id,
        documents=doc_evidences,
        summary=summary,
    )

    logger.info(
        "证据聚合完成: %d 文档, %d 密钥, %d 合规, %d 关键词, "
        "%d 正则, %d PII, %d 不安全, %d 争议",
        summary["total_documents"],
        summary["total_secret_hits"],
        summary["total_compliance_hits"],
        summary["total_keyword_hits"],
        summary["total_regex_hits"],
        summary["total_pii_entities"],
        summary["unsafe_documents"],
        summary["controversial_documents"],
    )
    return bundle

# ──────────────────────────────────────────────────────────────
# 步骤 D – 早期去重 (Early Deduplication)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   对步骤 C 输出的清洗文档进行去重处理，分两个阶段：
#   1. 精确去重（Phase 1）：基于文本内容的 SHA-256 哈希
#   2. 近似去重（Phase 2）：基于 MinHash LSH 算法检测相似文档
#
# 设计决策：
#   - 计划文档中的 Duplodocus (Rust CLI) 当前未集成
#   - 使用 Python datasketch 库的 MinHash LSH 作为 fallback 实现
#   - datasketch 未安装时仅执行精确去重
#
# 算法说明：
#   - 精确去重：对文本内容计算 SHA-256，相同哈希的文档视为完全重复
#   - 近似去重：将文本切割为 k-shingle（字符级），生成 MinHash 签名，
#     通过 LSH 索引快速查找相似文档，再用 Jaccard 相似度确认
#
# 在流水线中的位置：
#   C(文本提取) → D(本步骤) → E1(规则扫描) / F(隐私检测)
#
# 输出产物：
#   deduped_documents.jsonl（标记了去重状态的文档）
#   dedup_map.jsonl（重复文档映射关系）
# ──────────────────────────────────────────────────────────────

"""
步骤 D – 早期去重。

Phase 1: SHA-256 精确去重
Phase 2: datasketch MinHash LSH 近似去重

输出 → deduped_documents.jsonl + dedup_map.jsonl
"""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import CleanedDocument, DedupDocument, DedupMapEntry

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Phase 1: 精确去重（基于哈希）
# 最快速的去重方式，通过比较文本内容的 SHA-256 哈希值
# 时间复杂度 O(n)，空间复杂度 O(n)
# ────────────────────────────────────────────────────────────

def _exact_dedup(
    documents: list[CleanedDocument],
) -> tuple[list[CleanedDocument], list[DedupMapEntry]]:
    """
    使用 SHA-256 哈希进行精确去重。

    遍历所有文档，对文本内容计算哈希值。若哈希已出现过，
    则标记为重复并记录映射关系。

    Args:
        documents: 清洗后的文档列表

    Returns:
        (去重后的唯一文档列表, 重复映射记录列表)
    """
    seen: dict[str, str] = {}  # 哈希值 → 首次出现的 doc_id
    unique: list[CleanedDocument] = []
    dedup_map: list[DedupMapEntry] = []

    for doc in documents:
        # 对文本内容计算 SHA-256
        text_hash = hashlib.sha256(doc.text.encode("utf-8")).hexdigest()

        if text_hash in seen:
            # 已存在相同内容 → 标记为重复
            dedup_map.append(
                DedupMapEntry(
                    doc_id=doc.doc_id,
                    duplicate_of=seen[text_hash],
                    jaccard_similarity=1.0,  # 完全相同
                )
            )
            logger.debug("精确重复: %s == %s", doc.doc_id, seen[text_hash])
        else:
            # 首次出现 → 保留
            seen[text_hash] = doc.doc_id
            unique.append(doc)

    return unique, dedup_map


# ────────────────────────────────────────────────────────────
# Phase 2: 近似去重（基于 MinHash LSH）
# 使用 datasketch 库实现，能检测内容相似但不完全相同的文档
# 时间复杂度约 O(n)（LSH 近似搜索），空间复杂度 O(n)
# ────────────────────────────────────────────────────────────

def _shingle(text: str, k: int = 5) -> set[str]:
    """
    生成字符级 k-shingle 集合。

    将文本切割为长度为 k 的连续字符子串集合，
    作为 MinHash 算法的输入。k 值越大，对小的文本差异越敏感。

    Args:
        text: 输入文本
        k: shingle 长度（默认 5）

    Returns:
        k-shingle 字符串集合
    """
    return {text[i : i + k] for i in range(max(len(text) - k + 1, 1))}


def _minhash_dedup(
    documents: list[CleanedDocument],
    threshold: float = 0.8,
    num_perm: int = 128,
) -> tuple[list[DedupDocument], list[DedupMapEntry]]:
    """
    使用 MinHash LSH 进行近似去重。

    算法流程：
    1. 为每个文档生成 k-shingle 集合
    2. 计算 MinHash 签名
    3. 将签名插入 LSH 索引
    4. 查询每个文档的近似近邻
    5. 计算精确 Jaccard 相似度确认

    若 datasketch 未安装，则跳过近似去重，所有文档标记为非重复。

    Args:
        documents: 精确去重后的唯一文档列表
        threshold: Jaccard 相似度阈值（默认 0.8，即 80% 相似）
        num_perm: MinHash 排列数（越大越精确，但越慢）

    Returns:
        (去重后的文档列表（含 is_duplicate 标记）, 重复映射记录列表)
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        logger.warning(
            "datasketch 未安装；跳过近似去重。安装: pip install datasketch"
        )
        # datasketch 不可用时，所有文档标记为非重复直接返回
        return [
            DedupDocument(
                doc_id=d.doc_id,
                source_id=d.source_id,
                text=d.text,
                is_duplicate=False,
            )
            for d in documents
        ], []

    # 初始化 LSH 索引
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: dict[str, MinHash] = {}
    dedup_map: list[DedupMapEntry] = []
    results: list[DedupDocument] = []

    # 第 1 步：为每个文档生成 MinHash 签名
    for doc in documents:
        m = MinHash(num_perm=num_perm)
        for shingle in _shingle(doc.text):
            m.update(shingle.encode("utf-8"))
        minhashes[doc.doc_id] = m

    # 第 2 步：逐个插入 LSH 并查询是否有近似重复
    seen_duplicates: set[str] = set()  # 已标记为重复的文档集合
    for doc in documents:
        m = minhashes[doc.doc_id]
        # 查询 LSH 索引中的近似近邻
        query_result = lsh.query(m)

        duplicate_of: Optional[str] = None
        for candidate_id in query_result:
            if candidate_id != doc.doc_id and candidate_id not in seen_duplicates:
                # 计算精确 Jaccard 相似度进行确认
                jaccard = m.jaccard(minhashes[candidate_id])
                if jaccard >= threshold:
                    duplicate_of = candidate_id
                    dedup_map.append(
                        DedupMapEntry(
                            doc_id=doc.doc_id,
                            duplicate_of=candidate_id,
                            jaccard_similarity=round(jaccard, 4),
                        )
                    )
                    seen_duplicates.add(doc.doc_id)
                    break

        is_dup = duplicate_of is not None
        results.append(
            DedupDocument(
                doc_id=doc.doc_id,
                source_id=doc.source_id,
                text=doc.text,
                is_duplicate=is_dup,
                duplicate_of=duplicate_of,
            )
        )

        # 仅将非重复文档的签名插入 LSH 索引
        if not is_dup:
            try:
                lsh.insert(doc.doc_id, m)
            except ValueError:
                pass  # 已存在则跳过

    return results, dedup_map


# ────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[CleanedDocument],
    settings: Settings | None = None,
) -> tuple[list[DedupDocument], list[DedupMapEntry]]:
    """
    执行完整的去重流水线。

    两阶段处理：
    1. Phase 1: 精确去重（SHA-256 哈希比对）—— 去除完全相同的文档
    2. Phase 2: 近似去重（MinHash LSH）—— 去除高度相似的文档

    Args:
        documents: 清洗后的文档列表（步骤 C 输出）
        settings: 配置对象（可选）

    Returns:
        (去重后的文档列表, 所有重复映射记录的合并列表)
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    logger.info("开始去重处理: 共 %d 个文档", len(documents))

    # Phase 1: 精确去重
    unique_docs, exact_map = _exact_dedup(documents)
    logger.info(
        "精确去重: %d → %d 个唯一文档（移除 %d 个重复）",
        len(documents), len(unique_docs), len(exact_map),
    )

    # Phase 2: 对剩余文档进行近似去重
    dedup_results, near_map = _minhash_dedup(
        unique_docs,
        threshold=settings.dedup_threshold,
        num_perm=settings.dedup_num_perm,
    )

    # 合并两阶段的映射记录
    all_map = exact_map + near_map

    # 统计总重复数
    # 注意：exact_map 中的文档已在 Phase 1 被移除，不在 dedup_results 中，
    # 所以 Phase 2 中 is_duplicate=True 的计数 + exact_map 长度 = 总重复数
    near_dup_count = sum(1 for d in dedup_results if d.is_duplicate)
    total_dups = len(exact_map) + near_dup_count
    logger.info(
        "去重完成: 共 %d 个重复（精确 %d, 近似 %d）",
        total_dups, len(exact_map), near_dup_count,
    )

    return dedup_results, all_map

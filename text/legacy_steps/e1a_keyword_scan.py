# ──────────────────────────────────────────────────────────────
# 步骤 E1a – FlashText2 关键词扫描 (Keyword Scan)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   使用 FlashText2 的 KeywordProcessor 对去重后的文档语料库
#   执行高速多关键词匹配，检测敏感/违规关键词。
#
# FlashText2 原理：
#   基于 Aho-Corasick 自动机的多模式匹配算法，
#   时间复杂度 O(n+m)（n=文本长度，m=匹配数），
#   远快于逐个关键词搜索的 O(n*k)（k=关键词数）。
#
# Fallback 策略：
#   - FlashText2 未安装 → 使用 str.find() 逐词搜索（性能较低但功能等价）
#
# 在流水线中的位置：
#   D(去重) → E1a(本步骤，与 E1b 并行) → H(证据聚合)
#
# 输出产物：
#   keyword_hits.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 E1a – FlashText2 关键词扫描。

使用 FlashText2 KeywordProcessor 进行高速多关键词匹配。
输出 → keyword_hits.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path

from text.config.settings import Settings
from text.models.schemas import DedupDocument, KeywordHit

logger = logging.getLogger(__name__)

# 上下文窗口大小：在匹配位置前后各取 60 个字符作为上下文
_CONTEXT_WINDOW = 60


def _load_keywords(keywords_file: Path) -> list[str]:
    """
    从文本文件加载关键词列表。

    文件格式要求：
    - 每行一个关键词
    - 以 # 开头的行视为注释，自动跳过
    - 空行自动跳过

    Args:
        keywords_file: 关键词文件路径

    Returns:
        关键词字符串列表
    """
    if not keywords_file.exists():
        logger.warning("关键词文件不存在: %s", keywords_file)
        return []

    lines: list[str] = []
    for line in keywords_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # 跳过空行和注释行
        if line and not line.startswith("#"):
            lines.append(line)

    logger.info("已从 %s 加载 %d 个关键词", keywords_file, len(lines))
    return lines


def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
    """
    提取匹配位置周围的上下文片段。

    在匹配位置前后各取 window 个字符，并在截断处添加 "..." 标记。

    Args:
        text: 完整文本
        start: 匹配起始位置
        end: 匹配结束位置
        window: 上下文窗口大小

    Returns:
        带有省略号标记的上下文字符串
    """
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[KeywordHit]:
    """
    执行 FlashText2 关键词扫描。

    对每个非重复文档的文本进行关键词匹配。优先使用 FlashText2
    的 KeywordProcessor（高性能），若不可用则回退到 str.find()。

    Args:
        documents: 去重后的文档列表（步骤 D 输出）
        settings: 配置对象（可选）

    Returns:
        KeywordHit 列表
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # 加载关键词列表
    keywords = _load_keywords(settings.keywords_file)
    if not keywords:
        logger.warning("无关键词可扫描 – 跳过步骤 E1a")
        return []

    # 尝试加载 FlashText2，失败则使用 fallback
    try:
        from flashtext2 import KeywordProcessor
        kp = KeywordProcessor(case_sensitive=False)  # 不区分大小写
        kp.add_keywords_from_list(keywords)
        use_flashtext = True
        logger.info("使用 FlashText2 进行关键词扫描")
    except ImportError:
        logger.warning(
            "flashtext2 未安装；回退到基本字符串搜索。"
            "安装: pip install flashtext2"
        )
        use_flashtext = False

    all_hits: list[KeywordHit] = []

    for doc in documents:
        # 跳过重复文档
        if doc.is_duplicate:
            continue

        if use_flashtext:
            # FlashText2 模式：返回 (关键词, 起始位置, 结束位置) 三元组列表
            matches = kp.extract_keywords(doc.text, span_info=True)
            for keyword, start, end in matches:
                all_hits.append(
                    KeywordHit(
                        doc_id=doc.doc_id,
                        keyword=keyword,
                        start_pos=start,
                        end_pos=end,
                        context=_extract_context(doc.text, start, end),
                    )
                )
        else:
            # Fallback 模式：逐个关键词进行大小写不敏感搜索
            text_lower = doc.text.lower()
            for kw in keywords:
                kw_lower = kw.lower()
                start = 0
                while True:
                    # 查找下一个匹配位置
                    idx = text_lower.find(kw_lower, start)
                    if idx == -1:
                        break
                    end = idx + len(kw_lower)
                    all_hits.append(
                        KeywordHit(
                            doc_id=doc.doc_id,
                            keyword=kw,
                            start_pos=idx,
                            end_pos=end,
                            context=_extract_context(doc.text, idx, end),
                        )
                    )
                    # 从当前匹配结束位置继续搜索
                    start = end

    logger.info(
        "关键词扫描完成: 共 %d 个命中，涉及 %d 个文档",
        len(all_hits), len(documents),
    )
    return all_hits

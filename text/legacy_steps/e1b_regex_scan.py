# ──────────────────────────────────────────────────────────────
# 步骤 E1b – Hyperscan / 正则扫描 (Regex Scan)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   使用正则表达式对去重后的文档进行敏感模式匹配。
#   支持邮箱、SSN、信用卡号、API Key 等多种模式。
#
# 双后端设计：
#   - 主后端：python-hyperscan（Intel 开发的高性能正则引擎）
#   - 回退后端：Python 标准库 re 模块
#
# Hyperscan 特点：
#   - 基于 DFA 引擎，能同时匹配数千个正则模式
#   - 吞吐量远超 Python re（10x-100x）
#   - 但需要系统安装 Vectorscan/Hyperscan C 库
#   - Windows 上编译困难，开发时通常使用 re fallback
#
# 注意事项：
#   Hyperscan 返回的 from_/to 是字节偏移而非字符偏移。
#   对于 UTF-8 多字节文本（如中文），需要转换为字符偏移。
#
# 在流水线中的位置：
#   D(去重) → E1b(本步骤，与 E1a 并行) → H(证据聚合)
#
# 输出产物：
#   regex_hits.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 E1b – Hyperscan / 正则扫描。

主后端：python-hyperscan（高性能多模式正则匹配）
回退后端：Python stdlib re 模块

输出 → regex_hits.jsonl
"""

from __future__ import annotations

import logging
import re as re_stdlib
from pathlib import Path
from typing import Any

import yaml

from text.config.settings import Settings
from text.models.schemas import DedupDocument, RegexHit

logger = logging.getLogger(__name__)

# 上下文窗口大小（字符数）
_CONTEXT_WINDOW = 60


def _load_patterns(patterns_file: Path) -> dict[str, str]:
    """
    从 YAML 文件加载命名正则模式。

    YAML 格式：模式名称 → 正则表达式
    示例：
      email_address: "[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+"
      us_ssn: "\\b\\d{3}-\\d{2}-\\d{4}\\b"

    Args:
        patterns_file: YAML 模式文件路径

    Returns:
        {模式名称: 正则表达式} 字典
    """
    if not patterns_file.exists():
        logger.warning("模式文件不存在: %s", patterns_file)
        return {}
    try:
        with open(patterns_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        logger.info("已从 %s 加载 %d 个正则模式", patterns_file, len(data))
        return {str(k): str(v) for k, v in data.items()}
    except Exception as e:
        logger.error("加载模式文件失败: %s", e)
        return {}


def _extract_context(text: str, start: int, end: int, window: int = _CONTEXT_WINDOW) -> str:
    """
    提取匹配位置周围的上下文片段。

    注意：start 和 end 必须是字符偏移（非字节偏移）。

    Args:
        text: 完整文本
        start: 匹配起始字符位置
        end: 匹配结束字符位置
        window: 上下文窗口大小

    Returns:
        带省略号的上下文字符串
    """
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    return prefix + text[ctx_start:ctx_end] + suffix


# ────────────────────────────────────────────────────────────
# Hyperscan 后端
# 使用 Intel Hyperscan 引擎进行高性能多模式匹配
# ────────────────────────────────────────────────────────────

def _scan_with_hyperscan(
    documents: list[DedupDocument],
    patterns: dict[str, str],
) -> list[RegexHit]:
    """
    使用 python-hyperscan 进行多模式正则扫描。

    Hyperscan 将所有模式编译为一个 DFA 数据库，
    然后在单次扫描中同时匹配所有模式，效率极高。

    注意：Hyperscan 的回调返回字节偏移，需要转换为字符偏移。

    Args:
        documents: 去重后的文档列表
        patterns: {模式名称: 正则表达式} 字典

    Returns:
        RegexHit 列表
    """
    import hyperscan  # type: ignore

    pattern_names = list(patterns.keys())
    pattern_exprs = [p.encode("utf-8") for p in patterns.values()]
    pattern_ids = list(range(len(pattern_names)))
    # HS_FLAG_DOTALL: 点号匹配换行符
    # HS_FLAG_SINGLEMATCH: 每个模式仅报告第一次匹配
    pattern_flags = [hyperscan.HS_FLAG_DOTALL | hyperscan.HS_FLAG_SINGLEMATCH] * len(pattern_exprs)

    # 编译 Hyperscan 数据库
    db = hyperscan.Database()
    db.compile(
        expressions=pattern_exprs,
        ids=pattern_ids,
        flags=pattern_flags,
    )

    all_hits: list[RegexHit] = []

    for doc in documents:
        if doc.is_duplicate:
            continue

        doc_hits: list[dict[str, Any]] = []

        # Hyperscan 匹配回调函数
        def on_match(id_: int, from_: int, to: int, flags: int, context: Any = None) -> bool:
            doc_hits.append({"id": id_, "from": from_, "to": to})
            return False  # 返回 False 继续扫描

        # 修正 Bug 6：Hyperscan 返回的 from_/to 是字节偏移
        text_bytes = doc.text.encode("utf-8")
        db.scan(text_bytes, match_event_handler=on_match)

        for hit in doc_hits:
            pid = hit["id"]
            name = pattern_names[pid]
            byte_start = hit["from"]
            byte_end = hit["to"]
            # 提取匹配到的字节片段并解码
            matched = text_bytes[byte_start:byte_end].decode("utf-8", errors="replace")
            # 将字节偏移转换为字符偏移，以便 _extract_context 正确截取
            char_start = len(text_bytes[:byte_start].decode("utf-8", errors="replace"))
            char_end = len(text_bytes[:byte_end].decode("utf-8", errors="replace"))
            all_hits.append(
                RegexHit(
                    doc_id=doc.doc_id,
                    pattern_name=name,
                    pattern=patterns[name],
                    matched_text=matched[:200],  # 截断到 200 字符
                    start_pos=char_start,
                    end_pos=char_end,
                    context=_extract_context(doc.text, char_start, char_end),
                )
            )

    return all_hits


# ────────────────────────────────────────────────────────────
# Python re fallback 后端
# 使用 Python 标准库 re 模块作为降级方案
# ────────────────────────────────────────────────────────────

def _scan_with_re(
    documents: list[DedupDocument],
    patterns: dict[str, str],
) -> list[RegexHit]:
    """
    使用 Python 标准库 re 进行多模式正则扫描。

    逐个编译正则表达式，逐文档逐模式进行匹配。
    性能不如 Hyperscan，但无需额外依赖。

    Args:
        documents: 去重后的文档列表
        patterns: {模式名称: 正则表达式} 字典

    Returns:
        RegexHit 列表
    """
    # 预编译所有正则表达式，跳过无效的模式
    compiled: list[tuple[str, re_stdlib.Pattern]] = []
    for name, expr in patterns.items():
        try:
            compiled.append((name, re_stdlib.compile(expr)))
        except re_stdlib.error as e:
            logger.warning("无效的正则模式 '%s': %s", name, e)

    all_hits: list[RegexHit] = []

    for doc in documents:
        if doc.is_duplicate:
            continue
        for name, regex in compiled:
            # finditer 返回所有非重叠匹配
            for m in regex.finditer(doc.text):
                all_hits.append(
                    RegexHit(
                        doc_id=doc.doc_id,
                        pattern_name=name,
                        pattern=regex.pattern,
                        matched_text=m.group()[:200],
                        start_pos=m.start(),
                        end_pos=m.end(),
                        context=_extract_context(doc.text, m.start(), m.end()),
                    )
                )
    return all_hits


# ────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[RegexHit]:
    """
    执行正则模式扫描。

    优先使用 Hyperscan 后端（高性能），若不可用则自动回退到
    Python re 后端。

    Args:
        documents: 去重后的文档列表（步骤 D 输出）
        settings: 配置对象（可选）

    Returns:
        RegexHit 列表
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # 加载正则模式
    patterns = _load_patterns(settings.patterns_file)
    if not patterns:
        logger.warning("无正则模式可加载 – 跳过步骤 E1b")
        return []

    # 尝试使用 Hyperscan 后端
    try:
        hits = _scan_with_hyperscan(documents, patterns)
        logger.info("Hyperscan 正则扫描完成: %d 个命中", len(hits))
        return hits
    except ImportError:
        logger.warning(
            "python-hyperscan 未安装；回退到标准库 re。"
            "安装: pip install hyperscan"
        )
    except Exception as e:
        logger.warning("Hyperscan 扫描失败 (%s)；回退到 re", e)

    # Fallback 到 Python re
    hits = _scan_with_re(documents, patterns)
    logger.info("正则扫描 (re fallback) 完成: %d 个命中", len(hits))
    return hits

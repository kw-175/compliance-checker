# ──────────────────────────────────────────────────────────────
# 步骤 C – 文本提取与预处理 (Text Extraction & Preprocessing)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   从步骤 B1 分类后的来源中提取纯文本内容，并执行清洗预处理。
#   根据来源类型选择不同的提取策略：
#     - web_text  → Trafilatura（HTML 正文提取）
#     - pdf_text  → PyMuPDF（PDF 文本提取）
#     - code/repo/package/binary/mixed → 直接文本读取
#
# 清洗步骤：
#   1. Unicode NFC 规范化
#   2. 替换不可见字符（零宽空格、BOM 等）
#   3. 压缩连续空白和空行
#   4. 检测文本语言
#
# Fallback 策略：
#   - Trafilatura 未安装 → 使用简单的 HTML 标签剥离
#   - PyMuPDF 未安装 → 跳过 PDF 提取
#   - langdetect 未安装 → 使用 CJK 字符比例启发式判断
#
# 在流水线中的位置：
#   B1(来源分类) → C(本步骤) → D(去重)
#
# 输出产物：
#   cleaned_documents.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 C – 文本提取与预处理。

使用 DataTrove/Trafilatura 提取 HTML，PyMuPDF 提取 PDF，
直读纯文本/代码。执行 Unicode 规范化和空白清洗。

输出 → cleaned_documents.jsonl
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import CleanedDocument, SourceProfile, SourceType

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 文本提取辅助函数
# 根据不同的来源类型使用不同的提取策略
# ────────────────────────────────────────────────────────────

def _extract_plain_text(file_path: str) -> Optional[str]:
    """
    直接读取纯文本/代码文件。

    适用于 code、repo、package、binary、mixed 类型的文件。
    使用 UTF-8 编码读取，遇到无法解码的字节使用替换字符。

    Args:
        file_path: 文件路径

    Returns:
        文件文本内容；读取失败时返回 None
    """
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("读取纯文本失败 %s: %s", file_path, e)
        return None


def _extract_html_text(file_path: str) -> Optional[str]:
    """
    使用 Trafilatura 从 HTML 中提取正文文本。

    Trafilatura 是一个专业的 Web 内容提取库，能智能识别网页正文，
    过滤导航栏、页脚、广告等无关内容。

    Fallback：若 Trafilatura 未安装或提取失败，回退到简单的 HTML 标签剥离。

    Args:
        file_path: HTML 文件路径

    Returns:
        提取的正文文本；失败时尝试 fallback
    """
    try:
        import trafilatura
        html = Path(file_path).read_text(encoding="utf-8", errors="replace")
        # include_comments=False: 排除网页注释
        # include_tables=True: 保留表格内容
        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        return text
    except ImportError:
        logger.warning(
            "trafilatura 未安装；回退到简单 HTML 标签剥离。"
            "安装: pip install trafilatura"
        )
        return _strip_html_basic(file_path)
    except Exception as e:
        logger.warning("Trafilatura 提取失败 %s: %s", file_path, e)
        return _strip_html_basic(file_path)


def _strip_html_basic(file_path: str) -> Optional[str]:
    """
    简单的 HTML 标签剥离 fallback。

    使用正则表达式移除所有 HTML 标签，仅保留文本内容。
    这是 Trafilatura 不可用时的降级方案，提取质量较低。

    Args:
        file_path: HTML 文件路径

    Returns:
        剥离标签后的文本；失败时返回 None
    """
    try:
        raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
        # 用正则移除所有 HTML 标签，替换为空格
        return re.sub(r"<[^>]+>", " ", raw)
    except Exception:
        return None


def _extract_pdf_text(file_path: str) -> Optional[str]:
    """
    使用 PyMuPDF (fitz) 从 PDF 中提取文本。

    逐页提取文本内容并拼接。PyMuPDF 支持大多数 PDF 格式，
    包括扫描 PDF（需 OCR 支持）。

    Fallback：PyMuPDF 未安装时跳过 PDF 提取。

    Args:
        file_path: PDF 文件路径

    Returns:
        提取的文本内容；失败时返回 None
    """
    try:
        import fitz  # PyMuPDF 的导入名称
        doc = fitz.open(file_path)
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n".join(pages)
    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) 未安装；跳过 PDF 提取 %s。"
            "安装: pip install PyMuPDF",
            file_path,
        )
        return None
    except Exception as e:
        logger.warning("PDF 提取失败 %s: %s", file_path, e)
        return None


# ────────────────────────────────────────────────────────────
# 文本清洗
# ────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """
    执行标准文本清洗流程。

    处理步骤：
    1. Unicode NFC 规范化 → 统一字符编码表示
    2. 替换常见不可见字符 → 消除排版干扰
    3. 压缩行内多余空白 → 规范化空格
    4. 压缩过多空行 → 保持可读性

    Args:
        raw: 原始文本

    Returns:
        清洗后的文本
    """
    # 第 1 步：Unicode NFC 规范化
    # 将组合字符序列转换为预组合形式（如 é 的两种表示统一为一种）
    text = unicodedata.normalize("NFC", raw)

    # 第 2 步：替换常见的不可见/特殊字符
    text = text.replace("\u00a0", " ")   # 不换行空格 → 普通空格
    text = text.replace("\u200b", "")    # 零宽空格 → 删除
    text = text.replace("\u200c", "")    # 零宽非连接符 → 删除
    text = text.replace("\u200d", "")    # 零宽连接符 → 删除
    text = text.replace("\ufeff", "")    # BOM (字节顺序标记) → 删除

    # 第 3 步：压缩行内连续空白字符（保留换行符）
    # [^\S\n]+ 匹配除换行符外的连续空白字符
    text = re.sub(r"[^\S\n]+", " ", text)

    # 第 4 步：压缩过多空行（3 个以上连续换行 → 2 个）
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _detect_language(text: str) -> str:
    """
    检测文本的语言。

    优先使用 langdetect 库进行检测，回退到基于 CJK 字符比例的启发式判断。

    Args:
        text: 待检测的文本（仅使用前 2000 字符）

    Returns:
        ISO 639-1 语言代码（如 "en"、"zh"）
    """
    try:
        from langdetect import detect
        return detect(text[:2000])
    except Exception:
        # 启发式判断：如果前 1000 个字符中 CJK 字符占比超过 30%，则判为中文
        cjk_count = sum(1 for c in text[:1000] if "\u4e00" <= c <= "\u9fff")
        if cjk_count / max(len(text[:1000]), 1) > 0.3:
            return "zh"
        return "en"  # 默认返回英文


# ────────────────────────────────────────────────────────────
# 提取策略分发器
# 根据 SourceType 映射到对应的提取函数
# ────────────────────────────────────────────────────────────

_EXTRACTOR_MAP = {
    SourceType.WEB_TEXT: _extract_html_text,    # HTML → Trafilatura
    SourceType.PDF_TEXT: _extract_pdf_text,      # PDF → PyMuPDF
    SourceType.CODE: _extract_plain_text,        # 代码 → 直读
    SourceType.REPO: _extract_plain_text,        # 仓库 → 直读
    SourceType.PACKAGE: _extract_plain_text,     # 包 → 直读
    SourceType.BINARY: _extract_plain_text,      # 二进制 → 直读（可能得到乱码）
    SourceType.MIXED: _extract_plain_text,       # 混合/纯文本 → 直读
}


def run(
    profiles: list[SourceProfile],
    settings: Settings | None = None,
) -> list[CleanedDocument]:
    """
    执行文本提取与清洗。

    根据每个来源的分类类型选择对应的提取策略，提取文本后
    执行清洗和语言检测。

    Args:
        profiles: 来源画像列表（步骤 B1 输出）
        settings: 配置对象（可选）

    Returns:
        CleanedDocument 列表，每条记录包含清洗后的文本
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    documents: list[CleanedDocument] = []

    for profile in profiles:
        # 根据来源类型选择提取函数（未知类型时默认使用纯文本提取）
        extractor = _EXTRACTOR_MAP.get(profile.source_type, _extract_plain_text)

        # 执行文本提取
        raw_text = extractor(profile.path)
        if raw_text is None or not raw_text.strip():
            logger.debug("来源 %s 未提取到文本", profile.source_id)
            continue

        # 执行文本清洗
        cleaned = _clean_text(raw_text)
        if not cleaned:
            continue

        # 检测语言
        lang = _detect_language(cleaned)

        # 构建 CleanedDocument
        doc = CleanedDocument(
            source_id=profile.source_id,
            text=cleaned,
            char_count=len(cleaned),
            language=lang,
            metadata={
                "source_type": profile.source_type.value,
                "original_path": profile.path,
            },
        )
        documents.append(doc)
        logger.debug(
            "提取文档 %s: 来源=%s, 字符数=%d, 语言=%s",
            doc.doc_id, profile.source_id, doc.char_count, lang,
        )

    logger.info(
        "文本提取完成: 从 %d 个来源中提取了 %d 个文档",
        len(profiles), len(documents),
    )
    return documents

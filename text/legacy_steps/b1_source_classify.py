# ──────────────────────────────────────────────────────────────
# 步骤 B1 – 来源分类 (Source Classification)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   读取步骤 A 输出的来源注册表，根据文件扩展名和 MIME 类型
#   将每个来源分类为以下七种类型之一：
#     code | repo | package | binary | web_text | pdf_text | mixed
#
# 分类策略（优先级从高到低）：
#   1. MIME 类型精确匹配（如 text/html → web_text）
#   2. 文件扩展名匹配（如 .py → code, .pdf → pdf_text）
#   3. MIME 前缀启发式（text/* → code）
#   4. 默认 → mixed
#
# 在流水线中的位置：
#   A(输入接入) → B1(本步骤) → B2(扫描) / C(文本提取)
#
# 输出产物：
#   source_profile.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 B1 – 来源分类。

读取来源注册表，根据 MIME 类型和扩展名将每个来源分类为
code / repo / package / binary / web_text / pdf_text / mixed。

输出 → source_profile.jsonl
"""

from __future__ import annotations

import logging
from pathlib import Path

from text.models.schemas import SourceProfile, SourceRecord, SourceType

logger = logging.getLogger(__name__)

# ─── 文件扩展名 → 来源类型映射表 ────────────────────────────

# 代码文件扩展名集合（涵盖主流编程语言、脚本、配置文件）
_CODE_EXTS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".lua", ".pl", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".r", ".R", ".m", ".sql", ".html", ".css", ".scss", ".sass",
    ".vue", ".svelte", ".yaml", ".yml", ".toml", ".json", ".xml",
    ".dockerfile", ".tf", ".hcl", ".proto", ".graphql", ".makefile",
}

# 软件包/压缩包扩展名集合
_PACKAGE_EXTS = {
    ".whl", ".tar.gz", ".tgz", ".egg", ".gem", ".jar", ".war",
    ".zip", ".rar", ".7z", ".deb", ".rpm", ".nupkg", ".apk",
}

# 二进制文件扩展名集合
_BINARY_EXTS = {
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".bin", ".dat",
    ".elf", ".msi", ".img", ".iso",
}

# PDF 文件扩展名
_PDF_EXTS = {".pdf"}

# 网页文件扩展名（HTML 系列）
_WEB_EXTS = {".html", ".htm", ".xhtml", ".mhtml"}

# 纯文本文件扩展名（注意：.html 同时出现在 _CODE_EXTS 中，
# 但 MIME 映射优先级更高，所以 text/html 会被正确分类为 web_text）
_TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".tsv", ".log", ".ini", ".cfg"}

# ─── MIME 类型精确匹配映射表 ─────────────────────────────────
# 优先级最高的分类依据

_MIME_MAP: dict[str, SourceType] = {
    "text/html": SourceType.WEB_TEXT,
    "application/pdf": SourceType.PDF_TEXT,
    "application/zip": SourceType.PACKAGE,
    "application/x-tar": SourceType.PACKAGE,
    "application/gzip": SourceType.PACKAGE,
    "application/x-executable": SourceType.BINARY,
    "application/x-mach-binary": SourceType.BINARY,
    "application/x-dosexec": SourceType.BINARY,
    "application/octet-stream": SourceType.BINARY,
}


def _classify_single(record: SourceRecord) -> SourceType:
    """
    对单个来源记录进行分类。

    分类策略按优先级：
    1) MIME 类型精确匹配（最可靠）
    2) 扩展名匹配（覆盖面广）
    3) MIME 前缀启发式（兜底）
    4) 默认返回 MIXED

    Args:
        record: 来源记录

    Returns:
        分类结果 SourceType
    """
    # 策略 1：尝试 MIME 类型精确匹配
    if record.mime_type in _MIME_MAP:
        return _MIME_MAP[record.mime_type]

    # 策略 2：尝试文件扩展名匹配
    suffix = Path(record.path).suffix.lower()
    if suffix in _CODE_EXTS:
        return SourceType.CODE
    if suffix in _PACKAGE_EXTS:
        return SourceType.PACKAGE
    if suffix in _BINARY_EXTS:
        return SourceType.BINARY
    if suffix in _PDF_EXTS:
        return SourceType.PDF_TEXT
    if suffix in _WEB_EXTS:
        return SourceType.WEB_TEXT
    # 修正 Bug 1：纯文本文件归为 MIXED 而非 web_text
    # MIXED 在步骤 C 的提取器映射中对应 _extract_plain_text（直读），
    # 这比 web_text 对应的 Trafilatura HTML 提取更合适
    if suffix in _TEXT_EXTS:
        return SourceType.MIXED

    # 策略 3：启发式——未知的 text/* MIME 类型保守地归为 CODE
    if record.mime_type.startswith("text/"):
        return SourceType.CODE

    # 兜底：返回 MIXED
    return SourceType.MIXED


def run(sources: list[SourceRecord]) -> list[SourceProfile]:
    """
    对所有来源记录执行分类。

    为每个 SourceRecord 生成对应的 SourceProfile，
    包含分类结果和附加元数据。

    Args:
        sources: 步骤 A 输出的来源记录列表

    Returns:
        SourceProfile 列表，每条记录包含来源类型分类
    """
    profiles: list[SourceProfile] = []

    for src in sources:
        # 执行单条记录分类
        source_type = _classify_single(src)

        # 构建 SourceProfile，携带原始元数据
        profile = SourceProfile(
            source_id=src.source_id,
            path=src.path,
            source_type=source_type,
            mime_type=src.mime_type,
            metadata={"size_bytes": src.size_bytes, "sha256": src.sha256},
        )
        profiles.append(profile)
        logger.debug(
            "分类结果 %s → %s (mime=%s)",
            src.source_id, source_type.value, src.mime_type,
        )

    logger.info("来源分类完成: 共 %d 个画像", len(profiles))
    return profiles

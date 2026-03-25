# ──────────────────────────────────────────────────────────────
# 步骤 A – 输入接入 (Source Intake)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   扫描用户输入的路径（文件、目录、URL），为每个独立的输入来源
#   生成一条 SourceRecord 记录，包含文件哈希、MIME 类型和大小信息。
#
# 在流水线中的位置：
#   A(本步骤) → B1(来源分类)
#
# 输出产物：
#   source_registry.jsonl（每行一条 SourceRecord JSON）
# ──────────────────────────────────────────────────────────────

"""
步骤 A – 输入接入。

扫描输入路径（文件、目录、URL），为每个来源对象生成
SourceRecord 记录，包含 SHA-256 哈希、MIME 类型和文件大小。

输出 → source_registry.jsonl
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional

from text.models.schemas import SourceRecord

logger = logging.getLogger(__name__)

# 流式哈希计算的缓冲区大小：64 KiB
# 使用流式读取避免大文件一次性加载到内存
BUFFER_SIZE = 65_536


def _sha256(file_path: Path) -> str:
    """
    计算文件的 SHA-256 哈希值。

    使用流式读取方式处理，每次读取 BUFFER_SIZE 字节的数据块，
    避免大文件耗尽内存。

    Args:
        file_path: 文件路径

    Returns:
        文件内容的 SHA-256 十六进制哈希字符串
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 海象运算符 (:=)：读取数据块并赋值给 chunk，当 chunk 为空时退出循环
        while chunk := f.read(BUFFER_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _detect_mime(file_path: Path) -> str:
    """
    检测文件的 MIME 类型。

    使用 Python 内置的 mimetypes 模块根据文件扩展名猜测 MIME 类型。
    若无法识别，返回通用的 "application/octet-stream"。

    Args:
        file_path: 文件路径

    Returns:
        MIME 类型字符串（如 "text/plain"、"application/pdf"）
    """
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"


def _is_url(path: str) -> bool:
    """
    判断给定路径是否为 URL。

    Args:
        path: 输入路径字符串

    Returns:
        如果路径以 http:// 或 https:// 开头则返回 True
    """
    return path.startswith("http://") or path.startswith("https://")


def _download_url(url: str) -> Optional[Path]:
    """
    下载 URL 指向的资源到临时文件。

    支持 HTTP/HTTPS 协议，将远程资源下载到本地临时文件中，
    以便后续步骤进行扫描和分析。

    Args:
        url: 要下载的 URL

    Returns:
        下载后的临时文件路径；下载失败时返回 None
    """
    try:
        import httpx
    except ImportError:
        logger.warning(
            "httpx 未安装，无法下载 URL '%s'。请安装：pip install httpx", url
        )
        return None

    try:
        # 使用流式下载处理大文件
        with httpx.stream("GET", url, follow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            # 从 URL 中提取文件后缀名，用于生成临时文件
            suffix = Path(url.split("?")[0].split("#")[0]).suffix or ".tmp"
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            for chunk in resp.iter_bytes(BUFFER_SIZE):
                tmp.write(chunk)
            tmp.close()
            logger.info("已下载 URL '%s' 到临时文件 '%s'", url, tmp.name)
            return Path(tmp.name)
    except Exception as e:
        logger.error("下载 URL '%s' 失败: %s", url, e)
        return None


def _collect_files(input_path: str) -> list[Path]:
    """
    将单个输入路径展开为具体的文件路径列表。

    处理三种输入类型：
    1. 单个文件路径 → 返回包含该文件的列表
    2. 目录路径 → 递归遍历所有子文件
    3. URL → 下载到临时文件后返回
    4. 不存在的路径 → 记录警告，返回空列表

    Args:
        input_path: 输入路径（文件路径、目录路径或 URL）

    Returns:
        展开后的文件路径列表（已排序）
    """
    # 处理 URL 输入：下载到临时文件
    if _is_url(input_path):
        downloaded = _download_url(input_path)
        if downloaded is not None:
            return [downloaded]
        return []

    # 处理本地路径
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        # 递归遍历目录下所有文件（排除子目录本身），按路径排序保证确定性
        return sorted(f for f in p.rglob("*") if f.is_file())

    # 路径不存在
    logger.warning("跳过不存在的路径: %s", input_path)
    return []


def run(input_paths: list[str]) -> list[SourceRecord]:
    """
    执行输入接入步骤。

    遍历所有输入路径，为每个文件生成一条 SourceRecord，
    包含文件的绝对路径、大小、SHA-256 哈希和 MIME 类型。

    Args:
        input_paths: 文件路径、目录路径或 URL 的列表

    Returns:
        SourceRecord 列表，每条记录对应一个输入文件
    """
    records: list[SourceRecord] = []

    for raw_path in input_paths:
        # 将输入路径展开为具体文件列表
        files = _collect_files(raw_path)

        for fp in files:
            try:
                # 为每个文件创建 SourceRecord
                record = SourceRecord(
                    path=str(fp.resolve()),           # 文件绝对路径
                    size_bytes=fp.stat().st_size,      # 文件大小（字节）
                    sha256=_sha256(fp),                # SHA-256 哈希值
                    mime_type=_detect_mime(fp),         # MIME 类型
                )
                records.append(record)
                logger.debug("已注册来源: %s", record.source_id)
            except Exception:
                # 单个文件处理失败不影响其他文件
                logger.exception("注册来源失败: %s", fp)

    logger.info("输入接入完成: 共注册 %d 个来源", len(records))
    return records

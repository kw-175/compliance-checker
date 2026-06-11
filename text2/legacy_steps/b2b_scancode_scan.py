# ──────────────────────────────────────────────────────────────
# 步骤 B2b – ScanCode 许可证/版权扫描 (License / Copyright Scan)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   对符合条件的来源（code, repo, package, binary, mixed）
#   运行 ScanCode-toolkit CLI，检测开源许可证和版权声明。
#
# 筛选逻辑：
#   仅对 source_type ∈ {CODE, REPO, PACKAGE, BINARY, MIXED} 的来源执行扫描，
#   跳过 WEB_TEXT 和 PDF_TEXT（它们通常不含需要检查的许可证代码）。
#
# 工作原理：
#   1. 筛选出符合条件的来源
#   2. 创建临时 JSON 文件用于 ScanCode 输出
#   3. 执行 `scancode --license --copyright --info --json-pp <output> <path>`
#   4. 解析 JSON 输出，提取许可证和版权信息
#   5. 清理临时文件
#
# Fallback 策略：
#   - ScanCode 未安装 → 记录错误，返回空结果
#   - 扫描超时 → 记录错误，返回空结果
#
# 在流水线中的位置：
#   B1(来源分类) → B2b(本步骤，与 B2a 并行) → H(证据聚合)
#
# 输出产物：
#   source_compliance.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 B2b – ScanCode 许可证/版权扫描。

对符合条件的来源执行 ScanCode-toolkit 扫描，检测开源许可证和版权声明。
输出 → source_compliance.jsonl
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from text.config.settings import Settings
from text.models.schemas import (
    ComplianceHit,
    LicenseMatch,
    SourceProfile,
    SourceType,
)

logger = logging.getLogger(__name__)

# 需要进行许可证扫描的来源类型集合
# 排除 WEB_TEXT 和 PDF_TEXT，因为它们通常不包含需要许可证检查的代码
_ELIGIBLE_TYPES = {
    SourceType.CODE,
    SourceType.REPO,
    SourceType.PACKAGE,
    SourceType.BINARY,
    SourceType.MIXED,
}


def _run_scancode(binary: str, target_path: str, output_file: str) -> dict[str, Any]:
    """
    执行 ScanCode CLI 并返回解析后的 JSON 输出。

    命令格式：
    scancode --license --copyright --info --json-pp <output_file> <target_path> --timeout 120

    Args:
        binary: ScanCode 二进制文件路径
        target_path: 要扫描的目标文件/目录路径
        output_file: JSON 输出文件路径

    Returns:
        解析后的 JSON 字典；若出错则返回空字典
    """
    cmd = [
        binary,
        "--license",               # 启用许可证检测
        "--copyright",             # 启用版权检测
        "--info",                  # 包含文件基本信息
        "--json-pp", output_file,  # 输出格式化 JSON 到指定文件
        target_path,
        "--timeout", "120",        # 单文件扫描超时 120 秒
    ]
    logger.debug("执行命令: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        logger.error(
            "ScanCode 二进制文件 '%s' 未找到。"
            "安装方式: pip install scancode-toolkit",
            binary,
        )
        return {}
    except subprocess.TimeoutExpired:
        logger.error("ScanCode 扫描超时: %s", target_path)
        return {}

    # 检查退出码
    if result.returncode != 0:
        logger.warning(
            "ScanCode 退出码 %d: %s",
            result.returncode, result.stderr[:500],
        )

    # 读取并解析 JSON 输出文件
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("读取 ScanCode 输出失败: %s", e)
        return {}


def _parse_scancode_result(
    source_id: str,
    scan_result: dict[str, Any],
) -> list[ComplianceHit]:
    """
    解析 ScanCode JSON 输出，转换为 ComplianceHit 模型列表。

    ScanCode 输出结构：
    {
        "files": [
            {
                "type": "file",
                "license_detections": [...],
                "copyrights": [...],
                "scan_errors": [...]
            },
            ...
        ]
    }

    Args:
        source_id: 关联的来源 ID
        scan_result: ScanCode 的 JSON 输出

    Returns:
        ComplianceHit 列表（每个包含许可证的文件一条记录）
    """
    hits: list[ComplianceHit] = []

    for file_entry in scan_result.get("files", []):
        # 跳过目录条目，只处理文件
        if file_entry.get("type") != "file":
            continue

        # 提取许可证检测结果
        licenses: list[LicenseMatch] = []
        for lic in file_entry.get("license_detections", []):
            for match in lic.get("matches", []):
                licenses.append(
                    LicenseMatch(
                        license_expression=lic.get("license_expression", ""),
                        spdx_id=match.get("spdx_license_expression", ""),
                        score=match.get("score", 0.0),
                        # 截断匹配文本到 500 字符，避免数据过大
                        matched_text=match.get("matched_text", "")[:500],
                        start_line=match.get("start_line", 0),
                        end_line=match.get("end_line", 0),
                    )
                )

        # 提取版权声明
        copyrights = [
            c.get("copyright", "")
            for c in file_entry.get("copyrights", [])
            if c.get("copyright")
        ]

        # 提取扫描错误
        scan_errors = file_entry.get("scan_errors", [])

        # 只有存在许可证、版权或错误时才生成 ComplianceHit
        if licenses or copyrights or scan_errors:
            hits.append(
                ComplianceHit(
                    source_id=source_id,
                    file_path=file_entry.get("path", ""),
                    licenses=licenses,
                    copyrights=copyrights,
                    scan_errors=scan_errors,
                )
            )

    return hits


def run(
    profiles: list[SourceProfile],
    settings: Settings | None = None,
) -> list[ComplianceHit]:
    """
    执行 ScanCode 许可证扫描。

    根据来源类型筛选出需要扫描的文件，逐一执行 ScanCode 分析，
    将结果汇总为 ComplianceHit 列表。

    Args:
        profiles: 来源画像列表（步骤 B1 输出）
        settings: 配置对象（可选）

    Returns:
        ComplianceHit 列表
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # 筛选出符合条件的来源类型
    eligible = [p for p in profiles if p.source_type in _ELIGIBLE_TYPES]
    logger.info(
        "ScanCode: %d/%d 个来源符合扫描条件",
        len(eligible), len(profiles),
    )

    all_hits: list[ComplianceHit] = []

    for profile in eligible:
        target = profile.path
        # 检查文件是否存在
        if not Path(target).exists():
            logger.warning("跳过不存在的路径: %s", target)
            continue

        # 创建临时文件存放 ScanCode 的 JSON 输出
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_file = tmp.name

        # 执行扫描
        scan_result = _run_scancode(settings.scancode_bin, target, output_file)
        if scan_result:
            hits = _parse_scancode_result(profile.source_id, scan_result)
            all_hits.extend(hits)

        # 清理临时文件
        try:
            Path(output_file).unlink(missing_ok=True)
        except Exception:
            pass  # 清理失败不影响流程

    logger.info("ScanCode 扫描完成: %d 条合规检测记录", len(all_hits))
    return all_hits

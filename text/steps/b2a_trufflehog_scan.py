# ──────────────────────────────────────────────────────────────
# 步骤 B2a – TruffleHog 密钥扫描 (Secret Scan)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   通过 subprocess 调用 TruffleHog v3 CLI 扫描原始源文件，
#   检测泄露的密钥、API Key、令牌、密码等敏感凭证。
#
# 工作原理：
#   1. 收集所有来源文件的父目录（去重，避免重复扫描）
#   2. 对每个目录执行 `trufflehog filesystem <path> --json --no-update`
#   3. 解析 JSON 行输出，转换为 SecretHit 记录
#
# Fallback 策略：
#   - TruffleHog 二进制不存在 → 记录错误日志，返回空结果
#   - 扫描超时 → 记录错误日志，返回空结果
#   - 不会中断主流程
#
# 在流水线中的位置：
#   B1(来源分类) → B2a(本步骤，与 B2b 并行) → H(证据聚合)
#
# 输出产物：
#   raw_secret_hits.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 B2a – TruffleHog 密钥扫描。

调用 TruffleHog v3 CLI 扫描源文件中的泄露密钥/凭证。
输出 → raw_secret_hits.jsonl
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from text.config.settings import Settings
from text.models.schemas import SecretHit, SourceRecord

logger = logging.getLogger(__name__)


def _run_trufflehog(binary: str, target_path: str) -> list[dict[str, Any]]:
    """
    执行 TruffleHog 文件系统扫描并返回解析后的 JSON 结果。

    命令格式：trufflehog filesystem <path> --json --no-update
    每行 stdout 输出为一个独立的 JSON finding。

    Args:
        binary: TruffleHog 二进制文件路径
        target_path: 要扫描的目标路径

    Returns:
        解析后的 finding 字典列表；若出错则返回空列表
    """
    cmd = [binary, "filesystem", target_path, "--json", "--no-update"]
    logger.debug("执行命令: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,     # 捕获 stdout 和 stderr
            text=True,               # 以文本模式解码输出
            timeout=300,             # 5 分钟超时
        )
    except FileNotFoundError:
        # TruffleHog 未安装
        logger.error(
            "TruffleHog 二进制文件 '%s' 未找到。"
            "安装方式: https://github.com/trufflesecurity/trufflehog#installation",
            binary,
        )
        return []
    except subprocess.TimeoutExpired:
        # 扫描超时
        logger.error("TruffleHog 扫描超时: %s", target_path)
        return []

    # 逐行解析 JSON 输出
    findings: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            findings.append(json.loads(line))
        except json.JSONDecodeError:
            # 跳过无法解析的输出行
            logger.warning("无法解析的 TruffleHog 输出行: %s", line[:120])

    # 检查退出码（0=无发现, 1=有发现, 其他=错误）
    if result.returncode not in (0, 1):
        logger.warning(
            "TruffleHog 退出码 %d: %s",
            result.returncode, result.stderr[:500],
        )

    return findings


def _parse_finding(source_id: str, finding: dict[str, Any]) -> SecretHit:
    """
    将单个 TruffleHog JSON finding 转换为 SecretHit 模型。

    TruffleHog 的输出结构大致为：
    {
        "SourceMetadata": {"Data": {"Filesystem": {"file": "..."}}},
        "DetectorType": "...",
        "Raw": "...",
        "Verified": true/false,
        ...
    }

    Args:
        source_id: 关联的来源 ID
        finding: TruffleHog 的原始 JSON 输出

    Returns:
        转换后的 SecretHit 实例
    """
    # 提取文件系统相关元数据
    source_meta = finding.get("SourceMetadata", {}).get("Data", {})
    filesystem_data = source_meta.get("Filesystem", {})

    return SecretHit(
        source_id=source_id,
        detector_type=finding.get("DetectorType", ""),   # 检测器类型（如 AWS、GitHub）
        decoder_type=finding.get("DecoderType", ""),     # 解码器类型
        raw_value=finding.get("Raw", ""),                # 原始密钥值
        redacted=finding.get("Redacted", ""),            # 脱敏后的值
        file_path=filesystem_data.get("file", ""),       # 发现密钥的文件路径
        line_number=filesystem_data.get("line", 0),      # 所在行号
        verified=finding.get("Verified", False),         # 是否已验证有效
        extra={
            "detector_name": finding.get("DetectorName", ""),
            "extra_data": finding.get("ExtraData", {}),
        },
    )


def run(
    sources: list[SourceRecord],
    settings: Settings | None = None,
) -> list[SecretHit]:
    """
    执行 TruffleHog 密钥扫描。

    遍历所有来源文件，按父目录去重后执行扫描。
    通过 finding 中的文件路径反查正确的 source_id，
    确保每个 finding 关联到正确的来源。

    Args:
        sources: 来源记录列表（步骤 A 输出）
        settings: 配置对象（可选，默认从环境加载）

    Returns:
        SecretHit 列表
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    all_hits: list[SecretHit] = []
    scanned_dirs: set[str] = set()  # 已扫描目录集合（避免重复扫描）

    # 修正 Bug 4：构建文件路径到 source_id 的反向映射表
    # 当同一目录包含多个文件时，确保每个 finding 关联到正确的来源
    path_to_source_id: dict[str, str] = {}
    for src in sources:
        path_to_source_id[str(Path(src.path).resolve())] = src.source_id

    for src in sources:
        # TruffleHog 按目录扫描效果最好，因此取父目录
        target = str(Path(src.path).parent)
        if target in scanned_dirs:
            continue
        scanned_dirs.add(target)

        findings = _run_trufflehog(settings.trufflehog_bin, target)
        for f in findings:
            # 尝试从 finding 中获取文件路径，反查正确的 source_id
            finding_file = f.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("file", "")
            resolved_file = str(Path(finding_file).resolve()) if finding_file else ""
            # 如果能匹配到具体来源则用匹配到的 source_id，否则用当前循环的 source_id
            matched_source_id = path_to_source_id.get(resolved_file, src.source_id)
            hit = _parse_finding(matched_source_id, f)
            all_hits.append(hit)

    logger.info("TruffleHog 扫描完成: 发现 %d 个密钥泄露", len(all_hits))
    return all_hits

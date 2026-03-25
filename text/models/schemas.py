# ──────────────────────────────────────────────────────────────
# 数据模型定义模块 (Data Models / Schemas)
# ──────────────────────────────────────────────────────────────
#
# 本模块使用 Pydantic v2 定义了流水线各步骤的输入/输出数据模型。
# 每个模型对应一行 JSONL 或一个 JSON 文件中的顶层对象。
#
# 模型对照表：
#   步骤 A  → SourceRecord         (source_registry.jsonl)
#   步骤 B1 → SourceProfile        (source_profile.jsonl)
#   步骤 B2a→ SecretHit            (raw_secret_hits.jsonl)
#   步骤 B2b→ ComplianceHit        (source_compliance.jsonl)
#   步骤 C  → CleanedDocument      (cleaned_documents.jsonl)
#   步骤 D  → DedupDocument        (deduped_documents.jsonl)
#              DedupMapEntry        (dedup_map.jsonl)
#   步骤 E1a→ KeywordHit           (keyword_hits.jsonl)
#   步骤 E1b→ RegexHit             (regex_hits.jsonl)
#   步骤 F  → PrivacyResult        (privacy_checked.jsonl)
#   步骤 G  → SafetyResult         (safety_checked.jsonl)
#   步骤 H  → EvidenceBundle       (evidence_bundle.json)
#   步骤 I  → PolicyDecision       (decision.json)
#   API     → CheckRequest / CheckTaskInfo
# ──────────────────────────────────────────────────────────────

"""
Pydantic 数据模型定义模块。

为流水线的每个步骤定义输入输出数据结构，确保数据在各步骤间传递时
类型安全、可序列化、可校验。所有模型使用 Pydantic v2 BaseModel，
支持自动 JSON 序列化/反序列化。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────
# 枚举类型定义
# ────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    """
    来源类型枚举。

    在步骤 B1（来源分类）中，根据文件扩展名和 MIME 类型
    将每个输入来源分类为以下类型之一：
    """
    CODE = "code"           # 代码文件（.py, .js, .java 等）
    REPO = "repo"           # 代码仓库
    PACKAGE = "package"     # 软件包（.whl, .tar.gz, .jar 等）
    BINARY = "binary"       # 二进制文件（.exe, .dll, .so 等）
    WEB_TEXT = "web_text"   # 网页文本（.html, .htm 等）
    PDF_TEXT = "pdf_text"   # PDF 文档
    MIXED = "mixed"         # 混合/纯文本类型（.txt, .md 等也归入此类）


class Decision(str, Enum):
    """
    策略决策结果枚举。

    步骤 I 的输出决策，优先级从高到低：
    REJECT > QUARANTINE > REVIEW > ALLOW
    """
    ALLOW = "allow"             # 允许通过：无合规风险
    REVIEW = "review"           # 需人工审核：存在中等风险
    QUARANTINE = "quarantine"   # 隔离：存在较高风险（如 copyleft 许可证）
    REJECT = "reject"           # 拒绝：存在严重风险（如密钥泄露、不安全内容）


class SafetyLevel(str, Enum):
    """
    安全等级枚举。

    步骤 G（安全审核）的三级分类结果：
    """
    SAFE = "safe"                   # 安全：无有害内容
    CONTROVERSIAL = "controversial" # 争议：内容存在争议但不直接有害
    UNSAFE = "unsafe"               # 不安全：包含暴力、仇恨等有害内容


# ────────────────────────────────────────────────────────────
# 步骤 A – source_registry.jsonl
# 输入接入阶段：记录每个原始输入文件的元信息
# ────────────────────────────────────────────────────────────

class SourceRecord(BaseModel):
    """
    来源记录模型（步骤 A 输出）。

    每条记录对应一个输入文件，包含文件路径、大小、SHA-256 哈希、
    MIME 类型和注册时间。用于后续步骤的来源追溯。

    Attributes:
        source_id: 唯一标识符（自动生成的 12 位十六进制字符串）
        path: 文件的绝对路径
        size_bytes: 文件大小（字节）
        sha256: 文件内容的 SHA-256 哈希值
        mime_type: 文件的 MIME 类型
        created_at: 记录创建时间（UTC）
    """
    source_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: str
    size_bytes: int = 0
    sha256: str = ""
    mime_type: str = ""
    # 修正：使用 timezone-aware 的 datetime 替代已弃用的 datetime.utcnow()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}


# ────────────────────────────────────────────────────────────
# 步骤 B1 – source_profile.jsonl
# 来源分类阶段：为每个来源打上类型标签
# ────────────────────────────────────────────────────────────

class SourceProfile(BaseModel):
    """
    来源画像模型（步骤 B1 输出）。

    在 SourceRecord 基础上增加了分类信息，用于后续步骤
    决定使用哪种提取/扫描策略。

    Attributes:
        source_id: 关联的来源 ID（与 SourceRecord.source_id 对应）
        path: 文件路径
        source_type: 来源类型分类结果
        mime_type: MIME 类型
        metadata: 附加元数据（如文件大小、哈希等）
    """
    source_id: str
    path: str
    source_type: SourceType
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 步骤 B2a – raw_secret_hits.jsonl
# TruffleHog 密钥扫描结果
# ────────────────────────────────────────────────────────────

class SecretHit(BaseModel):
    """
    密钥泄露检测结果模型（步骤 B2a 输出）。

    每条记录代表 TruffleHog 在源文件中发现的一个可能的密钥/凭证泄露。

    Attributes:
        source_id: 关联的来源 ID
        detector_type: 检测器类型（如 AWS、GitHub 等）
        decoder_type: 解码器类型
        raw_value: 原始密钥值（敏感，仅在安全环境使用）
        redacted: 脱敏后的值
        file_path: 发现密钥的文件路径
        line_number: 所在行号
        verified: TruffleHog 是否已验证该密钥有效
        extra: 额外信息（检测器名称、附加数据等）
    """
    source_id: str
    detector_type: str = ""
    decoder_type: str = ""
    raw_value: str = ""
    redacted: str = ""
    file_path: str = ""
    line_number: int = 0
    verified: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 步骤 B2b – source_compliance.jsonl
# ScanCode 许可证/版权扫描结果
# ────────────────────────────────────────────────────────────

class LicenseMatch(BaseModel):
    """
    许可证匹配详情（ScanCode 单条匹配记录）。

    Attributes:
        license_expression: 许可证表达式（如 "MIT", "GPL-3.0-only"）
        spdx_id: SPDX 标准许可证标识符
        score: 匹配置信度（0-100）
        matched_text: 匹配到的原文片段（截断至 500 字符）
        start_line: 匹配起始行号
        end_line: 匹配结束行号
    """
    license_expression: str = ""
    spdx_id: str = ""
    score: float = 0.0
    matched_text: str = ""
    start_line: int = 0
    end_line: int = 0


class ComplianceHit(BaseModel):
    """
    合规扫描结果模型（步骤 B2b 输出）。

    汇总单个源文件的所有许可证检测、版权声明和扫描错误。

    Attributes:
        source_id: 关联的来源 ID
        file_path: 被扫描文件的路径
        licenses: 检测到的许可证列表
        copyrights: 检测到的版权声明列表
        scan_errors: 扫描过程中出现的错误信息
    """
    source_id: str
    file_path: str = ""
    licenses: list[LicenseMatch] = Field(default_factory=list)
    copyrights: list[str] = Field(default_factory=list)
    scan_errors: list[str] = Field(default_factory=list)


# ────────────────────────────────────────────────────────────
# 步骤 C – cleaned_documents.jsonl
# 文本提取与清洗结果
# ────────────────────────────────────────────────────────────

class CleanedDocument(BaseModel):
    """
    清洗后文档模型（步骤 C 输出）。

    从原始来源中提取的纯文本内容，经过 Unicode 规范化、
    空白压缩、特殊字符清理后的干净文本。

    Attributes:
        doc_id: 文档唯一标识符（自动生成）
        source_id: 关联的来源 ID
        text: 清洗后的文本内容
        char_count: 字符数
        language: 检测到的语言（ISO 639-1 代码，如 "en"、"zh"）
        metadata: 附加元数据（来源类型、原始路径等）
    """
    doc_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_id: str
    text: str
    char_count: int = 0
    language: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 步骤 D – deduped_documents.jsonl + dedup_map.jsonl
# 去重结果
# ────────────────────────────────────────────────────────────

class DedupDocument(BaseModel):
    """
    去重后文档模型（步骤 D 输出之一）。

    标记每个文档是否为重复项，若重复则记录其原始文档 ID。

    Attributes:
        doc_id: 文档 ID
        source_id: 关联的来源 ID
        text: 文本内容
        is_duplicate: 是否为重复文档
        duplicate_of: 若为重复，指向原始文档的 doc_id
        minhash_signature: MinHash 签名（可选，用于调试）
    """
    doc_id: str
    source_id: str
    text: str
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    minhash_signature: Optional[list[int]] = None


class DedupMapEntry(BaseModel):
    """
    去重映射记录（步骤 D 输出之二）。

    记录重复文档对之间的关系和相似度。

    Attributes:
        doc_id: 被判定为重复的文档 ID
        duplicate_of: 原始（保留的）文档 ID
        jaccard_similarity: 两篇文档的 Jaccard 相似度（1.0 表示完全相同）
    """
    doc_id: str
    duplicate_of: str
    jaccard_similarity: float = 0.0


# ────────────────────────────────────────────────────────────
# 步骤 E1a – keyword_hits.jsonl
# 关键词扫描结果
# ────────────────────────────────────────────────────────────

class KeywordHit(BaseModel):
    """
    关键词命中记录（步骤 E1a 输出）。

    记录在文档中检测到的敏感关键词及其位置信息。

    Attributes:
        doc_id: 所属文档 ID
        keyword: 命中的关键词
        start_pos: 关键词在文本中的起始字符位置
        end_pos: 关键词在文本中的结束字符位置
        context: 关键词周围的上下文片段
    """
    doc_id: str
    keyword: str
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# ────────────────────────────────────────────────────────────
# 步骤 E1b – regex_hits.jsonl
# 正则表达式扫描结果
# ────────────────────────────────────────────────────────────

class RegexHit(BaseModel):
    """
    正则匹配记录（步骤 E1b 输出）。

    记录在文档中匹配到的敏感模式（如邮箱、SSN、API 密钥等）。

    Attributes:
        doc_id: 所属文档 ID
        pattern_name: 模式名称（如 "email_address"、"us_ssn"）
        pattern: 使用的正则表达式
        matched_text: 匹配到的文本（截断至 200 字符）
        start_pos: 匹配起始位置
        end_pos: 匹配结束位置
        context: 匹配周围的上下文片段
    """
    doc_id: str
    pattern_name: str
    pattern: str = ""
    matched_text: str = ""
    start_pos: int = 0
    end_pos: int = 0
    context: str = ""


# ────────────────────────────────────────────────────────────
# 步骤 F – privacy_checked.jsonl
# PII 检测与脱敏结果
# ────────────────────────────────────────────────────────────

class PIIEntity(BaseModel):
    """
    PII（个人身份信息）实体记录。

    记录单个检测到的 PII 实体的详细信息。

    Attributes:
        entity_type: 实体类型（如 PERSON、EMAIL_ADDRESS、PHONE_NUMBER）
        start: 实体在原始文本中的起始位置
        end: 实体在原始文本中的结束位置
        score: 检测置信度分数
        original_text: 原始文本片段（截断至 100 字符）
    """
    entity_type: str
    start: int
    end: int
    score: float = 0.0
    original_text: str = ""


class PrivacyResult(BaseModel):
    """
    隐私检测结果模型（步骤 F 输出）。

    包含原始文本、脱敏后文本以及检测到的所有 PII 实体列表。

    Attributes:
        doc_id: 文档 ID
        original_text: 原始文本
        redacted_text: 脱敏后的文本（PII 被替换为 <REDACTED>、<EMAIL> 等）
        pii_entities: 检测到的 PII 实体列表
        pii_count: PII 实体总数
    """
    doc_id: str
    original_text: str = ""
    redacted_text: str = ""
    pii_entities: list[PIIEntity] = Field(default_factory=list)
    pii_count: int = 0


# ────────────────────────────────────────────────────────────
# 步骤 G – safety_checked.jsonl
# 安全内容审核结果
# ────────────────────────────────────────────────────────────

class SafetyResult(BaseModel):
    """
    安全审核结果模型（步骤 G 输出）。

    对脱敏后文本进行安全性分类，标记危害类别。

    Attributes:
        doc_id: 文档 ID
        safety_level: 安全等级（Safe/Controversial/Unsafe）
        harm_categories: 检测到的危害类别列表
        raw_output: 模型原始输出文本（截断至 500 字符）
        score: 安全评分（1.0=安全，0.5=争议，0.0=不安全）
    """
    doc_id: str
    safety_level: SafetyLevel = SafetyLevel.SAFE
    harm_categories: list[str] = Field(default_factory=list)
    raw_output: str = ""
    score: float = 1.0


# ────────────────────────────────────────────────────────────
# 步骤 H – evidence_bundle.json
# 证据聚合结果
# ────────────────────────────────────────────────────────────

class DocumentEvidence(BaseModel):
    """
    单文档证据模型（步骤 H 中间模型）。

    将步骤 B2/D/E1/F/G 的所有检测结果按文档维度聚合。

    Attributes:
        doc_id: 文档 ID
        source_id: 关联的来源 ID
        secret_hits: 该来源的密钥泄露记录
        compliance_hits: 该来源的许可证合规记录
        is_duplicate: 是否为重复文档
        keyword_hits: 该文档的关键词命中记录
        regex_hits: 该文档的正则匹配记录
        privacy: 该文档的隐私检测结果
        safety: 该文档的安全审核结果
    """
    doc_id: str
    source_id: str
    secret_hits: list[SecretHit] = Field(default_factory=list)
    compliance_hits: list[ComplianceHit] = Field(default_factory=list)
    is_duplicate: bool = False
    keyword_hits: list[KeywordHit] = Field(default_factory=list)
    regex_hits: list[RegexHit] = Field(default_factory=list)
    privacy: Optional[PrivacyResult] = None
    safety: Optional[SafetyResult] = None


class EvidenceBundle(BaseModel):
    """
    证据包模型（步骤 H 输出）。

    汇总整个流水线运行的所有检测证据，包含统计摘要。

    Attributes:
        pipeline_run_id: 流水线运行 ID
        created_at: 创建时间（UTC）
        documents: 所有文档的证据列表
        summary: 统计摘要（总文档数、重复数、各类检测命中数等）
    """
    pipeline_run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # 修正：使用 timezone-aware 的 datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    documents: list[DocumentEvidence] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


# ────────────────────────────────────────────────────────────
# 步骤 I – decision.json
# 策略决策结果
# ────────────────────────────────────────────────────────────

class DocumentDecision(BaseModel):
    """
    单文档决策模型（步骤 I 中间模型）。

    记录针对单个文档的合规决策、原因和各维度评分。

    Attributes:
        doc_id: 文档 ID
        decision: 决策结果（allow/review/quarantine/reject）
        reasons: 决策原因列表（人类可读的说明）
        scores: 各维度评分字典（secrets、safety、privacy、compliance、text_scan）
    """
    doc_id: str
    decision: Decision = Decision.REVIEW
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    """
    策略决策模型（步骤 I 输出）。

    包含总体决策和所有文档的单独决策。
    总体决策取所有文档中最严格的决策。

    Attributes:
        pipeline_run_id: 流水线运行 ID
        overall_decision: 总体决策（取最严格的）
        document_decisions: 各文档的决策列表
        evaluated_at: 评估时间（UTC）
    """
    pipeline_run_id: str
    overall_decision: Decision = Decision.REVIEW
    document_decisions: list[DocumentDecision] = Field(default_factory=list)
    # 修正：使用 timezone-aware 的 datetime
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ────────────────────────────────────────────────────────────
# FastAPI 服务相关模型
# 用于 API 请求/响应和任务状态追踪
# ────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    """
    任务状态枚举。

    用于追踪后台异步任务的执行状态：
    PENDING → RUNNING → COMPLETED / FAILED
    """
    PENDING = "pending"       # 待处理：任务已创建，等待执行
    RUNNING = "running"       # 运行中：流水线正在执行
    COMPLETED = "completed"   # 已完成：流水线执行成功
    FAILED = "failed"         # 已失败：流水线执行出错


class CheckRequest(BaseModel):
    """
    合规检查请求模型（POST /api/v1/check 请求体）。

    Attributes:
        input_paths: 待检查的文件路径、目录路径或 URL 列表
        config_overrides: 可选的配置覆盖项（如临时修改阈值）
    """
    input_paths: list[str] = Field(
        ..., description="File paths, directory paths, or URLs to check"
    )
    config_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional overrides for pipeline settings",
    )


class CheckTaskInfo(BaseModel):
    """
    任务信息模型（API 返回的任务追踪信息）。

    Attributes:
        task_id: 任务唯一标识符
        status: 当前任务状态
        created_at: 任务创建时间（UTC）
        completed_at: 任务完成时间（可选）
        result: 流水线执行结果（仅在 COMPLETED 状态下有值）
        error: 错误信息（仅在 FAILED 状态下有值）
    """
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    # 修正：使用 timezone-aware 的 datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    result: Optional[PolicyDecision] = None
    error: Optional[str] = None

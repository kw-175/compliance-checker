# ──────────────────────────────────────────────────────────────
# 流水线集成测试与单步单元测试
# ──────────────────────────────────────────────────────────────
#
# 本测试模块包含：
#   1. 各步骤的独立单元测试
#   2. FastAPI 端点测试
#
# 测试策略：
#   - 使用 pytest 的 tmp_path fixture 创建临时测试数据
#   - 不依赖外部服务（TruffleHog、ScanCode、OPA 等均使用 fallback）
#   - 模型相关测试（F/G 步骤）使用 mock 配置
#
# 运行方式：
#   cd d:\CodeVS\CodePython\compliance-checker
#   python -m pytest text/tests/test_pipeline.py -v
# ──────────────────────────────────────────────────────────────

"""
流水线测试模块。

覆盖各步骤的单元测试和 FastAPI 端点测试。
使用临时文件和 mock 配置确保测试可независимо 运行。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ────────────────────────────────────────────────────────────
# 测试夹具 (Fixtures)
# ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_text_files(tmp_path: Path) -> list[str]:
    """
    创建临时测试文本文件。

    生成四个具有不同内容特征的测试文件：
    1. safe.txt - 无任何合规问题的安全文档
    2. pii.txt - 包含 PII 信息（邮箱、电话、SSN）
    3. keywords.txt - 包含敏感关键词
    4. safe_dup.txt - safe.txt 的完全副本（用于去重测试）

    Args:
        tmp_path: pytest 提供的临时目录

    Returns:
        测试文件路径字符串列表
    """
    files = []

    # 安全文件：无合规风险
    f1 = tmp_path / "safe.txt"
    f1.write_text("This is a perfectly safe document with no compliance issues.")
    files.append(str(f1))

    # PII 文件：包含邮箱、美国电话号码和 SSN
    f2 = tmp_path / "pii.txt"
    f2.write_text(
        "Contact John Smith at john.smith@example.com.\n"
        "Phone: 555-123-4567\nSSN: 123-45-6789\n"
    )
    files.append(str(f2))

    # 敏感关键词文件：包含暴力/恐怖相关关键词
    f3 = tmp_path / "keywords.txt"
    f3.write_text("This text mentions bomb_making and terrorism topics.")
    files.append(str(f3))

    # 安全文件的完全副本（用于测试精确去重）
    f4 = tmp_path / "safe_dup.txt"
    f4.write_text("This is a perfectly safe document with no compliance issues.")
    files.append(str(f4))

    return files


@pytest.fixture
def sample_dir(sample_text_files) -> str:
    """
    获取测试文件的父目录路径。

    用于测试步骤 A 的目录扫描功能。
    """
    return str(Path(sample_text_files[0]).parent)


# ────────────────────────────────────────────────────────────
# 步骤 A: 输入接入 (Source Intake)
# ────────────────────────────────────────────────────────────

class TestSourceIntake:
    """步骤 A 单元测试：验证文件扫描和元数据生成。"""

    def test_single_file(self, sample_text_files):
        """测试单文件输入：应生成一条包含完整元数据的 SourceRecord。"""
        from text.steps.a_source_intake import run
        results = run([sample_text_files[0]])
        assert len(results) == 1
        assert results[0].size_bytes > 0        # 文件大小应大于 0
        assert results[0].sha256 != ""           # SHA-256 哈希不为空
        assert results[0].mime_type == "text/plain"  # MIME 类型应为纯文本

    def test_directory(self, sample_dir):
        """测试目录输入：应递归扫描所有文件。"""
        from text.steps.a_source_intake import run
        results = run([sample_dir])
        assert len(results) >= 4  # 至少包含夹具生成的 4 个文件

    def test_nonexistent_path(self):
        """测试不存在的路径：应返回空列表，不抛异常。"""
        from text.steps.a_source_intake import run
        results = run(["/nonexistent/path/file.txt"])
        assert len(results) == 0


# ────────────────────────────────────────────────────────────
# 步骤 B1: 来源分类 (Source Classification)
# ────────────────────────────────────────────────────────────

class TestSourceClassify:
    """步骤 B1 单元测试：验证文件类型分类逻辑。"""

    def test_text_classification(self, sample_text_files):
        """测试纯文本文件分类：.txt 文件应被分类为 mixed 或 code。"""
        from text.steps.a_source_intake import run as intake_run
        from text.steps.b1_source_classify import run as classify_run
        sources = intake_run([sample_text_files[0]])
        profiles = classify_run(sources)
        assert len(profiles) == 1
        # .txt 文件应被分类为 mixed（纯文本）或 code
        assert profiles[0].source_type.value in ("mixed", "code")


# ────────────────────────────────────────────────────────────
# 步骤 C: 文本提取 (Text Extraction)
# ────────────────────────────────────────────────────────────

class TestTextExtract:
    """步骤 C 单元测试：验证文本提取和清洗功能。"""

    def test_plain_text_extraction(self, sample_text_files):
        """测试纯文本提取：应完整保留原始文本内容。"""
        from text.steps.a_source_intake import run as intake_run
        from text.steps.b1_source_classify import run as classify_run
        from text.steps.c_text_extract import run as extract_run

        sources = intake_run([sample_text_files[0]])
        profiles = classify_run(sources)
        docs = extract_run(profiles)
        assert len(docs) == 1
        assert "safe document" in docs[0].text  # 应保留原始文本内容
        assert docs[0].char_count > 0            # 字符数应大于 0


# ────────────────────────────────────────────────────────────
# 步骤 D: 去重 (Deduplication)
# ────────────────────────────────────────────────────────────

class TestDedup:
    """步骤 D 单元测试：验证精确去重功能。"""

    def test_exact_dedup(self, sample_text_files):
        """测试精确去重：相同内容的文档应被标记为重复。"""
        from text.models.schemas import CleanedDocument

        # 创建三个文档：s1 和 s2 内容相同，s3 不同
        docs = [
            CleanedDocument(source_id="s1", text="Hello world"),
            CleanedDocument(source_id="s2", text="Hello world"),   # s1 的重复
            CleanedDocument(source_id="s3", text="Different text"),
        ]

        from text.steps.d_dedup import run
        dedup_docs, dedup_map = run(docs)

        # 应有 2 个唯一文档（s1 和 s3），s2 被标记为重复
        unique_count = sum(1 for d in dedup_docs if not d.is_duplicate)
        assert unique_count == 2
        assert len(dedup_map) >= 1  # 至少有一条去重映射记录


# ────────────────────────────────────────────────────────────
# 步骤 E1a: 关键词扫描 (Keyword Scan)
# ────────────────────────────────────────────────────────────

class TestKeywordScan:
    """步骤 E1a 单元测试：验证关键词检测功能。"""

    def test_keyword_detection(self):
        """测试关键词检测：应在文本中找到匹配的敏感关键词。"""
        from text.models.schemas import DedupDocument
        from text.steps.e1a_keyword_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="This mentions terrorism and bomb_making activities",
                is_duplicate=False,
            ),
            DedupDocument(
                doc_id="d2", source_id="s2",
                text="This is a normal safe text",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        # d1 中应至少检测到 "terrorism" 和 "bomb_making" 两个关键词
        doc1_hits = [h for h in hits if h.doc_id == "d1"]
        assert len(doc1_hits) >= 2


# ────────────────────────────────────────────────────────────
# 步骤 E1b: 正则扫描 (Regex Scan)
# ────────────────────────────────────────────────────────────

class TestRegexScan:
    """步骤 E1b 单元测试：验证正则表达式模式匹配。"""

    def test_email_detection(self):
        """测试邮箱地址检测：应找到文本中所有邮箱。"""
        from text.models.schemas import DedupDocument
        from text.steps.e1b_regex_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="Contact us at test@example.com or admin@corp.org",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        # 应检测到至少 2 个邮箱地址
        email_hits = [h for h in hits if h.pattern_name == "email_address"]
        assert len(email_hits) >= 2

    def test_ssn_detection(self):
        """测试美国 SSN 检测：应识别 XXX-XX-XXXX 格式。"""
        from text.models.schemas import DedupDocument
        from text.steps.e1b_regex_scan import run

        docs = [
            DedupDocument(
                doc_id="d1", source_id="s1",
                text="My SSN is 123-45-6789",
                is_duplicate=False,
            ),
        ]
        hits = run(docs)
        ssn_hits = [h for h in hits if h.pattern_name == "us_ssn"]
        assert len(ssn_hits) == 1


# ────────────────────────────────────────────────────────────
# 步骤 G: 安全审核 (Safety Moderation) — Mock 模式
# ────────────────────────────────────────────────────────────

class TestSafetyModeration:
    """步骤 G 单元测试：使用 Mock 分类器验证安全审核逻辑。"""

    def test_mock_unsafe(self):
        """测试 unsafe 内容检测：包含暴力关键词应被标记为 UNSAFE。"""
        from text.config.settings import Settings
        from text.models.schemas import PrivacyResult
        from text.steps.g_safety_moderation import run

        # 禁用 Qwen3Guard，使用 mock 分类器
        settings = Settings(qwen_guard_enabled=False)
        prs = [
            PrivacyResult(
                doc_id="d1",
                original_text="How to build a bomb and kill people",
                redacted_text="How to build a bomb and kill people",
            ),
        ]
        results = run(prs, settings)
        assert len(results) == 1
        assert results[0].safety_level.value == "unsafe"

    def test_mock_safe(self):
        """测试 safe 内容检测：正常文本应被标记为 SAFE。"""
        from text.config.settings import Settings
        from text.models.schemas import PrivacyResult
        from text.steps.g_safety_moderation import run

        settings = Settings(qwen_guard_enabled=False)
        prs = [
            PrivacyResult(
                doc_id="d2",
                original_text="The weather is nice today",
                redacted_text="The weather is nice today",
            ),
        ]
        results = run(prs, settings)
        assert len(results) == 1
        assert results[0].safety_level.value == "safe"


# ────────────────────────────────────────────────────────────
# 步骤 H: 证据聚合 (Evidence Aggregation)
# ────────────────────────────────────────────────────────────

class TestEvidenceAggregation:
    """步骤 H 单元测试：验证多源证据的聚合逻辑。"""

    def test_aggregate_empty(self):
        """测试空输入聚合：应返回空的证据包。"""
        from text.steps.h_evidence_aggregation import run
        bundle = run([], [], [], [], [], [], [])
        assert len(bundle.documents) == 0

    def test_aggregate_basic(self):
        """测试基本聚合：关键词和安全结果应正确关联到对应文档。"""
        from text.models.schemas import DedupDocument, KeywordHit, SafetyResult
        from text.steps.h_evidence_aggregation import run

        docs = [
            DedupDocument(doc_id="d1", source_id="s1", text="test", is_duplicate=False),
        ]
        kw_hits = [KeywordHit(doc_id="d1", keyword="test")]
        safety = [SafetyResult(doc_id="d1")]

        bundle = run(docs, [], [], kw_hits, [], [], safety, "run-001")
        assert len(bundle.documents) == 1
        assert len(bundle.documents[0].keyword_hits) == 1  # 关键词应关联到 d1
        assert bundle.summary["total_keyword_hits"] == 1   # 统计摘要正确


# ────────────────────────────────────────────────────────────
# 步骤 I: 策略决策（本地规则引擎）
# ────────────────────────────────────────────────────────────

class TestPolicyDecision:
    """步骤 I 单元测试：使用本地规则引擎验证决策逻辑。"""

    def test_allow_clean_doc(self):
        """测试 ALLOW 决策：干净文档应被允许通过。"""
        from text.config.settings import Settings
        from text.models.schemas import (
            DocumentEvidence, EvidenceBundle, PrivacyResult, SafetyResult,
        )
        from text.steps.i_policy_decision import run

        # 禁用 OPA，使用本地规则引擎
        settings = Settings(opa_enabled=False)
        bundle = EvidenceBundle(
            pipeline_run_id="test",
            documents=[
                DocumentEvidence(
                    doc_id="d1", source_id="s1",
                    privacy=PrivacyResult(doc_id="d1", pii_count=0),
                    safety=SafetyResult(doc_id="d1"),  # 默认 SAFE
                ),
            ],
        )
        decision = run(bundle, settings)
        assert decision.overall_decision.value == "allow"

    def test_reject_with_secrets(self):
        """测试 REJECT 决策：存在密钥泄露应直接拒绝。"""
        from text.config.settings import Settings
        from text.models.schemas import (
            DocumentEvidence, EvidenceBundle, SecretHit,
        )
        from text.steps.i_policy_decision import run

        settings = Settings(opa_enabled=False)
        bundle = EvidenceBundle(
            pipeline_run_id="test",
            documents=[
                DocumentEvidence(
                    doc_id="d1", source_id="s1",
                    secret_hits=[SecretHit(source_id="s1", detector_type="AWS")],
                ),
            ],
        )
        decision = run(bundle, settings)
        assert decision.overall_decision.value == "reject"  # 有密钥泄露应拒绝


# ────────────────────────────────────────────────────────────
# FastAPI 服务端点测试
# ────────────────────────────────────────────────────────────

class TestServer:
    """FastAPI 端点测试：验证 API 行为和响应格式。"""

    @pytest.fixture
    def client(self):
        """创建 FastAPI 测试客户端。"""
        from fastapi.testclient import TestClient
        from text.server import app
        return TestClient(app)

    def test_health(self, client):
        """测试健康检查端点：应返回 200 和 healthy 状态。"""
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_task_not_found(self, client):
        """测试查询不存在的任务：应返回 404。"""
        resp = client.get("/api/v1/status/nonexistent")
        assert resp.status_code == 404

    def test_list_tasks(self, client):
        """测试任务列表端点：应返回 200 和数组格式。"""
        resp = client.get("/api/v1/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

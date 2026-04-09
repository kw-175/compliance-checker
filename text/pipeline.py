# ──────────────────────────────────────────────────────────────
# 流水线编排器 (Pipeline Orchestrator)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   将步骤 A → J 串联成完整的合规检测流水线。
#   支持以下特性：
#   - 顺序执行和并行执行（B2a/B2b 并行, E1a/E1b 并行）
#   - 每步输出持久化到 JSONL/JSON 文件
#   - 每步集成 OpenLineage 血缘追踪
#   - 步骤失败优雅降级（不中断整体流水线）
#
# 执行流程：
#   A(输入接入) → B1(分类) → B2a/B2b(并行扫描)
#   → C(文本提取) → D(去重) → E1a/E1b(并行规则扫描)
#   → F(隐私检测) → G(安全审核) → H(证据聚合)
#   → I(策略决策)
#
# 输出目录结构：
#   {work_dir}/{run_id}/
#   ├── source_registry.jsonl      (步骤 A)
#   ├── source_profile.jsonl       (步骤 B1)
#   ├── raw_secret_hits.jsonl      (步骤 B2a)
#   ├── source_compliance.jsonl    (步骤 B2b)
#   ├── cleaned_documents.jsonl    (步骤 C)
#   ├── deduped_documents.jsonl    (步骤 D)
#   ├── dedup_map.jsonl            (步骤 D)
#   ├── keyword_hits.jsonl         (步骤 E1a)
#   ├── regex_hits.jsonl           (步骤 E1b)
#   ├── privacy_checked.jsonl      (步骤 F)
#   ├── safety_checked.jsonl       (步骤 G)
#   ├── evidence_bundle.json       (步骤 H)
#   └── decision.json              (步骤 I)
# ──────────────────────────────────────────────────────────────

"""
流水线编排器。

将步骤 A→J 串联为 CompliancePipeline，
支持顺序/并行执行、JSONL 持久化和 OpenLineage 血缘。
"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from text.config.settings import Settings, get_settings
from text.models.schemas import (
    CleanedDocument,
    ComplianceHit,
    DedupDocument,
    DedupMapEntry,
    EvidenceBundle,
    KeywordHit,
    PolicyDecision,
    PrivacyResult,
    RegexHit,
    SafetyResult,
    SecretHit,
    SourceProfile,
    SourceRecord,
)

# 统一契约层导入
from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from common.evidence import DegradeEvent
from common.runtime import PipelineExecutionContext, TrustEvaluator

logger = logging.getLogger(__name__)


def _write_jsonl(records: list, output_path: Path) -> None:
    """
    将 Pydantic 模型列表写入 JSONL 文件。

    JSONL 格式：每行一个 JSON 对象，适合流式处理和追加写入。
    自动创建父目录。

    Args:
        records: Pydantic 模型实例列表
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.debug("已写入 %d 条记录到 %s", len(records), output_path)


def _write_json(obj: Any, output_path: Path) -> None:
    """
    将单个 Pydantic 模型写入 JSON 文件。

    使用 indent=2 进行格式化，方便人工审查。
    自动创建父目录。

    Args:
        obj: Pydantic 模型实例
        output_path: 输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if hasattr(obj, "model_dump_json"):
            f.write(obj.model_dump_json(indent=2))
        else:
            f.write(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    logger.debug("已写入 JSON 到 %s", output_path)


class CompliancePipeline:
    """
    合规检测流水线编排器。

    管理所有步骤的执行顺序、并行调度和结果持久化。
    每次执行生成一个唯一的 run_id，所有中间产物保存在
    {work_dir}/{run_id}/ 目录下。

    改进特性：
    - 使用 PipelineExecutionContext 记录每步执行状态和降级事件
    - 步骤失败生成 DegradeEvent 而非仅日志
    - 最终输出 ComplianceOutput（含双轨交付物）
    - 旧的 PolicyDecision 通过 legacy_decision 保留向后兼容

    Parameters:
        settings: 流水线配置，默认从环境变量加载

    Attributes:
        settings: 配置对象
        run_id: 当前运行的唯一标识符
        output_dir: 中间产物输出目录
    """

    def __init__(self, settings: Settings | None = None):
        """
        初始化流水线。

        Args:
            settings: 配置对象（可选，默认从环境加载）
        """
        self.settings = settings or get_settings()
        self.run_id = uuid.uuid4().hex  # 生成唯一运行 ID
        self.output_dir = self.settings.work_dir / self.run_id

        # 延迟初始化血缘追踪器（避免导入时加载 OpenLineage）
        self._tracker = None

        # 统一执行上下文（记录步骤状态、降级事件、失败信息）
        self.exec_ctx = PipelineExecutionContext(pipeline_run_id=self.run_id)

    @property
    def tracker(self):
        """
        延迟加载的 LineageTracker 实例。

        避免在导入模块时初始化 OpenLineage 客户端，
        仅在首次调用时创建。
        """
        if self._tracker is None:
            from text.steps.j_lineage_audit import LineageTracker
            self._tracker = LineageTracker(self.settings)
        return self._tracker

    # ── 步骤执行器：集成血缘追踪 + 执行上下文 ────────────

    def _run_step(self, step_name: str, func, *args, output_file: str | None = None, **kwargs):
        """
        通用步骤执行器（带血缘追踪 + 执行上下文记录）。

        在步骤前后自动：
        - 记录 PipelineExecutionContext 步骤开始/完成/失败
        - 发送 OpenLineage 的 START / COMPLETE / FAIL 事件
        - 失败时生成 DegradeEvent

        Args:
            step_name: 步骤名称（用于血缘事件）
            func: 步骤执行函数
            *args: 传递给步骤函数的位置参数
            output_file: 输出文件名（用于血缘事件的 outputs）
            **kwargs: 传递给步骤函数的关键字参数

        Returns:
            步骤函数的返回值
        """
        # 记录步骤开始
        self.exec_ctx.record_step_start(step_name)

        # 发送 START 事件
        run_id = self.tracker.start_step(
            step_name,
            outputs=[{"name": output_file}] if output_file else None,
        )
        try:
            # 执行步骤函数
            result = func(*args, **kwargs)
            # 记录步骤完成
            self.exec_ctx.record_step_complete(step_name)
            # 发送 COMPLETE 事件
            self.tracker.complete_step(
                step_name, run_id,
                outputs=[{"name": output_file}] if output_file else None,
            )
            return result
        except Exception as e:
            # 记录步骤失败并生成 DegradeEvent
            self.exec_ctx.record_step_failure(step_name, error=str(e))
            # 发送 FAIL 事件
            self.tracker.fail_step(step_name, run_id, str(e))
            raise

    def execute(self, input_paths: list[str]) -> ComplianceOutput:
        """
        执行完整的合规检测流水线。

        按照 A → B1 → B2(并行) → C → D → E1(并行) → F → G → H → I
        的顺序执行所有步骤，并将每步的结果持久化到磁盘。

        改进：
        - 步骤失败时生成 DegradeEvent 并记录到执行上下文
        - 最终输出 ComplianceOutput（含标注样本包 + 审计证据包）
        - 旧的 PolicyDecision 通过 legacy_decision 保留

        Args:
            input_paths: 文件路径、目录路径或 URL 列表

        Returns:
            ComplianceOutput 统一输出契约（含双轨交付物）
        """
        logger.info(
            "═══ 流水线运行 %s 开始 ═══ (输入路径数: %d)",
            self.run_id[:8], len(input_paths),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── 步骤 A: 输入接入 ─────────────────────────────
        # 扫描输入路径，生成来源注册表
        from text.steps import a_source_intake
        sources: list[SourceRecord] = self._run_step(
            "step_a_source_intake",
            a_source_intake.run,
            input_paths,
            output_file="source_registry.jsonl",
        )
        _write_jsonl(sources, self.output_dir / "source_registry.jsonl")

        # 若无来源则提前终止
        if not sources:
            logger.warning("未找到任何来源 – 中止流水线")
            return self._build_empty_output()

        # ── 步骤 B1: 来源分类 ────────────────────────────
        # 根据 MIME 类型和扩展名对来源进行分类
        from text.steps import b1_source_classify
        profiles: list[SourceProfile] = self._run_step(
            "step_b1_source_classify",
            b1_source_classify.run,
            sources,
            output_file="source_profile.jsonl",
        )
        _write_jsonl(profiles, self.output_dir / "source_profile.jsonl")

        # ── 步骤 B2: 原始对象扫描（并行）─────────────────
        # B2a(TruffleHog 密钥扫描) 和 B2b(ScanCode 许可证扫描) 并行执行
        from text.steps import b2a_trufflehog_scan, b2b_scancode_scan

        secret_hits: list[SecretHit] = []
        compliance_hits: list[ComplianceHit] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            # 提交两个并行任务
            future_secrets = executor.submit(
                self._run_step,
                "step_b2a_trufflehog",
                b2a_trufflehog_scan.run,
                sources, self.settings,
                output_file="raw_secret_hits.jsonl",
            )
            future_compliance = executor.submit(
                self._run_step,
                "step_b2b_scancode",
                b2b_scancode_scan.run,
                profiles, self.settings,
                output_file="source_compliance.jsonl",
            )

            # 收集结果（单个任务失败不中断其他任务）
            try:
                secret_hits = future_secrets.result()
            except Exception as e:
                logger.error("TruffleHog 扫描失败（已记录降级事件）: %s", e)

            try:
                compliance_hits = future_compliance.result()
            except Exception as e:
                logger.error("ScanCode 扫描失败（已记录降级事件）: %s", e)

        _write_jsonl(secret_hits, self.output_dir / "raw_secret_hits.jsonl")
        _write_jsonl(compliance_hits, self.output_dir / "source_compliance.jsonl")

        # ── 步骤 C: 文本提取与预处理 ─────────────────────
        # 从分类后的来源中提取纯文本，执行 Unicode 规范化和清洗
        from text.steps import c_text_extract
        cleaned_docs: list[CleanedDocument] = self._run_step(
            "step_c_text_extract",
            c_text_extract.run,
            profiles, self.settings,
            output_file="cleaned_documents.jsonl",
        )
        _write_jsonl(cleaned_docs, self.output_dir / "cleaned_documents.jsonl")

        # 若无提取到文本则提前终止
        if not cleaned_docs:
            logger.warning("未提取到文本 – 中止流水线")
            return self._build_empty_output()

        # ── 步骤 D: 早期去重 ─────────────────────────────
        # SHA-256 精确去重 + MinHash LSH 近似去重
        from text.steps import d_dedup
        dedup_docs, dedup_map = self._run_step(
            "step_d_dedup",
            d_dedup.run,
            cleaned_docs, self.settings,
            output_file="deduped_documents.jsonl",
        )
        _write_jsonl(dedup_docs, self.output_dir / "deduped_documents.jsonl")
        _write_jsonl(dedup_map, self.output_dir / "dedup_map.jsonl")

        # 统计非重复文档数量
        active_docs = [d for d in dedup_docs if not d.is_duplicate]
        logger.info("活跃（非重复）文档数: %d", len(active_docs))

        # ── 步骤 E1: 确定性文本扫描（并行）──────────────
        # E1a(关键词扫描) 和 E1b(正则扫描) 并行执行
        from text.steps import e1a_keyword_scan, e1b_regex_scan

        keyword_hits: list[KeywordHit] = []
        regex_hits: list[RegexHit] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_kw = executor.submit(
                self._run_step,
                "step_e1a_keyword_scan",
                e1a_keyword_scan.run,
                dedup_docs, self.settings,
                output_file="keyword_hits.jsonl",
            )
            future_rx = executor.submit(
                self._run_step,
                "step_e1b_regex_scan",
                e1b_regex_scan.run,
                dedup_docs, self.settings,
                output_file="regex_hits.jsonl",
            )

            try:
                keyword_hits = future_kw.result()
            except Exception as e:
                logger.error("关键词扫描失败（已记录降级事件）: %s", e)

            try:
                regex_hits = future_rx.result()
            except Exception as e:
                logger.error("正则扫描失败（已记录降级事件）: %s", e)

        _write_jsonl(keyword_hits, self.output_dir / "keyword_hits.jsonl")
        _write_jsonl(regex_hits, self.output_dir / "regex_hits.jsonl")

        # ── 步骤 F: 隐私检测与脱敏 ──────────────────────
        # 使用 Presidio 检测 PII 并进行脱敏替换
        from text.steps import f_privacy_detection
        privacy_results: list[PrivacyResult] = self._run_step(
            "step_f_privacy_detection",
            f_privacy_detection.run,
            dedup_docs, self.settings,
            output_file="privacy_checked.jsonl",
        )
        _write_jsonl(privacy_results, self.output_dir / "privacy_checked.jsonl")

        # 检查隐私步骤是否降级
        if any(r.is_degraded for r in privacy_results):
            self.exec_ctx.record_step_failure(
                "step_f_privacy_detection",
                error="Presidio 不可用，使用 fallback passthrough",
                fallback_provider="fallback_passthrough",
            )

        # ── 步骤 G: 语义安全审核 ─────────────────────────
        # 使用 Qwen3Guard 或 Mock 分类器进行安全性分类
        from text.steps import g_safety_moderation
        safety_results: list[SafetyResult] = self._run_step(
            "step_g_safety_moderation",
            g_safety_moderation.run,
            privacy_results, self.settings,
            output_file="safety_checked.jsonl",
        )
        _write_jsonl(safety_results, self.output_dir / "safety_checked.jsonl")

        # 检查安全审核是否降级
        if any(getattr(r, "is_degraded", False) for r in safety_results):
            self.exec_ctx.record_step_failure(
                "step_g_safety_moderation",
                error="安全审核使用了 mock/fallback",
                fallback_provider="mock_keyword",
                is_mock=True,
            )

        # ── 步骤 H: 证据聚合 ─────────────────────────────
        # 按文档维度汇总所有检测结果
        from text.steps import h_evidence_aggregation
        evidence_bundle: EvidenceBundle = self._run_step(
            "step_h_evidence_aggregation",
            h_evidence_aggregation.run,
            dedup_docs,
            secret_hits,
            compliance_hits,
            keyword_hits,
            regex_hits,
            privacy_results,
            safety_results,
            self.run_id,
            output_file="evidence_bundle.json",
        )
        _write_json(evidence_bundle, self.output_dir / "evidence_bundle.json")

        # ── 步骤 I: 策略决策 ─────────────────────────────
        # 使用 OPA 或本地规则引擎做最终合规决策
        from text.steps import i_policy_decision
        decision: PolicyDecision = self._run_step(
            "step_i_policy_decision",
            i_policy_decision.run,
            evidence_bundle, self.settings,
            output_file="decision.json",
        )
        _write_json(decision, self.output_dir / "decision.json")

        # ── 步骤 K: 双轨交付物构建 ──────────────────────
        # 将检测结果转换为统一证据单元，构建标注样本包 + 审计证据包
        compliance_output = self._build_compliance_output(
            evidence_bundle, decision, input_paths,
        )
        _write_json(compliance_output, self.output_dir / "compliance_output.json")

        logger.info(
            "═══ 流水线运行 %s 完成 ═══ 总体决策: %s, 可信等级: %s",
            self.run_id[:8],
            compliance_output.decision.value,
            compliance_output.trust_level.value,
        )
        return compliance_output

    def _build_empty_output(self) -> ComplianceOutput:
        """构建无内容的输出（无来源或无文本时使用）。"""
        from common.adapters import build_compliance_output
        return build_compliance_output(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=UnifiedDecision.REVIEW,
            trust_level=TrustLevel.FULL,
            release_package=None,
        )

    def _build_compliance_output(
        self,
        evidence_bundle: EvidenceBundle,
        decision: PolicyDecision,
        input_paths: list[str],
    ) -> ComplianceOutput:
        """
        从旧模型构建统一输出契约。

        步骤：
        1. 将 DocumentEvidence 中的各类 hit 转换为 EvidenceUnit
        2. 去重与归并同一对象的多条命中
        3. 使用 evaluate_with_profile 进行 Profile 化策略评估
        4. 构建 AnnotationPackage（标注样本包）
        5. 构建 AuditPackage（审计证据包）
        6. 组装 ReleasePackage 和最终 ComplianceOutput
        """
        from common.adapters import (
            build_annotation_package,
            build_audit_package,
            build_compliance_output,
            build_release_package,
            convert_text_evidence,
            deduplicate_evidence_units,
            map_text_decision_to_unified,
        )
        from common.policy import evaluate_with_profile, load_policy_profile

        # 1. 转换所有证据
        all_evidence_units = []
        for doc in evidence_bundle.documents:
            units = convert_text_evidence(doc)
            all_evidence_units.extend(units)

        # 2. 去重
        all_evidence_units = deduplicate_evidence_units(all_evidence_units)

        # 3. Profile 化策略评估
        profile = load_policy_profile("default")
        policy_result = evaluate_with_profile(
            all_evidence_units,
            profile=profile,
            degrade_events=self.exec_ctx.degrade_events,
        )

        # 4. 可信等级
        trust_level = TrustEvaluator.evaluate(self.exec_ctx)
        unified_decision = policy_result.decision

        # 5. 标注样本包
        content_uri = str(self.output_dir / "cleaned_documents.jsonl")
        annotation_pkg = build_annotation_package(
            modality=Modality.TEXT,
            pipeline_run_id=self.run_id,
            clean_content_uri=content_uri,
            content_format="application/jsonl",
            evidence_units=all_evidence_units,
            decision=unified_decision,
            trust_level=trust_level,
        )

        # 6. 审计证据包
        audit_pkg = build_audit_package(
            modality=Modality.TEXT,
            pipeline_run_id=self.run_id,
            evidence_units=all_evidence_units,
            degrade_events=self.exec_ctx.degrade_events,
            policy_result=policy_result,
            ctx=self.exec_ctx,
        )

        # 7. 组装发布包
        release_pkg = build_release_package(
            modality=Modality.TEXT,
            pipeline_run_id=self.run_id,
            annotation_package=annotation_pkg,
            audit_package=audit_pkg,
            decision=unified_decision,
            trust_level=trust_level,
        )

        # 8. 最终输出
        legacy = decision.model_dump() if decision else None
        return build_compliance_output(
            pipeline_run_id=self.run_id,
            modality=Modality.TEXT,
            decision=unified_decision,
            trust_level=trust_level,
            release_package=release_pkg,
            degrade_summary=policy_result.degrade_summary,
            review_suggestions=policy_result.review_suggestions,
            explanation_summary=audit_pkg.review_summary,
            legacy_decision=legacy,
        )

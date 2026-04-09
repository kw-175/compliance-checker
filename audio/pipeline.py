"""
Pipeline orchestrator for audio compliance checking.
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from audio.config.settings import Settings, get_settings
from audio.models.schemas import EvidenceBundle, PolicyDecision, ReleasePackage

# 统一契约层导入
from common.contracts import ComplianceOutput
from common.enums import Modality, TrustLevel, UnifiedDecision
from common.runtime import PipelineExecutionContext, TrustEvaluator

logger = logging.getLogger(__name__)


def _write_jsonl(records: list, output_path: Path) -> None:
    # 确保输出目录存在，避免写文件时因目录缺失失败。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            # 统一按 JSONL 格式逐行落盘，便于后续步骤流式读取。
            handle.write(record.model_dump_json() + "\n")


def _write_json(record: Any, output_path: Path) -> None:
    # 单对象结果写入 JSON 文件（带缩进，便于审计阅读）。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if hasattr(record, "model_dump_json"):
            handle.write(record.model_dump_json(indent=2))
        else:
            import json
            json.dump(record, handle, indent=2, ensure_ascii=False)


class AudioCompliancePipeline:
    def __init__(self, settings: Settings | None = None):
        # 配置优先使用显式传入值，否则从环境变量动态加载。
        self.settings = settings or get_settings()
        # 每次运行生成独立 run_id，用于追踪产物和审计链路。
        self.run_id = uuid.uuid4().hex
        self.output_dir = self.settings.work_dir / self.run_id
        # 延迟初始化 lineage tracker，未使用时不额外开销。
        self._tracker = None
        # 统一执行上下文（记录步骤状态、降级事件、失败信息）
        self.exec_ctx = PipelineExecutionContext(pipeline_run_id=self.run_id)

    @property
    def tracker(self):
        if self._tracker is None:
            # 复用 text 侧的谱系跟踪实现，记录每步状态与产出。
            from audio.steps.j_lineage_audit import LineageTracker
            self._tracker = LineageTracker(self.settings)
        return self._tracker

    def _run_step(self, step_name: str, func, *args, output_file: str | None = None, **kwargs):
        # 所有步骤统一经过该包装器，确保开始/成功/失败都进入审计轨迹和执行上下文。
        self.exec_ctx.record_step_start(step_name)
        run_id = self.tracker.start_step(step_name, outputs=[{"name": output_file}] if output_file else None)
        try:
            result = func(*args, **kwargs)
            self.exec_ctx.record_step_complete(step_name)
            self.tracker.complete_step(step_name, run_id, outputs=[{"name": output_file}] if output_file else None)
            return result
        except Exception as exc:
            self.exec_ctx.record_step_failure(step_name, error=str(exc))
            self.tracker.fail_step(step_name, run_id, str(exc))
            raise

    def execute(self, input_paths: list[str]) -> PolicyDecision:
        # 在方法内部导入，避免模块初始化时加载大量可选依赖。
        from audio.steps import (
            a_source_intake,
            b1_source_classify,
            b2a_trufflehog_scan,
            b2b_scancode_scan,
            c0_audio_normalize,
            c1_asr_transcribe,
            c1b_diarization,
            c1c_alignment,
            c2_transcript_build,
            d_dedup,
            e1a_keyword_scan,
            e1b_regex_scan,
            f_privacy_detection,
            g_safety_moderation,
            h_evidence_aggregation,
            i_policy_decision,
            k_audio_redaction,
            l_release_package,
        )

        logger.info("Audio pipeline run %s started", self.run_id[:8])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # A: 采集输入源元数据（路径、哈希、MIME 等）。
        sources = self._run_step("step_a_source_intake", a_source_intake.run, input_paths, output_file="source_registry.jsonl")
        _write_jsonl(sources, self.output_dir / "source_registry.jsonl")
        if not sources:
            # 无可处理输入时返回默认空输出。
            return self._build_empty_output()

        # B1: 识别输入类型（音频/仓库/压缩包等）。
        profiles = self._run_step("step_b1_source_classify", b1_source_classify.run, sources, output_file="source_profile.jsonl")
        _write_jsonl(profiles, self.output_dir / "source_profile.jsonl")

        secret_hits = []
        compliance_hits = []
        # B2a/B2b 并行执行，缩短整体耗时；某个分支失败时允许降级继续。
        with ThreadPoolExecutor(max_workers=max(1, min(2, int(self.settings.max_workers or 2)))) as executor:
            secret_future = executor.submit(self._run_step, "step_b2a_trufflehog", b2a_trufflehog_scan.run, sources, self.settings, output_file="raw_secret_hits.jsonl")
            compliance_future = executor.submit(self._run_step, "step_b2b_scancode", b2b_scancode_scan.run, profiles, self.settings, output_file="source_compliance.jsonl")
            try:
                secret_hits = secret_future.result()
            except Exception as exc:
                # 扫描失败不阻断全流程，留日志用于运维排查。
                logger.warning("TruffleHog step degraded: %s", exc)
            try:
                compliance_hits = compliance_future.result()
            except Exception as exc:
                logger.warning("ScanCode step degraded: %s", exc)
        _write_jsonl(secret_hits, self.output_dir / "raw_secret_hits.jsonl")
        _write_jsonl(compliance_hits, self.output_dir / "source_compliance.jsonl")

        # C0: 音频归一化（采样率/声道/编码统一）。
        normalized = self._run_step("step_c0_audio_normalize", c0_audio_normalize.run, profiles, self.settings, self.output_dir, output_file="normalized_audio_manifest.jsonl")
        _write_jsonl(normalized, self.output_dir / "normalized_audio_manifest.jsonl")
        if not normalized:
            return self._build_empty_output()

        # C1: 语音转写；C1b: 说话人分离；C1c: 对齐占位。
        asr_segments = self._run_step("step_c1_asr_transcribe", c1_asr_transcribe.run, normalized, self.settings, output_file="asr_segments.jsonl")
        _write_jsonl(asr_segments, self.output_dir / "asr_segments.jsonl")

        speaker_segments = self._run_step("step_c1b_diarization", c1b_diarization.run, normalized, self.settings, output_file="speaker_segments.jsonl")
        _write_jsonl(speaker_segments, self.output_dir / "speaker_segments.jsonl")

        aligned_segments = self._run_step("step_c1c_alignment", c1c_alignment.run, asr_segments, output_file="aligned_segments.jsonl")
        _write_jsonl(aligned_segments, self.output_dir / "aligned_segments.jsonl")

        # C2: 结合说话人信息构建统一 transcript unit。
        transcript_units = self._run_step("step_c2_transcript_build", c2_transcript_build.run, aligned_segments, speaker_segments, output_file="transcript_units.jsonl")
        _write_jsonl(transcript_units, self.output_dir / "transcript_units.jsonl")

        # D: 去重并记录重复映射关系，减少后续重复检测成本。
        deduped_units, dedup_map = self._run_step("step_d_dedup", d_dedup.run, transcript_units, self.settings, output_file="deduped_transcript_units.jsonl")
        _write_jsonl(deduped_units, self.output_dir / "deduped_transcript_units.jsonl")
        _write_jsonl(dedup_map, self.output_dir / "dedup_map.jsonl")

        keyword_hits = []
        regex_hits = []
        # E1a/E1b 并行文本规则扫描，继续遵循“失败降级不阻断”策略。
        with ThreadPoolExecutor(max_workers=max(1, min(2, int(self.settings.max_workers or 2)))) as executor:
            keyword_future = executor.submit(self._run_step, "step_e1a_keyword_scan", e1a_keyword_scan.run, deduped_units, self.settings, output_file="keyword_hits.jsonl")
            regex_future = executor.submit(self._run_step, "step_e1b_regex_scan", e1b_regex_scan.run, deduped_units, self.settings, output_file="regex_hits.jsonl")
            try:
                keyword_hits = keyword_future.result()
            except Exception as exc:
                logger.warning("Keyword scan degraded: %s", exc)
            try:
                regex_hits = regex_future.result()
            except Exception as exc:
                logger.warning("Regex scan degraded: %s", exc)
        _write_jsonl(keyword_hits, self.output_dir / "keyword_hits.jsonl")
        _write_jsonl(regex_hits, self.output_dir / "regex_hits.jsonl")

        # F/G: 隐私识别与语义安全审查。
        privacy_results, redaction_spans = self._run_step("step_f_privacy_detection", f_privacy_detection.run, deduped_units, self.settings, output_file="privacy_checked.jsonl")
        _write_jsonl(privacy_results, self.output_dir / "privacy_checked.jsonl")
        _write_jsonl(redaction_spans, self.output_dir / "redaction_spans.jsonl")

        safety_results = self._run_step("step_g_safety_moderation", g_safety_moderation.run, privacy_results, self.settings, output_file="safety_checked.jsonl")
        _write_jsonl(safety_results, self.output_dir / "safety_checked.jsonl")

        # H/I: 证据聚合并据此做策略决策。
        evidence_bundle: EvidenceBundle = self._run_step(
            "step_h_evidence_aggregation",
            h_evidence_aggregation.run,
            deduped_units,
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

        decision = self._run_step("step_i_policy_decision", i_policy_decision.run, evidence_bundle, self.settings, output_file="decision.json")
        _write_json(decision, self.output_dir / "decision.json")

        # K: 将隐私打码区间映射回音频生成脱敏版本。
        redacted_audio = self._run_step("step_k_audio_redaction", k_audio_redaction.run, normalized, redaction_spans, self.settings, self.output_dir, output_file="redacted_audio_manifest.jsonl")
        _write_jsonl(redacted_audio, self.output_dir / "redacted_audio_manifest.jsonl")

        # L: 汇总最终交付包，包含决策、证据摘要和产物索引。
        release_package: ReleasePackage = self._run_step(
            "step_l_release_package",
            l_release_package.run,
            self.run_id,
            normalized,
            evidence_bundle,
            decision,
            redacted_audio,
            {"output_dir": str(self.output_dir.resolve())},
            output_file="release_package.json",
        )
        _write_json(release_package, self.output_dir / "release_package.json")

        logger.info("Audio pipeline run %s completed with decision=%s", self.run_id[:8], decision.overall_decision.value)

        # ── 步骤 M: 统一契约输出构建 ────────────────
        compliance_output = self._build_compliance_output(
            evidence_bundle, decision, redaction_spans, normalized,
        )
        _write_json(compliance_output, self.output_dir / "compliance_output.json")
        return compliance_output

    def _build_empty_output(self) -> ComplianceOutput:
        """构建无内容的输出（无来源或无音频时使用）。"""
        from common.adapters import build_compliance_output
        return build_compliance_output(
            pipeline_run_id=self.run_id,
            modality=Modality.AUDIO,
            decision=UnifiedDecision.REVIEW,
            trust_level=TrustLevel.FULL,
            release_package=None,
        )

    def _build_compliance_output(
        self,
        evidence_bundle: EvidenceBundle,
        decision: PolicyDecision,
        redaction_spans: list,
        normalized: list,
    ) -> ComplianceOutput:
        """从旧模型构建统一输出契约。"""
        from common.adapters import (
            audio_redaction_span_to_evidence,
            build_annotation_package,
            build_audit_package,
            build_compliance_output,
            build_release_package,
            deduplicate_evidence_units,
            map_text_decision_to_unified,
        )
        from common.policy import evaluate_with_profile, load_policy_profile

        # 1. 转换证据
        all_units = []
        for span in redaction_spans:
            all_units.append(audio_redaction_span_to_evidence(span))
        all_units = deduplicate_evidence_units(all_units)

        # 2. Profile 化策略评伋
        profile = load_policy_profile("default")
        policy_result = evaluate_with_profile(
            all_units, profile=profile,
            degrade_events=self.exec_ctx.degrade_events,
        )

        trust_level = TrustEvaluator.evaluate(self.exec_ctx)
        unified_decision = policy_result.decision

        # 3. 标注样本包
        content_uri = str(self.output_dir / "normalized_audio_manifest.jsonl")
        annotation_pkg = build_annotation_package(
            modality=Modality.AUDIO,
            pipeline_run_id=self.run_id,
            clean_content_uri=content_uri,
            content_format="audio/wav",
            evidence_units=all_units,
            decision=unified_decision,
            trust_level=trust_level,
        )

        # 4. 审计证据包
        audit_pkg = build_audit_package(
            modality=Modality.AUDIO,
            pipeline_run_id=self.run_id,
            evidence_units=all_units,
            degrade_events=self.exec_ctx.degrade_events,
            policy_result=policy_result,
            ctx=self.exec_ctx,
        )

        # 5. 组装发布包
        release_pkg = build_release_package(
            modality=Modality.AUDIO,
            pipeline_run_id=self.run_id,
            annotation_package=annotation_pkg,
            audit_package=audit_pkg,
            decision=unified_decision,
            trust_level=trust_level,
        )

        legacy = decision.model_dump() if decision else None
        return build_compliance_output(
            pipeline_run_id=self.run_id,
            modality=Modality.AUDIO,
            decision=unified_decision,
            trust_level=trust_level,
            release_package=release_pkg,
            degrade_summary=policy_result.degrade_summary,
            review_suggestions=policy_result.review_suggestions,
            explanation_summary=audit_pkg.review_summary,
            legacy_decision=legacy,
        )

# ──────────────────────────────────────────────────────────────
# 跨模态桥接适配器
# ──────────────────────────────────────────────────────────────
#
# 负责将各模态（text / audio / picture / video）的旧数据模型
# 转换为 common/ 统一契约中的标准结构，包括：
#   - EvidenceUnit（统一证据单元）
#   - AnnotationPackage（标注样本包）
#   - AuditPackage（审计证据包）
#   - ReleasePackage（统一发布包）
#   - ComplianceOutput（统一输出契约）
#
# 使用方式：
#   在各模态 pipeline 的末尾调用本模块的工具函数，
#   将模态特有的检测结果统一转换为下游可消费的标准产物。
# ──────────────────────────────────────────────────────────────

"""跨模态桥接适配器：旧模型 → 统一契约。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from common.contracts import ComplianceOutput
from common.delivery import (
    AnnotationPackage,
    AuditPackage,
    ReleasePackage,
    RiskSegment,
)
from common.enums import Modality, TrustLevel, UnifiedDecision
from common.evidence import (
    AudioTimeSpan,
    DegradeEvent,
    EvidenceUnit,
    ImageRegion,
    TextSpan,
    VideoTimeRegion,
)
from common.policy import PolicyResult, RuleTrace, ScoreBreakdown, ScoreDimension
from common.runtime import PipelineExecutionContext, TrustEvaluator

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Text 模态：各类 Hit → EvidenceUnit ─────────────────────


def text_keyword_hit_to_evidence(hit: Any, doc_id: str = "") -> EvidenceUnit:
    """将 text.KeywordHit 转为 EvidenceUnit。"""
    return EvidenceUnit(
        modality=Modality.TEXT,
        category="keyword",
        sub_category=hit.keyword,
        text_span=TextSpan(
            start=hit.start_pos,
            end=hit.end_pos,
            text_snippet=hit.keyword,
            context_before=hit.context[:50] if hit.context else "",
            context_after=hit.context[50:] if hit.context and len(hit.context) > 50 else "",
        ),
        confidence=1.0,
        provider="flashtext2",
        reason_code=f"keyword:{hit.keyword}",
        doc_id=doc_id or getattr(hit, "doc_id", ""),
    )


def text_regex_hit_to_evidence(hit: Any, doc_id: str = "") -> EvidenceUnit:
    """将 text.RegexHit 转为 EvidenceUnit。"""
    return EvidenceUnit(
        modality=Modality.TEXT,
        category="regex",
        sub_category=hit.pattern_name,
        text_span=TextSpan(
            start=hit.start_pos,
            end=hit.end_pos,
            text_snippet=hit.matched_text[:100] if hit.matched_text else "",
            context_before=hit.context[:50] if hit.context else "",
            context_after=hit.context[50:] if hit.context and len(hit.context) > 50 else "",
        ),
        confidence=1.0,
        provider="hyperscan",
        reason_code=f"regex:{hit.pattern_name}",
        doc_id=doc_id or getattr(hit, "doc_id", ""),
    )


def text_secret_hit_to_evidence(hit: Any, source_id: str = "") -> EvidenceUnit:
    """将 text.SecretHit 转为 EvidenceUnit。"""
    return EvidenceUnit(
        modality=Modality.TEXT,
        category="secret",
        sub_category=hit.detector_type,
        text_span=TextSpan(
            start=hit.line_number,
            end=hit.line_number,
            text_snippet=hit.redacted[:100] if hit.redacted else "",
        ),
        confidence=1.0 if hit.verified else 0.7,
        provider="trufflehog",
        reason_code=f"secret:{hit.detector_type}",
        source_id=source_id or hit.source_id,
        metadata={"verified": hit.verified, "file_path": hit.file_path},
    )


def text_pii_entity_to_evidence(
    entity: Any, doc_id: str = "", provider_name: str = "", provider_version: str = "",
) -> EvidenceUnit:
    """将 text.PIIEntity 转为 EvidenceUnit。"""
    return EvidenceUnit(
        modality=Modality.TEXT,
        category="pii",
        sub_category=entity.entity_type.lower(),
        text_span=TextSpan(
            start=entity.start,
            end=entity.end,
            text_snippet=entity.original_text[:100] if entity.original_text else "",
            context_before=getattr(entity, "context_before", ""),
            context_after=getattr(entity, "context_after", ""),
        ),
        confidence=entity.score,
        provider=provider_name or "presidio",
        provider_version=provider_version,
        reason_code=f"pii:{entity.entity_type.lower()}",
        doc_id=doc_id,
    )


def text_safety_to_evidence(
    safety: Any, doc_id: str = "",
) -> Optional[EvidenceUnit]:
    """将 text.SafetyResult 转为 EvidenceUnit（仅非 SAFE 时）。"""
    if safety.safety_level.value == "safe":
        return None
    return EvidenceUnit(
        modality=Modality.TEXT,
        category="safety",
        sub_category=safety.safety_level.value,
        confidence=1.0 - safety.score,
        provider=getattr(safety, "provider_name", "") or "qwen3guard",
        model_version=getattr(safety, "model_version", ""),
        threshold_used=getattr(safety, "threshold_used", 0.0),
        explanation=getattr(safety, "explanation", "") or f"Safety level: {safety.safety_level.value}",
        reason_code=f"safety:{safety.safety_level.value}",
        doc_id=doc_id,
        metadata={"harm_categories": safety.harm_categories},
    )


def text_compliance_hit_to_evidence(hit: Any, source_id: str = "") -> list[EvidenceUnit]:
    """将 text.ComplianceHit 转为 EvidenceUnit 列表（每个许可证一条）。"""
    units = []
    for lic in getattr(hit, "licenses", []):
        units.append(EvidenceUnit(
            modality=Modality.TEXT,
            category="license",
            sub_category=lic.license_expression,
            text_span=TextSpan(
                start=lic.start_line,
                end=lic.end_line,
                text_snippet=lic.matched_text[:100] if lic.matched_text else "",
            ),
            confidence=lic.score / 100.0 if lic.score > 1 else lic.score,
            provider="scancode",
            reason_code=f"license:{lic.spdx_id or lic.license_expression}",
            source_id=source_id or hit.source_id,
        ))
    return units


# ── Audio 模态：Span / TranscriptUnit → EvidenceUnit ──────


def audio_redaction_span_to_evidence(span: Any) -> EvidenceUnit:
    """将 audio.RedactionSpan 转为 EvidenceUnit。"""
    return EvidenceUnit(
        modality=Modality.AUDIO,
        category="pii",
        sub_category=span.entity_type.lower(),
        audio_span=AudioTimeSpan(
            start_ms=int(span.start_time * 1000),
            end_ms=int(span.end_time * 1000),
            transcript_snippet=span.original_text[:100] if span.original_text else "",
        ),
        confidence=1.0,
        provider="presidio",
        reason_code=f"audio_pii:{span.entity_type.lower()}",
        source_id=span.source_id,
        doc_id=span.unit_id,
    )


# ── Picture 模态：PictureFinding → EvidenceUnit ────────────


def picture_finding_to_evidence(finding: Any) -> EvidenceUnit:
    """将 picture.PictureFinding 转为 EvidenceUnit。"""
    region = None
    if finding.region:
        bbox = finding.region.bbox
        polygon_pts = None
        if finding.region.polygon:
            polygon_pts = finding.region.polygon.points
        region = ImageRegion(
            x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h,
            polygon_points=polygon_pts,
            mask_uri=finding.region.mask_path,
        )
    return EvidenceUnit(
        modality=Modality.PICTURE,
        category=finding.finding_type.value if hasattr(finding.finding_type, "value") else str(finding.finding_type),
        sub_category=finding.category,
        image_region=region,
        confidence=finding.score,
        provider=finding.provider,
        provider_version=getattr(finding, "provider_version", ""),
        threshold_used=getattr(finding, "threshold_used", 0.0),
        explanation=getattr(finding, "explanation", ""),
        reason_code=finding.reason_code,
        metadata=finding.metadata,
    )


# ── Video 模态：VideoFinding → EvidenceUnit ────────────────


def video_finding_to_evidence(finding: Any) -> EvidenceUnit:
    """将 video.VideoFinding 转为 EvidenceUnit。"""
    video_region = None
    if finding.span:
        # 如果有关联的 picture finding 且有 region，提取帧内区域
        frame_region = None
        if finding.picture_finding and finding.picture_finding.region:
            bbox = finding.picture_finding.region.bbox
            frame_region = ImageRegion(x=bbox.x, y=bbox.y, w=bbox.w, h=bbox.h)
        video_region = VideoTimeRegion(
            start_ms=finding.span.start_ms,
            end_ms=finding.span.end_ms,
            frame_id=finding.frame_id,
            region=frame_region,
        )
    return EvidenceUnit(
        modality=Modality.VIDEO,
        category=finding.source_modality,
        sub_category=finding.reason_code,
        video_region=video_region,
        confidence=getattr(finding, "confidence", 0.0),
        provider=getattr(finding, "provider_version", ""),
        explanation=getattr(finding, "explanation", ""),
        reason_code=finding.reason_code,
        metadata=finding.metadata,
    )


# ── 批量转换 ─────────────────────────────────────────────


def convert_text_evidence(
    doc_evidence: Any,
) -> list[EvidenceUnit]:
    """
    将一个 text.DocumentEvidence 的全部 hits 转换为 EvidenceUnit 列表。

    解决"连坐"问题：secret/compliance hits 仅在 doc 的 source_id
    与 hit 的 source_id 匹配时才关联，不再粗暴复制到所有 doc。
    """
    units: list[EvidenceUnit] = []

    # Secret hits — 仅关联同一 source_id 的 hits
    for hit in getattr(doc_evidence, "secret_hits", []):
        if hit.source_id == doc_evidence.source_id:
            units.append(text_secret_hit_to_evidence(hit, source_id=doc_evidence.source_id))

    # Compliance hits
    for hit in getattr(doc_evidence, "compliance_hits", []):
        if hit.source_id == doc_evidence.source_id:
            units.extend(text_compliance_hit_to_evidence(hit, source_id=doc_evidence.source_id))

    # Keyword hits
    for hit in getattr(doc_evidence, "keyword_hits", []):
        units.append(text_keyword_hit_to_evidence(hit, doc_id=doc_evidence.doc_id))

    # Regex hits
    for hit in getattr(doc_evidence, "regex_hits", []):
        units.append(text_regex_hit_to_evidence(hit, doc_id=doc_evidence.doc_id))

    # PII entities
    privacy = getattr(doc_evidence, "privacy", None)
    if privacy:
        prov_name = getattr(privacy, "provider_name", "presidio")
        prov_ver = getattr(privacy, "provider_version", "")
        for entity in privacy.pii_entities:
            units.append(text_pii_entity_to_evidence(
                entity, doc_id=doc_evidence.doc_id,
                provider_name=prov_name, provider_version=prov_ver,
            ))

    # Safety
    safety = getattr(doc_evidence, "safety", None)
    if safety:
        ev = text_safety_to_evidence(safety, doc_id=doc_evidence.doc_id)
        if ev:
            units.append(ev)

    return units


# ── 去重与归并 ────────────────────────────────────────────


def deduplicate_evidence_units(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """
    去重与归并同一对象的多条命中。

    去重策略：
    - 同一 (doc_id, category, sub_category, start, end) 的命中合并为一条
    - 保留 confidence 最高的记录
    """
    seen: dict[str, EvidenceUnit] = {}
    for unit in units:
        span_key = ""
        if unit.text_span:
            span_key = f"{unit.text_span.start}:{unit.text_span.end}"
        elif unit.audio_span:
            span_key = f"{unit.audio_span.start_ms}:{unit.audio_span.end_ms}"
        elif unit.image_region:
            span_key = f"{unit.image_region.x}:{unit.image_region.y}:{unit.image_region.w}:{unit.image_region.h}"
        elif unit.video_region:
            span_key = f"{unit.video_region.start_ms}:{unit.video_region.end_ms}"

        key = f"{unit.doc_id or unit.source_id}|{unit.category}|{unit.sub_category}|{span_key}"

        if key not in seen or unit.confidence > seen[key].confidence:
            seen[key] = unit

    return list(seen.values())


# ── 构建双轨交付物 ────────────────────────────────────────


def build_annotation_package(
    modality: Modality,
    pipeline_run_id: str,
    clean_content_uri: str,
    content_format: str,
    evidence_units: list[EvidenceUnit],
    decision: UnifiedDecision,
    trust_level: TrustLevel,
) -> AnnotationPackage:
    """
    从证据单元构建标注样本包。

    关键设计：原始内容保留可用性，风险区域以 RiskSegment 非侵入式标记。
    标注系统可自行决定展示策略。
    """
    risk_segments: list[RiskSegment] = []
    dispute_segments: list[RiskSegment] = []
    hints: list[str] = []

    for unit in evidence_units:
        snippet = ""
        if unit.text_span:
            snippet = unit.text_span.text_snippet
        elif unit.audio_span:
            snippet = unit.audio_span.transcript_snippet

        segment = RiskSegment(
            evidence_id=unit.evidence_id,
            original_content=snippet,
            replacement_suggestion=f"<{unit.sub_category.upper()}>" if unit.sub_category else "<REDACTED>",
            risk_level="high" if unit.confidence >= 0.8 else "medium" if unit.confidence >= 0.5 else "low",
            category=unit.category,
            sub_category=unit.sub_category,
            annotation_hint=unit.explanation or f"检测到 {unit.category}/{unit.sub_category}",
        )

        # 低置信度的定为争议片段
        if unit.confidence < 0.5:
            segment.is_disputed = True
            segment.dispute_reason = f"置信度较低 ({unit.confidence:.2f})"
            dispute_segments.append(segment)
        else:
            risk_segments.append(segment)

    # 生成标注提示
    categories = set(u.category for u in evidence_units)
    for cat in sorted(categories):
        count = sum(1 for u in evidence_units if u.category == cat)
        hints.append(f"本样本包含 {count} 条 {cat} 类型的风险检测结果")

    # 确定复核优先级
    if any(u.category in ("secret", "safety") for u in evidence_units):
        priority = "critical"
    elif len(evidence_units) > 10:
        priority = "high"
    else:
        priority = "normal"

    return AnnotationPackage(
        modality=modality,
        pipeline_run_id=pipeline_run_id,
        clean_content_uri=clean_content_uri,
        content_format=content_format,
        risk_segments=risk_segments,
        dispute_segments=dispute_segments,
        annotation_hints=hints,
        review_priority=priority,
        decision=decision,
        trust_level=trust_level,
    )


def build_audit_package(
    modality: Modality,
    pipeline_run_id: str,
    evidence_units: list[EvidenceUnit],
    degrade_events: list[DegradeEvent],
    policy_result: Optional[PolicyResult],
    ctx: Optional[PipelineExecutionContext],
) -> AuditPackage:
    """
    从证据链 + 策略结果 + 执行上下文构建审计证据包。

    包含完整的：证据单元、降级事件、评分分解、provider 清单、处理时间线。
    """
    # 收集 provider 版本清单
    provider_manifest: dict[str, str] = {}
    for unit in evidence_units:
        if unit.provider:
            key = f"{unit.category}.{unit.sub_category}"
            provider_manifest[key] = f"{unit.provider}@{unit.provider_version}" if unit.provider_version else unit.provider

    # 处理时间线
    timeline: dict[str, float] = {}
    if ctx:
        for rec in ctx.step_records:
            timeline[rec.step_name] = rec.duration_ms

    # 可信等级
    trust_level = TrustLevel.FULL
    trust_explanation = ""
    if ctx:
        trust_level = TrustEvaluator.evaluate(ctx)
        trust_explanation = TrustEvaluator.build_explanation(ctx)

    # 复核摘要
    review_suggestions: list[str] = []
    if policy_result:
        review_suggestions = policy_result.review_suggestions

    review_summary = ""
    if degrade_events:
        degraded_steps = [e.step_name for e in degrade_events]
        review_summary = f"以下步骤发生了降级：{', '.join(degraded_steps)}。建议人工复核这些步骤的检测结果。"
    elif not evidence_units:
        review_summary = "未检测到任何风险项。"
    else:
        cat_counts: dict[str, int] = {}
        for u in evidence_units:
            cat_counts[u.category] = cat_counts.get(u.category, 0) + 1
        parts = [f"{v} 条 {k}" for k, v in sorted(cat_counts.items())]
        review_summary = f"共检测到 {len(evidence_units)} 条风险项（{', '.join(parts)}）。"

    return AuditPackage(
        modality=modality,
        pipeline_run_id=pipeline_run_id,
        evidence_units=evidence_units,
        degrade_events=degrade_events,
        score_breakdown=policy_result.score_breakdown if policy_result else None,
        policy_result=policy_result,
        provider_manifest=provider_manifest,
        processing_timeline=timeline,
        review_summary=review_summary,
        review_suggestions=review_suggestions,
        trust_level=trust_level,
        trust_explanation=trust_explanation,
    )


def build_release_package(
    modality: Modality,
    pipeline_run_id: str,
    annotation_package: AnnotationPackage,
    audit_package: AuditPackage,
    decision: UnifiedDecision,
    trust_level: TrustLevel,
) -> ReleasePackage:
    """组合标注包 + 审计包为统一发布包。"""
    return ReleasePackage(
        modality=modality,
        pipeline_run_id=pipeline_run_id,
        annotation_package=annotation_package,
        audit_package=audit_package,
        decision=decision,
        trust_level=trust_level,
    )


def build_compliance_output(
    pipeline_run_id: str,
    modality: Modality,
    decision: UnifiedDecision,
    trust_level: TrustLevel,
    release_package: Optional[ReleasePackage],
    degrade_summary: str = "",
    review_suggestions: Optional[list[str]] = None,
    explanation_summary: str = "",
    legacy_decision: Optional[dict[str, Any]] = None,
) -> ComplianceOutput:
    """构建最终的统一输出契约。"""
    return ComplianceOutput(
        pipeline_run_id=pipeline_run_id,
        modality=modality,
        decision=decision,
        trust_level=trust_level,
        release_package=release_package,
        annotation_package_uri=release_package.annotation_package.clean_content_uri if release_package and release_package.annotation_package else "",
        audit_package_uri="",  # 将在落盘后由 pipeline 填充
        degrade_summary=degrade_summary,
        review_suggestions=review_suggestions or [],
        explanation_summary=explanation_summary,
        legacy_decision=legacy_decision,
    )


# ── Decision 映射 ─────────────────────────────────────────


def map_text_decision_to_unified(decision_value: str) -> UnifiedDecision:
    """将 text/audio 的 Decision enum 值映射为 UnifiedDecision。"""
    mapping = {
        "allow": UnifiedDecision.ALLOW,
        "review": UnifiedDecision.REVIEW,
        "quarantine": UnifiedDecision.QUARANTINE,
        "reject": UnifiedDecision.REJECT,
    }
    return mapping.get(decision_value.lower(), UnifiedDecision.REVIEW)


def map_picture_decision_to_unified(decision_value: str) -> UnifiedDecision:
    """将 picture 的 DecisionType enum 值映射为 UnifiedDecision。"""
    mapping = {
        "pass_raw": UnifiedDecision.ALLOW,
        "pass_redacted": UnifiedDecision.QUARANTINE,
        "drop": UnifiedDecision.REJECT,
    }
    return mapping.get(decision_value.lower(), UnifiedDecision.REVIEW)


def map_video_decision_to_unified(decision_value: str) -> UnifiedDecision:
    """将 video 的 VideoDecisionType enum 值映射为 UnifiedDecision。"""
    return map_picture_decision_to_unified(decision_value)

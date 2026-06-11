# ──────────────────────────────────────────────────────────────
# 步骤 F – 隐私检测与脱敏 (Privacy Detection & Redaction)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   使用 Microsoft Presidio 检测文本中的 PII（个人身份信息），
#   并对检测到的 PII 进行脱敏处理（替换为占位符）。
#
# 支持的 PII 类型：
#   PERSON, LOCATION, ORGANIZATION, EMAIL_ADDRESS, PHONE_NUMBER,
#   CREDIT_CARD, ID, DATE_TIME 等
#
# 增强识别：
#   可选加载 HuggingFace NER 模型（如 Meddies/meddies-pii）
#   作为 Presidio 的额外识别器，提高检测准确率。
#
# Fallback 策略：
#   - Presidio 未安装 → 透传原文（不做任何脱敏）
#   - HuggingFace 模型加载失败 → 仅使用 Presidio 内置识别器
#
# 脱敏替换规则：
#   DEFAULT → <REDACTED>
#   PHONE_NUMBER → <PHONE>
#   EMAIL_ADDRESS → <EMAIL>
#   CREDIT_CARD → <CREDIT_CARD>
#   PERSON → <PERSON>
#
# 在流水线中的位置：
#   D(去重) → F(本步骤) → G(安全审核)
#
# 输出产物：
#   privacy_checked.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 F – 隐私检测与脱敏。

使用 Microsoft Presidio 进行 PII 检测和脱敏。
可选加载 HuggingFace NER 模型作为增强识别器。

输出 → privacy_checked.jsonl
"""

from __future__ import annotations

import logging
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import DedupDocument, PIIEntity, PrivacyResult

logger = logging.getLogger(__name__)

# 模块级单例（延迟加载，避免每次调用都初始化）
_analyzer = None
_anonymizer = None


def _get_analyzer(settings: Settings):
    """
    延迟初始化 Presidio AnalyzerEngine。

    首次调用时：
    1. 配置 spaCy NLP 引擎（加载 en_core_web_sm 模型）
    2. 创建 AnalyzerEngine 实例
    3. 可选注册 HuggingFace Transformers NER 识别器

    后续调用直接返回缓存的实例。

    Args:
        settings: 配置对象（包含语言、模型名称等）

    Returns:
        初始化完成的 AnalyzerEngine 实例
    """
    global _analyzer
    if _analyzer is not None:
        return _analyzer

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # 配置 spaCy NLP 引擎
    # 注意：目前仅支持英文 (en_core_web_sm)
    # 中文 PII 检测依赖 Presidio 内置的正则识别器或自定义 NER
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()

    _analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=settings.presidio_languages,
    )

    # 可选：加载 HuggingFace Transformers NER 模型作为增强识别器
    if settings.pii_model_name:
        try:
            _register_transformers_recognizer(_analyzer, settings)
        except Exception as e:
            logger.warning(
                "加载自定义 PII 模型 '%s' 失败: %s。"
                "将仅使用 Presidio 内置识别器。",
                settings.pii_model_name, e,
            )

    logger.info("Presidio AnalyzerEngine 初始化完成")
    return _analyzer


def _register_transformers_recognizer(analyzer, settings: Settings):
    """
    注册 HuggingFace NER 模型作为 Presidio 识别器。

    优先使用 Presidio 内置的 TransformersRecognizer；
    若不可用则手动包装 HuggingFace pipeline。

    Args:
        analyzer: Presidio AnalyzerEngine 实例
        settings: 配置对象
    """
    try:
        from presidio_analyzer.predefined_recognizers import TransformersRecognizer

        # 使用 Presidio 内置的 Transformers 识别器适配器
        transformers_recognizer = TransformersRecognizer(
            model_path=settings.pii_model_name,
            supported_entities=[
                "PERSON", "LOCATION", "ORGANIZATION",
                "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
                "ID", "DATE_TIME",
            ],
            supported_language="en",
        )
        analyzer.registry.add_recognizer(transformers_recognizer)
        logger.info(
            "已注册 TransformersRecognizer，模型: %s",
            settings.pii_model_name,
        )
    except (ImportError, Exception) as e:
        logger.info(
            "TransformersRecognizer 不可用 (%s)，尝试手动注册 NER 包装器", e
        )
        _register_manual_ner(analyzer, settings)


def _register_manual_ner(analyzer, settings: Settings):
    """
    手动包装 HuggingFace NER pipeline 为 Presidio 识别器。

    当 Presidio 内置的 TransformersRecognizer 不可用时，
    手动创建一个 EntityRecognizer 子类来桥接二者。

    Args:
        analyzer: Presidio AnalyzerEngine 实例
        settings: 配置对象
    """
    from presidio_analyzer import EntityRecognizer, RecognizerResult

    class HFNerRecognizer(EntityRecognizer):
        """基于 HuggingFace NER pipeline 的 Presidio 识别器包装类。"""

        def __init__(self, model_name: str):
            super().__init__(
                supported_entities=["PERSON", "LOCATION", "ORGANIZATION"],
                supported_language="en",
                name="HFNerRecognizer",
            )
            # 加载 HuggingFace NER pipeline
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=model_name,
                aggregation_strategy="simple",  # 合并相邻同类实体
            )

        def load(self):
            """Presidio 要求的加载方法（模型已在 __init__ 中加载）。"""
            pass

        def analyze(self, text: str, entities, nlp_artifacts=None):
            """
            使用 HuggingFace NER 分析文本中的实体。

            Args:
                text: 待分析文本
                entities: 需要检测的实体类型列表
                nlp_artifacts: Presidio NLP 分析结果（本识别器不使用）

            Returns:
                RecognizerResult 列表
            """
            results = []
            # 截断文本到 5000 字符以避免 OOM
            ner_results = self.ner_pipeline(text[:5000])

            for ent in ner_results:
                # 将 HuggingFace 的实体标签映射到 Presidio 的实体类型
                entity_type = ent["entity_group"].upper()
                if entity_type in {"PER", "PERSON"}:
                    entity_type = "PERSON"
                elif entity_type in {"LOC", "LOCATION", "GPE"}:
                    entity_type = "LOCATION"
                elif entity_type in {"ORG", "ORGANIZATION"}:
                    entity_type = "ORGANIZATION"
                else:
                    continue  # 跳过未映射的实体类型

                # 如果指定了实体类型过滤，则只返回匹配的类型
                if entities and entity_type not in entities:
                    continue

                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=ent["start"],
                        end=ent["end"],
                        score=float(ent["score"]),
                    )
                )
            return results

    try:
        recognizer = HFNerRecognizer(settings.pii_model_name)
        analyzer.registry.add_recognizer(recognizer)
        logger.info("已注册手动 HF NER 识别器: %s", settings.pii_model_name)
    except Exception as e:
        logger.warning("注册 HF NER 识别器失败: %s", e)


def _get_anonymizer():
    """
    延迟初始化 Presidio AnonymizerEngine。

    AnonymizerEngine 负责对检测到的 PII 实体进行脱敏替换。

    Returns:
        初始化完成的 AnonymizerEngine 实例
    """
    global _anonymizer
    if _anonymizer is not None:
        return _anonymizer
    from presidio_anonymizer import AnonymizerEngine
    _anonymizer = AnonymizerEngine()
    return _anonymizer


def _analyze_and_redact(
    text: str,
    analyzer,
    anonymizer,
    languages: list[str],
    score_threshold: float,
) -> tuple[str, list[PIIEntity]]:
    """
    对单段文本执行 PII 检测 + 脱敏。

    处理步骤：
    1. 对每种语言分别运行 Presidio 分析器
    2. 合并结果并去除重叠（保留高分结果）
    3. 使用匿名器替换检测到的 PII

    Args:
        text: 待处理文本
        analyzer: Presidio AnalyzerEngine
        anonymizer: Presidio AnonymizerEngine
        languages: 分析使用的语言列表
        score_threshold: 最低置信度阈值

    Returns:
        (脱敏后文本, PIIEntity 列表)
    """
    from presidio_anonymizer.entities import OperatorConfig

    # 第 1 步：多语言分析
    all_results = []
    for lang in languages:
        try:
            results = analyzer.analyze(
                text=text,
                language=lang,
                score_threshold=score_threshold,
            )
            all_results.extend(results)
        except Exception as e:
            # 某种语言分析失败不影响其他语言
            logger.debug("语言 %s 分析失败: %s", lang, e)

    # 第 2 步：去重——按分数降序排列，移除重叠区间
    # 当两个检测结果在文本中重叠时，保留置信度更高的
    all_results.sort(key=lambda r: (-r.score, r.start))
    deduped = []
    used_ranges: list[tuple[int, int]] = []
    for r in all_results:
        overlap = False
        for start, end in used_ranges:
            # 检查是否与已保留的结果重叠
            if r.start < end and r.end > start:
                overlap = True
                break
        if not overlap:
            deduped.append(r)
            used_ranges.append((r.start, r.end))

    # 第 3 步：脱敏——将 PII 实体替换为占位符
    if deduped:
        anonymized = anonymizer.anonymize(
            text=text,
            analyzer_results=deduped,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
                "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            },
        )
        redacted_text = anonymized.text
    else:
        redacted_text = text

    # 构建 PIIEntity 列表
    pii_entities = [
        PIIEntity(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            score=round(r.score, 4),
            original_text=text[r.start : r.end][:100],  # 截断到 100 字符
        )
        for r in deduped
    ]

    return redacted_text, pii_entities


# ────────────────────────────────────────────────────────────
# Fallback：Presidio 不可用时的降级方案
# ────────────────────────────────────────────────────────────

def _fallback_scan(documents: list[DedupDocument]) -> list[PrivacyResult]:
    """
    Presidio 不可用时的简单透传 fallback。

    不进行任何 PII 检测或脱敏，直接将原始文本作为输出。
    仅用于开发/测试环境。

    Args:
        documents: 去重后的文档列表

    Returns:
        PrivacyResult 列表（redacted_text = original_text）
    """
    logger.warning("Presidio 不可用 – 文档将原样透传，不进行 PII 脱敏")
    return [
        PrivacyResult(
            doc_id=doc.doc_id,
            original_text=doc.text,
            redacted_text=doc.text,
            pii_entities=[],
            pii_count=0,
        )
        for doc in documents
        if not doc.is_duplicate
    ]


# ────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────

def run(
    documents: list[DedupDocument],
    settings: Settings | None = None,
) -> list[PrivacyResult]:
    """
    执行隐私检测与脱敏。

    对每个非重复文档执行 Presidio PII 检测，并将检测到的
    敏感信息替换为占位符。

    Args:
        documents: 去重后的文档列表（步骤 D 输出）
        settings: 配置对象（可选）

    Returns:
        PrivacyResult 列表，包含原始文本和脱敏后文本
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    # 尝试初始化 Presidio，失败则使用 fallback
    try:
        analyzer = _get_analyzer(settings)
        anonymizer = _get_anonymizer()
    except ImportError:
        return _fallback_scan(documents)

    results: list[PrivacyResult] = []
    for doc in documents:
        # 跳过重复文档
        if doc.is_duplicate:
            continue

        # 执行检测 + 脱敏
        redacted_text, pii_entities = _analyze_and_redact(
            doc.text,
            analyzer,
            anonymizer,
            settings.presidio_languages,
            settings.pii_score_threshold,
        )

        results.append(
            PrivacyResult(
                doc_id=doc.doc_id,
                original_text=doc.text,
                redacted_text=redacted_text,
                pii_entities=pii_entities,
                pii_count=len(pii_entities),
            )
        )

        if pii_entities:
            logger.debug(
                "文档 %s: 检测到 %d 个 PII 实体",
                doc.doc_id, len(pii_entities),
            )

    total_pii = sum(r.pii_count for r in results)
    logger.info(
        "隐私检测完成: 共 %d 个 PII 实体，涉及 %d 个文档",
        total_pii, len(results),
    )
    return results

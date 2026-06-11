# ──────────────────────────────────────────────────────────────
# 步骤 G – 语义安全审核 (Semantic Safety Moderation)
# ──────────────────────────────────────────────────────────────
#
# 功能：
#   对脱敏后的文本进行安全性分类，判断内容的危害等级。
#   使用 Qwen3Guard 模型进行三级分类：
#     Safe (安全) / Controversial (争议) / Unsafe (不安全)
#
# 危害类别检测：
#   violent_content, non_violent_illegal, sexual_content,
#   pii_exposure, suicide_self_harm, unethical_acts,
#   politically_sensitive, copyright_violation, jailbreak_attempt,
#   hate_speech, discrimination
#
# 模型信息：
#   Qwen/Qwen3-Guard-0.6B — 千问安全卫士模型（0.6B 参数量）
#   需要 GPU 运行（推荐 CUDA），CPU 推理速度较慢
#
# Fallback 策略：
#   - Qwen3Guard 不可用（无 GPU / 未安装 transformers）
#     → 使用关键词匹配的 mock 分类器
#
# 在流水线中的位置：
#   F(隐私检测) → G(本步骤) → H(证据聚合)
#
# 输出产物：
#   safety_checked.jsonl
# ──────────────────────────────────────────────────────────────

"""
步骤 G – 语义安全审核。

使用 Qwen3Guard 对脱敏文本进行安全性三级分类。
Fallback 为基于关键词的 mock 分类器。

输出 → safety_checked.jsonl
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from text.config.settings import Settings
from text.models.schemas import PrivacyResult, SafetyLevel, SafetyResult

logger = logging.getLogger(__name__)

# 模块级单例（模型和分词器的延迟加载缓存）
_model = None
_tokenizer = None


# ────────────────────────────────────────────────────────────
# Qwen3Guard 模型加载器
# ────────────────────────────────────────────────────────────

def _load_model(settings: Settings):
    """
    延迟加载 Qwen3Guard 模型和分词器。

    首次调用时从 HuggingFace Hub 下载模型并加载到指定设备。
    后续调用直接返回缓存实例。

    Args:
        settings: 配置对象（包含模型名和设备配置）

    Returns:
        (model, tokenizer) 元组
    """
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # 设备选择：auto 模式下自动检测 CUDA 可用性
    device = settings.qwen_guard_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "正在加载 Qwen3Guard 模型 '%s' 到设备 '%s'...",
        settings.qwen_guard_model, device,
    )

    # 加载分词器
    _tokenizer = AutoTokenizer.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,  # Qwen 模型需要信任远程代码
    )

    # 加载模型
    _model = AutoModelForCausalLM.from_pretrained(
        settings.qwen_guard_model,
        trust_remote_code=True,
        torch_dtype="auto",  # 自动选择精度（fp16/bf16/fp32）
    ).to(device)

    # 设为评估模式（禁用 dropout 等训练时组件）
    _model.eval()

    logger.info("Qwen3Guard 模型加载完成")
    return _model, _tokenizer


# ────────────────────────────────────────────────────────────
# 安全分类逻辑
# ────────────────────────────────────────────────────────────

# 提示词模板：指示模型进行安全分类
_SAFETY_PROMPT_TEMPLATE = """You are a content safety classifier. Classify the following text into one of three categories: Safe, Controversial, or Unsafe.

Also identify any specific harm categories present from this list:
- violent_content
- non_violent_illegal
- sexual_content
- pii_exposure
- suicide_self_harm
- unethical_acts
- politically_sensitive
- copyright_violation
- jailbreak_attempt
- hate_speech
- discrimination

Text to classify:
---
{text}
---

Respond in this exact format:
Safety: <Safe|Controversial|Unsafe>
Categories: <comma-separated list or "none">
"""

# 所有可能的危害类别集合
_HARM_CATEGORIES = {
    "violent_content", "non_violent_illegal", "sexual_content",
    "pii_exposure", "suicide_self_harm", "unethical_acts",
    "politically_sensitive", "copyright_violation", "jailbreak_attempt",
    "hate_speech", "discrimination",
}


def _parse_safety_output(raw_output: str) -> tuple[SafetyLevel, list[str]]:
    """
    解析模型输出，提取安全等级和危害类别。

    解析策略：
    1. 在输出中搜索 "unsafe"/"controversial" 关键词判断安全等级
    2. 匹配所有出现的危害类别名称

    Args:
        raw_output: 模型原始输出文本

    Returns:
        (安全等级, 危害类别列表)
    """
    raw_lower = raw_output.lower()

    # 提取安全等级（注意顺序："unsafe" 必须在 "safe" 之前检查）
    safety_level = SafetyLevel.SAFE
    if "unsafe" in raw_lower:
        safety_level = SafetyLevel.UNSAFE
    elif "controversial" in raw_lower:
        safety_level = SafetyLevel.CONTROVERSIAL

    # 提取危害类别
    categories: list[str] = []
    for cat in _HARM_CATEGORIES:
        if cat.lower() in raw_lower:
            categories.append(cat)

    return safety_level, categories


def _classify_text(
    text: str,
    model,
    tokenizer,
    device: str,
    max_input_len: int = 2048,
) -> tuple[SafetyLevel, list[str], str]:
    """
    使用 Qwen3Guard 对单段文本进行安全分类。

    处理流程：
    1. 构造提示词（截断过长文本）
    2. 使用 chat template 格式化输入
    3. Tokenize 并推理
    4. 解码输出并解析结果

    Args:
        text: 待分类文本
        model: Qwen3Guard 模型
        tokenizer: 分词器
        device: 计算设备
        max_input_len: 最大输入文本长度

    Returns:
        (安全等级, 危害类别列表, 原始输出文本)
    """
    import torch

    # 构造提示词，截断过长文本
    prompt = _SAFETY_PROMPT_TEMPLATE.format(text=text[:max_input_len])

    # 使用 chat template 格式化为模型期望的输入格式
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize 并转移到目标设备
    inputs = tokenizer(formatted, return_tensors="pt").to(device)

    # 推理（无梯度计算）
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,   # 限制输出长度
            do_sample=False,      # 贪心解码（确定性输出）
            temperature=0.0,      # temperature=0 等效于贪心解码
        )

    # 仅解码新生成的 token（排除输入部分）
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 解析输出
    safety_level, categories = _parse_safety_output(raw_output)
    return safety_level, categories, raw_output


# ────────────────────────────────────────────────────────────
# Mock 分类器 fallback（无 GPU / 无模型时使用）
# 基于简单关键词匹配的启发式分类
# ────────────────────────────────────────────────────────────

# 不安全关键词集合（中英文）
_UNSAFE_KEYWORDS = {
    "bomb", "kill", "murder", "terrorism", "exploit", "hack",
    "drug", "cocaine", "heroin", "meth",
    "炸弹", "杀人", "恐怖", "毒品",
}

# 争议关键词集合（中英文）
_CONTROVERSIAL_KEYWORDS = {
    "political", "protest", "rebellion", "censorship",
    "政治", "抗议", "审查",
}


def _mock_classify(text: str) -> tuple[SafetyLevel, list[str], str]:
    """
    基于关键词的 Mock 安全分类器。

    仅用于开发/测试环境，当 Qwen3Guard 不可用时作为降级方案。
    准确率很低，生产环境应使用真实模型。

    Args:
        text: 待分类文本

    Returns:
        (安全等级, 危害类别列表, 分类说明)
    """
    text_lower = text.lower()

    # 检查不安全关键词
    for kw in _UNSAFE_KEYWORDS:
        if kw in text_lower:
            return SafetyLevel.UNSAFE, ["violent_content"], f"mock: 匹配到关键词 '{kw}'"

    # 检查争议关键词
    for kw in _CONTROVERSIAL_KEYWORDS:
        if kw in text_lower:
            return SafetyLevel.CONTROVERSIAL, ["politically_sensitive"], f"mock: 匹配到关键词 '{kw}'"

    # 默认安全
    return SafetyLevel.SAFE, [], "mock: 未发现风险关键词"


# ────────────────────────────────────────────────────────────
# 公共 API
# ────────────────────────────────────────────────────────────

def run(
    privacy_results: list[PrivacyResult],
    settings: Settings | None = None,
) -> list[SafetyResult]:
    """
    执行语义安全审核。

    对每个文档的脱敏文本进行安全性分类。
    优先使用 Qwen3Guard 模型，不可用时回退到 Mock 分类器。

    Args:
        privacy_results: 隐私检测结果列表（步骤 F 输出，包含 redacted_text）
        settings: 配置对象（可选）

    Returns:
        SafetyResult 列表
    """
    if settings is None:
        from text.config.settings import get_settings
        settings = get_settings()

    use_model = settings.qwen_guard_enabled
    model = tokenizer = device = None

    # 尝试加载模型
    if use_model:
        try:
            model, tokenizer = _load_model(settings)
            device = settings.qwen_guard_device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception as e:
            logger.warning(
                "无法加载 Qwen3Guard (%s)；回退到 Mock 分类器", e
            )
            use_model = False

    results: list[SafetyResult] = []
    for pr in privacy_results:
        # 优先使用脱敏后文本，若为空则使用原文
        text = pr.redacted_text or pr.original_text

        # 执行分类
        if use_model and model is not None:
            safety_level, categories, raw = _classify_text(
                text, model, tokenizer, device
            )
        else:
            safety_level, categories, raw = _mock_classify(text)

        # 计算安全评分：Safe=1.0, Controversial=0.5, Unsafe=0.0
        score = 1.0 if safety_level == SafetyLevel.SAFE else (
            0.5 if safety_level == SafetyLevel.CONTROVERSIAL else 0.0
        )

        results.append(
            SafetyResult(
                doc_id=pr.doc_id,
                safety_level=safety_level,
                harm_categories=categories,
                raw_output=raw[:500],  # 截断模型输出
                score=score,
            )
        )

        # 仅对非安全文档输出日志
        if safety_level != SafetyLevel.SAFE:
            logger.debug(
                "文档 %s: safety=%s, categories=%s",
                pr.doc_id, safety_level.value, categories,
            )

    # 输出统计
    unsafe_count = sum(1 for r in results if r.safety_level == SafetyLevel.UNSAFE)
    controv_count = sum(1 for r in results if r.safety_level == SafetyLevel.CONTROVERSIAL)
    logger.info(
        "安全审核完成: %d 个文档（%d 不安全，%d 争议）",
        len(results), unsafe_count, controv_count,
    )
    return results

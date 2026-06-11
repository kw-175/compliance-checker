# ──────────────────────────────────────────────────────────────
# 文本数据合规检测模块 (Text Data Compliance Checker Module)
# ──────────────────────────────────────────────────────────────
#
# 本模块是一个综合性的文本数据合规检测流水线，集成了以下十个工具：
#   - TruffleHog    ：密钥/凭证泄露扫描
#   - ScanCode      ：开源许可证与版权合规扫描
#   - DataTrove     ：文本提取（HTML/PDF/纯文本）
#   - Duplodocus    ：去重（当前使用 datasketch MinHash LSH 作为 fallback）
#   - FlashText2    ：高速关键词匹配
#   - Hyperscan     ：高性能正则表达式扫描（fallback 使用 Python re）
#   - Presidio      ：PII（个人身份信息）检测与脱敏
#   - Qwen3Guard    ：安全内容审核（Safe/Controversial/Unsafe 三级分类）
#   - OPA           ：基于策略的合规决策引擎
#   - OpenLineage   ：数据血缘与审计追踪
#
# 流水线步骤 A → J：
#   A: 输入接入  →  B1: 来源分类  →  B2: 原始对象扫描(密钥/许可证)
#   →  C: 文本提取清洗  →  D: 去重  →  E1: 规则扫描(关键词/正则)
#   →  F: 隐私检测脱敏  →  G: 安全审核  →  H: 证据聚合
#   →  I: 策略决策  →  J: 血缘审计
#
# 服务入口：FastAPI 微服务 (server.py)
# 编排入口：CompliancePipeline (pipeline.py)
# ──────────────────────────────────────────────────────────────

"""
Text Data Compliance Checker Module

A comprehensive text data compliance checking pipeline that integrates
TruffleHog, ScanCode, DataTrove, Duplodocus, FlashText2, Hyperscan,
Presidio, Qwen3Guard, OPA, and OpenLineage.
"""

# 模块版本号，遵循语义化版本规范 (SemVer)
__version__ = "0.1.0"

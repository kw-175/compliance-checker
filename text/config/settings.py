# ──────────────────────────────────────────────────────────────
# 全局配置模块 (Global Configuration)
# ──────────────────────────────────────────────────────────────
#
# 使用 Pydantic BaseSettings 实现配置管理，支持两种加载方式：
#   1. 环境变量：以 COMPLIANCE_ 为前缀（如 COMPLIANCE_WORK_DIR）
#   2. .env 文件：自动从项目根目录加载
#
# 配置内容涵盖：
#   - 工具路径（TruffleHog、ScanCode、Duplodocus 等二进制路径）
#   - 模型配置（Qwen3Guard 模型名、设备选择）
#   - 阈值参数（去重阈值、PII 评分阈值等）
#   - 服务配置（OPA 地址、OpenLineage 地址、服务端口）
#   - 并发配置（线程池大小）
# ──────────────────────────────────────────────────────────────

"""
全局配置模块。

使用 Pydantic BaseSettings 从环境变量 / .env 文件加载所有配置项，
为合规检测流水线的各个步骤提供统一的配置入口。
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    合规检测流水线的全局配置类。

    所有配置项均可通过环境变量覆盖，环境变量名格式为：
    COMPLIANCE_{字段名大写}，例如 COMPLIANCE_WORK_DIR。

    Attributes:
        work_dir: 中间产物和最终输出的根目录
        trufflehog_bin: TruffleHog v3 二进制文件路径
        scancode_bin: ScanCode-toolkit 二进制文件路径
        dedup_threshold: MinHash 近似去重的 Jaccard 相似度阈值
        ...（其余字段见各步骤配置段）
    """

    # ─── Pydantic Settings 元数据配置 ─────────────────────
    # env_prefix: 环境变量前缀，所有配置对应的环境变量以 COMPLIANCE_ 开头
    # env_file: 指定 .env 文件路径
    # extra: "ignore" 表示忽略环境中多余的变量，不报错
    model_config = {"env_prefix": "COMPLIANCE_", "env_file": ".env", "extra": "ignore"}

    # ─── 通用路径配置 ─────────────────────────────────────
    # 所有步骤的中间产物（JSONL/JSON 文件）均输出到此目录下的 {run_id}/ 子目录
    work_dir: Path = Field(
        default=Path("./compliance_output"),
        description="Root directory for intermediate / final JSONL outputs",
    )

    # ─── 步骤 A: 输入接入 ─────────────────────────────────
    # 步骤 A 不需要额外配置，输入路径在运行时传入

    # ─── 步骤 B2a: TruffleHog 密钥扫描 ───────────────────
    # TruffleHog 是用 Go 编写的密钥泄露检测工具
    # 安装方式见：https://github.com/trufflesecurity/trufflehog#installation
    trufflehog_bin: str = Field(
        default="trufflehog",
        description="Path to the TruffleHog v3 binary",
    )

    # ─── 步骤 B2b: ScanCode 许可证扫描 ───────────────────
    # ScanCode-toolkit 用于检测代码中的开源许可证和版权声明
    # 安装：pip install scancode-toolkit
    scancode_bin: str = Field(
        default="scancode",
        description="Path to the ScanCode-toolkit binary",
    )

    # ─── 步骤 C: DataTrove 文本提取 ──────────────────────
    # 文本提取后端选择：trafilatura（默认）或 readability
    # Trafilatura 擅长从 HTML 中提取正文内容
    datatrove_extract_backend: str = Field(
        default="trafilatura",
        description="Text extraction backend: trafilatura | readability",
    )

    # ─── 步骤 D: Duplodocus / datasketch 去重 ────────────
    # Duplodocus 是 Rust 实现的 CLI 去重工具（当前未集成，使用 Python datasketch 替代）
    duplodocus_bin: str = Field(
        default="duplodocus",
        description="Path to the Duplodocus binary (Rust CLI)",
    )
    # MinHash LSH 近似去重的 Jaccard 相似度阈值
    # 值越大表示要求越高的相似度才判定为重复
    dedup_threshold: float = Field(
        default=0.8,
        description="MinHash Jaccard threshold for near-duplicate detection",
    )
    # MinHash 排列数，越大精度越高但内存和计算开销也越大
    dedup_num_perm: int = Field(
        default=128,
        description="Number of permutations for MinHash",
    )

    # ─── 步骤 E1a: FlashText2 关键词扫描 ─────────────────
    # 关键词列表文件路径，每行一个关键词，以 # 开头的行为注释
    keywords_file: Path = Field(
        default=Path(__file__).resolve().parent / "keywords.txt",
        description="Newline-delimited keyword list for FlashText2",
    )

    # ─── 步骤 E1b: Hyperscan 正则扫描 ────────────────────
    # 正则模式定义文件（YAML 格式），键为模式名称，值为正则表达式
    patterns_file: Path = Field(
        default=Path(__file__).resolve().parent / "patterns.yaml",
        description="YAML file mapping pattern-name → regex",
    )

    # ─── 步骤 F: Presidio 隐私检测 ───────────────────────
    # Presidio 支持的语言列表（需要对应的 NLP 模型支持）
    # 注意：目前仅加载了 en_core_web_sm，中文检测依赖 fallback 或自定义 NER
    presidio_languages: list[str] = Field(
        default=["en"],
        description="Languages for Presidio NER",
    )
    # 可选的 HuggingFace NER 模型，用于增强 PII 检测
    # 设为 None 可禁用，仅使用 Presidio 内置识别器
    pii_model_name: Optional[str] = Field(
        default="Meddies/meddies-pii",
        description="HuggingFace NER model for custom PII detection",
    )
    # PII 实体置信度分数阈值，低于此值的实体将被忽略
    pii_score_threshold: float = Field(
        default=0.35,
        description="Minimum score to accept a PII entity",
    )

    # ─── 步骤 G: Qwen3Guard 安全审核 ─────────────────────
    # Qwen3Guard 是千问系列的内容安全分类模型
    # 需要 GPU 运行，若不可用则回退到基于关键词的 mock 分类器
    qwen_guard_model: str = Field(
        default="Qwen/Qwen3-Guard-0.6B",
        description="HuggingFace model ID for Qwen3Guard",
    )
    # 模型运行设备：auto（自动检测）、cpu、cuda、cuda:0 等
    qwen_guard_device: str = Field(
        default="auto",
        description="Device for Qwen3Guard: auto | cpu | cuda | cuda:0",
    )
    # 是否启用 Qwen3Guard 模型；设为 False 时使用 mock 分类器（无需 GPU）
    qwen_guard_enabled: bool = Field(
        default=True,
        description="Set False to use mock safety scorer (no GPU needed)",
    )

    # ─── 步骤 I: OPA 策略决策 ────────────────────────────
    # OPA (Open Policy Agent) 是外部策略引擎，通过 REST API 进行策略评估
    # 需要预先启动 OPA server 或使用 Docker
    opa_url: str = Field(
        default="http://localhost:8181",
        description="Base URL of the running OPA server",
    )
    # OPA 策略评估的 REST 路径
    opa_policy_path: str = Field(
        default="v1/data/compliance/decision",
        description="OPA REST path for policy evaluation",
    )
    # 是否启用 OPA；设为 False 时使用本地 Python 规则引擎作为 fallback
    opa_enabled: bool = Field(
        default=True,
        description="Set False to use local rule-engine fallback",
    )

    # ─── 步骤 J: OpenLineage 血缘审计 ───────────────────
    # OpenLineage 用于记录每个步骤的数据血缘信息
    # 设为 None 使用 ConsoleTransport（输出到控制台）
    # 设为 URL（如 http://marquez:5000）则使用 HttpTransport 推送到 Marquez 后端
    openlineage_url: Optional[str] = Field(
        default=None,
        description="Marquez / OpenLineage backend URL.  None → ConsoleTransport",
    )
    # OpenLineage 命名空间，用于标识本流水线的所有 Job
    openlineage_namespace: str = Field(
        default="compliance-checker",
        description="OpenLineage namespace for all jobs",
    )

    # ─── 服务配置 ─────────────────────────────────────────
    # FastAPI 微服务的监听地址和端口
    server_host: str = "0.0.0.0"  # 默认监听所有网络接口
    server_port: int = 8000       # 默认端口 8000

    # ─── 并发配置 ─────────────────────────────────────────
    # 并行步骤（B2a/B2b、E1a/E1b）使用的最大线程/进程数
    max_workers: int = Field(
        default=4,
        description="Max threads/processes for parallel steps",
    )


def get_settings() -> Settings:
    """
    获取配置实例。

    返回一个新的 Settings 实例。每次调用都会重新从环境变量/文件加载配置，
    适用于需要动态更新配置的场景。

    Returns:
        Settings: 当前环境的配置对象
    """
    return Settings()

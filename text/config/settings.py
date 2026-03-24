"""
Global configuration for the text compliance checker.
Uses Pydantic BaseSettings for env-var / .env file loading.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All configurable knobs for the compliance pipeline."""

    model_config = {"env_prefix": "COMPLIANCE_", "env_file": ".env", "extra": "ignore"}

    # ── Paths ──────────────────────────────────────────────
    work_dir: Path = Field(
        default=Path("./compliance_output"),
        description="Root directory for intermediate / final JSONL outputs",
    )

    # ── Step A: Source Intake ──────────────────────────────
    # (no special config – input paths come at runtime)

    # ── Step B2a: TruffleHog ──────────────────────────────
    trufflehog_bin: str = Field(
        default="trufflehog",
        description="Path to the TruffleHog v3 binary",
    )

    # ── Step B2b: ScanCode ────────────────────────────────
    scancode_bin: str = Field(
        default="scancode",
        description="Path to the ScanCode-toolkit binary",
    )

    # ── Step C: DataTrove ─────────────────────────────────
    datatrove_extract_backend: str = Field(
        default="trafilatura",
        description="Text extraction backend: trafilatura | readability",
    )

    # ── Step D: Duplodocus / fallback ─────────────────────
    duplodocus_bin: str = Field(
        default="duplodocus",
        description="Path to the Duplodocus binary (Rust CLI)",
    )
    dedup_threshold: float = Field(
        default=0.8,
        description="MinHash Jaccard threshold for near-duplicate detection",
    )
    dedup_num_perm: int = Field(
        default=128,
        description="Number of permutations for MinHash",
    )

    # ── Step E1a: FlashText2 ──────────────────────────────
    keywords_file: Path = Field(
        default=Path(__file__).resolve().parent / "keywords.txt",
        description="Newline-delimited keyword list for FlashText2",
    )

    # ── Step E1b: Hyperscan ───────────────────────────────
    patterns_file: Path = Field(
        default=Path(__file__).resolve().parent / "patterns.yaml",
        description="YAML file mapping pattern-name → regex",
    )

    # ── Step F: Presidio ──────────────────────────────────
    presidio_languages: list[str] = Field(
        default=["en", "zh"],
        description="Languages for Presidio NER",
    )
    pii_model_name: Optional[str] = Field(
        default="Meddies/meddies-pii",
        description="HuggingFace NER model for custom PII detection",
    )
    pii_score_threshold: float = Field(
        default=0.35,
        description="Minimum score to accept a PII entity",
    )

    # ── Step G: Qwen3Guard ────────────────────────────────
    qwen_guard_model: str = Field(
        default="Qwen/Qwen3-Guard-0.6B",
        description="HuggingFace model ID for Qwen3Guard",
    )
    qwen_guard_device: str = Field(
        default="auto",
        description="Device for Qwen3Guard: auto | cpu | cuda | cuda:0",
    )
    qwen_guard_enabled: bool = Field(
        default=True,
        description="Set False to use mock safety scorer (no GPU needed)",
    )

    # ── Step I: OPA ───────────────────────────────────────
    opa_url: str = Field(
        default="http://localhost:8181",
        description="Base URL of the running OPA server",
    )
    opa_policy_path: str = Field(
        default="v1/data/compliance/decision",
        description="OPA REST path for policy evaluation",
    )
    opa_enabled: bool = Field(
        default=True,
        description="Set False to use local rule-engine fallback",
    )

    # ── Step J: OpenLineage ───────────────────────────────
    openlineage_url: Optional[str] = Field(
        default=None,
        description="Marquez / OpenLineage backend URL.  None → ConsoleTransport",
    )
    openlineage_namespace: str = Field(
        default="compliance-checker",
        description="OpenLineage namespace for all jobs",
    )

    # ── Server ────────────────────────────────────────────
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # ── Concurrency ───────────────────────────────────────
    max_workers: int = Field(
        default=4,
        description="Max threads/processes for parallel steps",
    )


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()

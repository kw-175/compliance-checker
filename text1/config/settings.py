from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "COMPLIANCE_", "env_file": ".env", "extra": "ignore"}

    work_dir: Path = Field(
        default=Path("./compliance_output/text"),
        description="Directory for JSONL artifacts produced by the text workflow.",
    )
    policy_version: str = "text-policy-2026.04"

    content_rules_path: Path = Field(
        default=Path(__file__).resolve().parent / "content_rules.yaml",
        description="Structured content safety rules for the detection layer.",
    )
    pii_rules_path: Path = Field(
        default=Path(__file__).resolve().parent / "pii_rules.yaml",
        description="Structured privacy and PII rules for the detection layer.",
    )

    enable_hard_case_adjudication: bool = True
    hard_case_model_name: str = "Qwen3.5-35B-A3B-FP8"
    hard_case_prompt_version: str = "qwen-hard-case-v1"
    hard_case_local_model_path: str = ""
    hard_case_endpoint: str = ""
    hard_case_timeout_seconds: int = 45
    hard_case_max_chars: int = 3500

    safety_hard_case_score_floor: float = 0.35
    safety_hard_case_score_ceiling: float = 0.85
    privacy_hard_case_score_floor: float = 0.45
    privacy_combination_threshold: int = 2

    preserve_original_text: bool = True
    max_text_chars_per_document: int = 100_000

    server_host: str = "0.0.0.0"
    server_port: int = 8000


def get_settings() -> Settings:
    return Settings()

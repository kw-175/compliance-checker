from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "COMPLIANCE_", "env_file": ".env", "extra": "ignore"}

    work_dir: Path = Field(
        default=Path("./compliance_output/text"),
        description="Directory for JSONL artifacts produced by the text workflow.",
    )
    upload_dir: Path = Field(
        default=Path("./compliance_uploads/text"),
        description="Directory for files uploaded by the dataset platform.",
    )
    policy_version: str = "text-policy-2026.04"
    platform_api_key: str = Field(
        default="",
        description="Optional bearer/API key required by the dataset platform integration endpoints.",
    )

    content_rules_path: Path = Field(
        default=Path(__file__).resolve().parent / "content_rules.yaml",
        description="Structured content safety rules for the detection layer.",
    )
    content_safety_policies_path: Path = Field(
        default=Path(__file__).resolve().parent / "content_safety_policies.yaml",
        description="Structured policy decisions for the content safety governance layer.",
    )
    pii_rules_path: Path = Field(
        default=Path(__file__).resolve().parent / "pii_rules.yaml",
        description="Structured privacy and PII rules for the detection layer.",
    )
    privacy_entity_catalog_path: Path = Field(
        default=Path(__file__).resolve().parent / "privacy_entity_catalog.yaml",
        description="Privacy entity sub-operator catalog used by the API privacy governance layer.",
    )
    privacy_policies_path: Path = Field(
        default=Path(__file__).resolve().parent / "privacy_policies.yaml",
        description="Structured privacy governance policies for sensitivity and downstream admission.",
    )

    enable_qwen3guard: bool = False
    qwen3guard_model_name: str = "Qwen3Guard-Gen-0.6B"
    qwen3guard_endpoint: str = ""
    qwen3guard_timeout_seconds: int = 30
    qwen3guard_max_chars: int = 8000

    enable_presidio: bool = False
    presidio_analyzer_endpoint: str = ""
    presidio_anonymizer_endpoint: str = ""
    presidio_language: str = "auto"
    presidio_supported_languages: str = "en,zh"
    presidio_language_fallback: str = "en"
    presidio_timeout_seconds: int = 30
    presidio_score_threshold: float = 0.45

    enable_hard_case_adjudication: bool = True
    hard_case_model_name: str = "Qwen3.5-9B"
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

    downstream_annotation_base_url: str = "http://127.0.0.1:8100"
    downstream_annotation_dataset_name: str = ""
    downstream_annotation_text_mode: str = "graded"
    downstream_annotation_include_hold: bool = True
    downstream_annotation_trusted_original_for_hold: bool = False

    compliance_provider_mode: str = "auto"
    local_compliance_base_url: str = ""
    local_compliance_api_key: str = ""
    local_compliance_model: str = ""
    local_compliance_timeout_seconds: int = 120
    local_compliance_temperature: float = 0.0
    local_compliance_max_chars: int = 12000
    local_compliance_max_tokens: int = 4096
    local_compliance_source_tool_prefix: str = "local_model"
    local_document_context_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "document_context_local_v1.md",
        description="Prompt file used by the local model document-context builder.",
    )
    local_privacy_fragment_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "privacy_fragment_adjudication_local_v1.md",
        description="Prompt file used by the local privacy fragment adjudicator.",
    )
    local_content_fragment_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_fragment_adjudication_local_v1.md",
        description="Prompt file used by the local content fragment adjudicator.",
    )
    local_content_localization_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_fragment_localization_local_v1.md",
        description="Prompt file used by the local content fragment localizer.",
    )
    local_privacy_document_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "privacy_document_assessment_local_v1.md",
        description="Prompt file used by the local privacy document assessor.",
    )
    local_content_document_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_document_assessment_local_v1.md",
        description="Prompt file used by the local content document assessor.",
    )

    api_compliance_base_url: str = ""
    api_compliance_api_key: str = ""
    api_compliance_model: str = ""
    api_compliance_timeout_seconds: int = 60
    api_compliance_temperature: float = 0.0
    api_compliance_max_chars: int = 8000
    api_compliance_max_tokens: int = 2048
    api_compliance_source_tool_prefix: str = "openai_compatible_api"
    api_content_safety_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_safety_api_v1.md",
        description="Prompt file used by the API content safety operator.",
    )
    api_content_safety_unified_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_safety_unified_api_v2.md",
        description="Prompt file used by the unified API content safety detector.",
    )
    api_content_safety_policy_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_safety_policy_api_v2.md",
        description="Prompt file used by the API content safety policy adjudicator.",
    )
    enable_content_safety_semantic_adjudication: bool = True
    api_content_safety_semantic_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "content_safety_semantic_adjudication_api_v1.md",
        description="Prompt file used by the API semantic adjudicator for contextual content-safety findings.",
    )
    content_safety_operator_dir: Path = Field(
        default=Path(__file__).resolve().parent / "content_safety_operators",
        description="Directory of content-safety sub-operator asset definitions.",
    )
    content_safety_operator_ids: list[str] = Field(
        default_factory=list,
        description="Optional content-safety sub-operator IDs selected by the platform.",
    )
    content_safety_labels_path: Path = Field(
        default=Path(__file__).resolve().parent / "content_safety_labels.yaml",
        description="Risk-label catalog used by the API content safety operator.",
    )
    content_safety_target_labels: list[str] = Field(
        default_factory=list,
        description="Optional content-safety risk labels selected by the platform.",
    )
    content_safety_custom_policy: str = Field(
        default="",
        description="Optional user-provided content-safety policy text.",
    )
    content_safety_custom_policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured custom content-safety policy passed by the platform.",
    )
    content_safety_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional scene metadata passed to the content-safety API operator.",
    )
    content_safety_training_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional downstream training context for content-safety governance hints.",
    )
    api_privacy_detection_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "privacy_detection_api_v1.md",
        description="Prompt file used by the API privacy detection operator.",
    )
    privacy_operator_ids: list[str] = Field(
        default_factory=list,
        description="Optional privacy sub-operator IDs selected by the platform.",
    )
    privacy_target_types: list[str] = Field(
        default_factory=list,
        description="Optional privacy entity types selected by the platform.",
    )
    privacy_custom_policy: str = Field(
        default="",
        description="Optional user-provided privacy governance policy text.",
    )
    privacy_custom_policy_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured custom privacy policy passed by the platform.",
    )
    privacy_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional scene/source metadata passed to the privacy governance layer.",
    )
    privacy_training_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional downstream training context for privacy governance decisions.",
    )
    api_hard_case_prompt_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "prompts" / "hard_case_adjudication_api_v1.md",
        description="Prompt file used by the API hard-case adjudication operator.",
    )
    api_server_port: int = 19002
    content_candidate_window_max_chars: int = 360
    content_candidate_window_overlap_chars: int = 80

    server_host: str = "0.0.0.0"
    server_port: int = 9000


def get_settings() -> Settings:
    return Settings()

"""Enum definitions for the video compliance blueprint."""

from __future__ import annotations

from enum import Enum


class VideoRouteType(str, Enum):
    """High-level route for video content."""

    SCREENCAST = "screencast"
    NATURAL = "natural"
    MIXED = "mixed"


class VideoJobStatus(str, Enum):
    """Lifecycle states for a video compliance job."""

    CREATED = "created"
    PREPROCESSING = "preprocessing"
    SAMPLING = "sampling"
    DETECTING = "detecting"
    TRACKING = "tracking"
    AUDIO_PROCESSING = "audio_processing"
    POLICY_EVALUATING = "policy_evaluating"
    RENDERING = "rendering"
    DONE = "done"
    DROPPED = "dropped"
    FAILED = "failed"


class VideoDecisionType(str, Enum):
    """Final compliance decision."""

    PASS_RAW = "pass_raw"
    PASS_REDACTED = "pass_redacted"
    DROP = "drop"


class VideoGovernanceDecision(str, Enum):
    """Use-aware governance decision for cleaned video assets."""

    ALLOW = "allow"
    ALLOW_WITH_RISK_LABELS = "allow_with_risk_labels"
    RESTRICTED = "restricted"
    REVIEW_REQUIRED = "review_required"
    TRANSFORM_REQUIRED = "transform_required"
    REJECT = "reject"


class VideoRiskSeverity(str, Enum):
    """Normalized risk severity for video annotations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

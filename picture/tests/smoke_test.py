"""Smoke tests for the picture compliance pipeline."""
from __future__ import annotations

from pathlib import Path

import pytest

from picture.application.use_cases import process_image
from picture.domain.enums import DecisionType, JobStatus
from picture.infra.config import PictureSettings


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_smoke_all_processing_chains(tmp_path: Path) -> None:
    settings = PictureSettings(
        work_dir=tmp_path / "work",
        storage_base_path=tmp_path / "storage",
        policy_config_dir=Path(__file__).resolve().parent.parent / "configs",
    )
    fixtures = Path(__file__).resolve().parent / "fixtures"

    for hint, fixture in [
        ("document", "sample_document.png"),
        ("natural", "sample_natural.png"),
        ("mixed", "sample_mixed.png"),
    ]:
        job = process_image(
            str(fixtures / fixture),
            settings=settings,
            options={"route_hint": hint},
        )

        assert job.status in {JobStatus.DONE, JobStatus.DROPPED}
        assert job.policy_result is not None
        assert job.report_uri is not None

    unsafe_job = process_image(
        str(fixtures / "sample_unsafe_explicit.png"),
        settings=settings,
        options={"route_hint": "natural"},
    )

    assert unsafe_job.status == JobStatus.DROPPED
    assert unsafe_job.policy_result is not None
    assert unsafe_job.policy_result.decision == DecisionType.DROP

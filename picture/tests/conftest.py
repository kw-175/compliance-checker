from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("PICTURE_RUN_INTEGRATION_TESTS", "").lower() in {"1", "true", "yes"}:
        return

    skip_integration = pytest.mark.skip(
        reason="set PICTURE_RUN_INTEGRATION_TESTS=1 to run slow picture integration tests"
    )
    for item in items:
        if "integration" in item.keywords or "slow" in item.keywords:
            item.add_marker(skip_integration)

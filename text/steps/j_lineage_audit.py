"""
Step J – Lineage & Audit (OpenLineage)

Records RunEvent (START / COMPLETE / FAIL) for each pipeline step using
the OpenLineage Python client.

Transports:
  - ConsoleTransport  (default, for development)
  - HttpTransport     (production → Marquez backend)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from text.config.settings import Settings

logger = logging.getLogger(__name__)

# Module-level singleton
_client = None
_namespace = "compliance-checker"


def _get_client(settings: Settings):
    """Lazy-init the OpenLineageClient."""
    global _client, _namespace
    if _client is not None:
        return _client

    try:
        from openlineage.client import OpenLineageClient
        from openlineage.client.transport.console import ConsoleTransport
        from openlineage.client.transport.http import HttpConfig, HttpTransport

        _namespace = settings.openlineage_namespace

        if settings.openlineage_url:
            transport = HttpTransport(
                HttpConfig.from_dict({"url": settings.openlineage_url})
            )
            logger.info("OpenLineage: using HttpTransport → %s", settings.openlineage_url)
        else:
            transport = ConsoleTransport()
            logger.info("OpenLineage: using ConsoleTransport (stdout)")

        _client = OpenLineageClient(transport=transport)
        return _client

    except ImportError:
        logger.warning(
            "openlineage-python not installed; lineage events will be logged only.  "
            "pip install openlineage-python"
        )
        return None


class LineageTracker:
    """
    Convenience wrapper to emit OpenLineage RunEvents for each pipeline step.

    Usage::

        tracker = LineageTracker(settings)
        run_id = tracker.start_step("step_c_text_extract")
        # ... do work ...
        tracker.complete_step("step_c_text_extract", run_id)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = _get_client(settings)
        self._runs: dict[str, str] = {}

    def start_step(
        self,
        step_name: str,
        inputs: list[dict[str, str]] | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> str:
        """Emit a START RunEvent and return the run_id."""
        run_id = uuid.uuid4().hex
        self._runs[step_name] = run_id

        if self.client is None:
            logger.info("[lineage] START  %s (run_id=%s)", step_name, run_id[:8])
            return run_id

        try:
            from openlineage.client.run import (
                InputDataset,
                Job,
                OutputDataset,
                Run,
                RunEvent,
                RunState,
            )

            event = RunEvent(
                eventType=RunState.START,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                inputs=[
                    InputDataset(namespace=_namespace, name=i.get("name", ""))
                    for i in (inputs or [])
                ],
                outputs=[
                    OutputDataset(namespace=_namespace, name=o.get("name", ""))
                    for o in (outputs or [])
                ],
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[lineage] Emitted START for %s", step_name)
        except Exception as e:
            logger.warning("[lineage] Failed to emit START for %s: %s", step_name, e)

        return run_id

    def complete_step(
        self,
        step_name: str,
        run_id: str | None = None,
        outputs: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit a COMPLETE RunEvent."""
        run_id = run_id or self._runs.get(step_name, uuid.uuid4().hex)

        if self.client is None:
            logger.info("[lineage] COMPLETE %s (run_id=%s)", step_name, run_id[:8])
            return

        try:
            from openlineage.client.run import (
                Job,
                OutputDataset,
                Run,
                RunEvent,
                RunState,
            )

            event = RunEvent(
                eventType=RunState.COMPLETE,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                outputs=[
                    OutputDataset(namespace=_namespace, name=o.get("name", ""))
                    for o in (outputs or [])
                ],
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[lineage] Emitted COMPLETE for %s", step_name)
        except Exception as e:
            logger.warning("[lineage] Failed to emit COMPLETE for %s: %s", step_name, e)

    def fail_step(
        self,
        step_name: str,
        run_id: str | None = None,
        error: str = "",
    ) -> None:
        """Emit a FAIL RunEvent."""
        run_id = run_id or self._runs.get(step_name, uuid.uuid4().hex)

        if self.client is None:
            logger.info("[lineage] FAIL %s (run_id=%s): %s", step_name, run_id[:8], error)
            return

        try:
            from openlineage.client.run import (
                Job,
                Run,
                RunEvent,
                RunState,
            )

            event = RunEvent(
                eventType=RunState.FAIL,
                eventTime=datetime.now(timezone.utc).isoformat(),
                run=Run(runId=run_id),
                job=Job(namespace=_namespace, name=step_name),
                producer=f"compliance-checker/{step_name}",
            )
            self.client.emit(event)
            logger.debug("[lineage] Emitted FAIL for %s: %s", step_name, error)
        except Exception as e:
            logger.warning("[lineage] Failed to emit FAIL for %s: %s", step_name, e)

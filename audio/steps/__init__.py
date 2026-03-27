"""
Audio pipeline step helpers.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_command(
    command: list[str],
    timeout: int = 300,
    ok_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str] | None:
    """Execute an external command with consistent logging and graceful failure."""

    if not command:
        return None
    binary = shutil.which(command[0]) or command[0]
    command = [binary, *command[1:]]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        logger.warning("Binary not available: %s", command[0])
        return None
    except subprocess.TimeoutExpired:
        logger.warning("Command timed out: %s", " ".join(command))
        return None

    if result.returncode not in ok_returncodes:
        logger.warning("Command failed (%s): %s", result.returncode, " ".join(command))
        if result.stderr:
            logger.debug("stderr: %s", result.stderr)
        return None
    return result


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        unquoted = value[1:-1]
        return bytes(unquoted, "utf-8").decode("unicode_escape")
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        key, sep, remainder = stripped.partition(":")
        if not sep:
            continue

        key = key.strip()
        value = remainder.strip()
        if not value:
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
            continue
        parent[key] = _parse_scalar(value)

    return root


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML unavailable, using simple YAML parser for %s", path)
        return _load_simple_yaml(path)

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL row in %s", path)
    return rows

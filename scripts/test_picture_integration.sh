#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
PYTHON="${PYTHON:-$PROJECT/.venvs/picture/bin/python}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"

cd "$PROJECT"
PICTURE_RUN_INTEGRATION_TESTS=1 \
PYTHONPATH="$PROJECT" \
timeout "$TIMEOUT_SECONDS" "$PYTHON" -m pytest -q -m "integration or slow" picture/tests "$@"

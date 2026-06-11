#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="${PROJECT:-/data/kw/compliance-checker}"
PYTHON="${PYTHON:-$PROJECT/.venvs/picture/bin/python}"

cd "$PROJECT"
PYTHONPATH="$PROJECT" "$PYTHON" -m pytest -q picture/tests "$@"

#!/usr/bin/env bash
# ==============================================================================
# Launch Frigate Model Training Studio Web Interface
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

if [[ -x "${SCRIPT_DIR}/venv-torq-312/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/venv-torq-312/bin/python"
elif [[ -x "${SCRIPT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${SCRIPT_DIR}/venv/bin/python"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/web_ui.py" "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WITH_GPU="/store/store5/software/simple-gpu-schedule/with-gpu"
GPU_POOL="${GPU_POOL:-1,2}"

cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/.hf-cache}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export PYTHONUNBUFFERED=1

VENV_DIR="${MOSS_DEMO_VENV:-$ROOT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python at $PYTHON_BIN. Create the venv with the setup command in README.md." >&2
  exit 1
fi

run_demo() {
  "$PYTHON_BIN" "$ROOT_DIR/scripts/moss_streaming_demo.py" "$@"
}

if [[ "${RUN_WITH_GPU_SCHEDULER:-1}" == "1" ]]; then
  exec "$WITH_GPU" "$GPU_POOL" -- "$PYTHON_BIN" "$ROOT_DIR/scripts/moss_streaming_demo.py" "$@"
fi

run_demo "$@"

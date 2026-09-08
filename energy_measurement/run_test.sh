#!/usr/bin/env bash
# Active le venv du projet et lance le test de energy_measurement.py.
# Usage : ./run_test.sh [gpu_index] [duration_s]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

GPU_INDEX="${1:-0}"
DURATION="${2:-60}"

source "$REPO_ROOT/.venv/bin/activate"
python "$SCRIPT_DIR/measure_dummy_inference.py" --gpu-index "$GPU_INDEX" --duration "$DURATION"

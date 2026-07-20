#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

PYTHON_BIN="${PYTHON_BIN:-/home/cs1095/.conda/envs/torch-env/bin/python}"
MODEL_NAME="${MODEL_NAME:-/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/coding_atomic_sweep_${TIMESTAMP}}"
DATA_DIR="${DATA_DIR:-${RUN_ROOT}/data}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
PREPARE="${PREPARE:-1}"
DRY_RUN="${DRY_RUN:-0}"

self_print_context \
  "Run root" "${RUN_ROOT}" \
  "Data dir" "${DATA_DIR}" \
  "Model" "${MODEL_NAME}" \
  "Python" "${PYTHON_BIN}" \
  "Prepare" "${PREPARE}" \
  "Dry run" "${DRY_RUN}"

if self_parse_bool "${PREPARE}"; then
  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] DRY_RUN=1; pinned data preparation is skipped."
  else
    "${PYTHON_BIN}" -m self.experiments.coding_atomic_sweep prepare \
      --output-dir "${RUN_ROOT}" \
      --model-name "${MODEL_NAME}"
  fi
fi

submit_args=(
  submit
  --run-root "${RUN_ROOT}"
  --data-dir "${DATA_DIR}"
  --model-name "${MODEL_NAME}"
  --python-bin "${PYTHON_BIN}"
  --log-dir "${LOG_DIR}"
)
if self_parse_bool "${DRY_RUN}"; then
  submit_args+=(--dry-run)
fi

"${PYTHON_BIN}" -m self.experiments.coding_atomic_sweep "${submit_args[@]}"

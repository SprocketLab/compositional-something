#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

PYTHON_BIN="${PYTHON_BIN:-/home/cs1095/.conda/envs/torch-env/bin/python}"
MODEL_NAME="${MODEL_NAME:-/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
SOURCE_RUN_ROOT="${SOURCE_RUN_ROOT:-${ROOT_DIR}/artifacts/runs/bfcl_cumulative_size_sweep_20260721_132230}"
ATOMIC_RUN_ROOT="${ATOMIC_RUN_ROOT:-${ROOT_DIR}/artifacts/runs/coding_atomic_sweep_20260718_014707}"
SEED_ADAPTER="${SEED_ADAPTER:-${ATOMIC_RUN_ROOT}/cells/bfcl/n240-s30-lr2em04-seed7/adapter}"
G1_ADAPTER="${G1_ADAPTER:-${SOURCE_RUN_ROOT}/cells/n1000-compose_g1/round_03/adapter}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/bfcl_schema_generalization_audit_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
TIME_LIMIT="${TIME_LIMIT:-01:15:00}"
PREPARE="${PREPARE:-1}"
DRY_RUN="${DRY_RUN:-0}"

self_print_context \
  "Run root" "${RUN_ROOT}" \
  "Source run" "${SOURCE_RUN_ROOT}" \
  "Seed adapter" "${SEED_ADAPTER}" \
  "G1 adapter" "${G1_ADAPTER}" \
  "Model" "${MODEL_NAME}" \
  "Python" "${PYTHON_BIN}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Max concurrent" "${MAX_CONCURRENT}" \
  "Time limit" "${TIME_LIMIT}" \
  "Prepare" "${PREPARE}" \
  "Dry run" "${DRY_RUN}"

common_args=(
  --run-root "${RUN_ROOT}"
  --source-run-root "${SOURCE_RUN_ROOT}"
  --model-name "${MODEL_NAME}"
  --seed-adapter "${SEED_ADAPTER}"
  --g1-adapter "${G1_ADAPTER}"
)

if self_parse_bool "${PREPARE}"; then
  "${PYTHON_BIN}" -m self.experiments.bfcl_schema_generalization_audit \
    prepare "${common_args[@]}" --resume
elif [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  echo "[ERROR] PREPARE=0 but ${RUN_ROOT}/manifest.json does not exist." >&2
  exit 2
fi

submit_args=(
  submit
  "${common_args[@]}"
  --python-bin "${PYTHON_BIN}"
  --log-dir "${LOG_DIR}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --max-concurrent "${MAX_CONCURRENT}"
  --time-limit "${TIME_LIMIT}"
)
if self_parse_bool "${DRY_RUN}"; then
  submit_args+=(--dry-run)
fi

"${PYTHON_BIN}" -m self.experiments.bfcl_schema_generalization_audit \
  "${submit_args[@]}"

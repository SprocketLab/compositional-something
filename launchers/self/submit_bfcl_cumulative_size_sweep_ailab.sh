#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

PYTHON_BIN="${PYTHON_BIN:-/home/cs1095/.conda/envs/torch-env/bin/python}"
MODEL_NAME="${MODEL_NAME:-/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}"
ATOMIC_RUN_ROOT="${ATOMIC_RUN_ROOT:-${ROOT_DIR}/artifacts/runs/coding_atomic_sweep_20260718_014707}"
ATOMIC_DATA_DIR="${ATOMIC_DATA_DIR:-${ATOMIC_RUN_ROOT}/data/bfcl}"
SEED_ADAPTER="${SEED_ADAPTER:-${ATOMIC_RUN_ROOT}/cells/bfcl/n240-s30-lr2em04-seed7/adapter}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/bfcl_cumulative_size_sweep_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
PREPARE="${PREPARE:-1}"
DRY_RUN="${DRY_RUN:-0}"
# Grid and construction options (prepare only).
SIZES="${SIZES:-}"
CONDITIONS="${CONDITIONS:-}"
COMPONENT_CONTEXT="${COMPONENT_CONTEXT:-}"
CANDIDATE_POOL="${CANDIDATE_POOL:-}"
REPEAT_POOL="${REPEAT_POOL:-}"
LEARNING_RATES="${LEARNING_RATES:-}"
# Optimizer budget; empty MAX_STEPS keeps the one-epoch default.
MAX_STEPS="${MAX_STEPS:-}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"

self_print_context \
  "Run root" "${RUN_ROOT}" \
  "Atomic data" "${ATOMIC_DATA_DIR}" \
  "Seed adapter" "${SEED_ADAPTER}" \
  "Model" "${MODEL_NAME}" \
  "Python" "${PYTHON_BIN}" \
  "Prepare" "${PREPARE}" \
  "Sizes" "${SIZES:-default}" \
  "Conditions" "${CONDITIONS:-default}" \
  "Component context" "${COMPONENT_CONTEXT:-default}" \
  "Candidate pool" "${CANDIDATE_POOL:-default}" \
  "Learning rates" "${LEARNING_RATES:-default}" \
  "Max steps" "${MAX_STEPS:-one epoch}" \
  "Checkpoint steps" "${CHECKPOINT_STEPS:-none}" \
  "Dry run" "${DRY_RUN}"

common_args=(
  --run-root "${RUN_ROOT}"
  --atomic-data-dir "${ATOMIC_DATA_DIR}"
  --model-name "${MODEL_NAME}"
  --seed-adapter "${SEED_ADAPTER}"
)

prepare_args=()
[[ -n "${SIZES}" ]] && prepare_args+=(--sizes "${SIZES}")
[[ -n "${CONDITIONS}" ]] && prepare_args+=(--conditions "${CONDITIONS}")
[[ -n "${COMPONENT_CONTEXT}" ]] && prepare_args+=(--component-context "${COMPONENT_CONTEXT}")
[[ -n "${CANDIDATE_POOL}" ]] && prepare_args+=(--candidate-pool "${CANDIDATE_POOL}")
[[ -n "${REPEAT_POOL}" ]] && prepare_args+=(--repeat-pool "${REPEAT_POOL}")
[[ -n "${LEARNING_RATES}" ]] && prepare_args+=(--learning-rates "${LEARNING_RATES}")

training_args=()
[[ -n "${MAX_STEPS}" ]] && training_args+=(--max-steps "${MAX_STEPS}")
[[ -n "${CHECKPOINT_STEPS}" ]] && training_args+=(--checkpoint-steps "${CHECKPOINT_STEPS}")

if self_parse_bool "${PREPARE}"; then
  "${PYTHON_BIN}" -m self.experiments.bfcl_cumulative_size_sweep prepare \
    "${common_args[@]}" ${prepare_args[@]+"${prepare_args[@]}"}
elif [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  echo "[ERROR] PREPARE=0 but ${RUN_ROOT}/manifest.json does not exist." >&2
  exit 2
fi

submit_args=(
  submit
  "${common_args[@]}"
  --python-bin "${PYTHON_BIN}"
  --log-dir "${LOG_DIR}"
)
if [[ ${#training_args[@]} -gt 0 ]]; then
  submit_args+=("${training_args[@]}")
fi
if self_parse_bool "${DRY_RUN}"; then
  submit_args+=(--dry-run)
fi

"${PYTHON_BIN}" -m self.experiments.bfcl_cumulative_size_sweep "${submit_args[@]}"

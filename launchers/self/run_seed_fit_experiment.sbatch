#!/usr/bin/env bash
#SBATCH --job-name=seedfit-grid
#SBATCH --output=artifacts/logs/seedfit-grid-%j.out
#SBATCH --error=artifacts/logs/seedfit-grid-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

self_resolve_python

DEFAULT_LOCAL_MODEL="${ROOT_DIR}/artifacts/models/SmolLM2-360M"
DEFAULT_HUB_MODEL="HuggingFaceTB/SmolLM2-360M"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  DEFAULT_MODEL="${DEFAULT_LOCAL_MODEL}"
else
  DEFAULT_MODEL="${DEFAULT_HUB_MODEL}"
fi

TASK_NAME="${TASK_NAME:?Set TASK_NAME to run_length or multiplication.}"
TASK_TAG="${TASK_TAG:-${TASK_NAME}}"
DATA_TAG="${DATA_TAG:-default}"
STEP_TAG="${STEP_TAG:-steps}"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL}}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-960}"
DECODE_MAX_NEW_TOKENS="${DECODE_MAX_NEW_TOKENS:-16}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
DRY_RUN="${DRY_RUN:-0}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/seed_fit_grid/${TASK_TAG}/${DATA_TAG}/${STEP_TAG}/seed_${SEED}}"

mkdir -p "${OUT_ROOT}"

COMMON_ARGS=(
  --task "${TASK_NAME}"
  --model-name "${MODEL_NAME}"
  --output-dir "${OUT_ROOT}"
  --seed "${SEED}"
  --bf16
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num-epochs "${NUM_EPOCHS}"
  --max-steps "${MAX_STEPS}"
  --decode-max-new-tokens "${DECODE_MAX_NEW_TOKENS}"
  --logging-steps "${LOGGING_STEPS}"
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

self_print_context \
  "Task" "${TASK_NAME}" \
  "Data tag" "${DATA_TAG}" \
  "Step tag" "${STEP_TAG}" \
  "Output dir" "${OUT_ROOT}" \
  "Python" "${PYTHON_BIN}" \
  "Model" "${MODEL_NAME}" \
  "Dry run" "${DRY_RUN}" \
  "Slurm Job ID" "${SLURM_JOB_ID:-unknown}" \
  "Hostname" "$(hostname)"
if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; skipping CUDA and model/tokenizer preflight."
else
  nvidia-smi -L || true

  self_print_torch_cuda_info
  self_preflight_model_snapshot "${MODEL_NAME}"
fi

echo "[INFO] Launch command: ${PYTHON_BIN} -m self.seed_fit_experiment ${COMMON_ARGS[*]}"
if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; command not executed."
  exit 0
fi

"${PYTHON_BIN}" -m self.seed_fit_experiment "${COMMON_ARGS[@]}"

echo "[INFO] Finished seed-fit task run."

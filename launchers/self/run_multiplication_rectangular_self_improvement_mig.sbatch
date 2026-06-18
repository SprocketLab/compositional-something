#!/usr/bin/env bash
#SBATCH --job-name=mult-rect-si
#SBATCH --output=artifacts/logs/mult-rect-si-%j.out
#SBATCH --error=artifacts/logs/mult-rect-si-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --partition=mig
#SBATCH --time=48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

self_resolve_python

RECIPE_NAME="multiplication_self_improve_v1"
FORMAT_VERSION="symbolic_v1"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/multiplication_rectangular_seed_best}"
BASELINE="${BASELINE:-compose}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-}"
MAX_STEPS="${MAX_STEPS:-}"
SEED_REPLAY_TRAIN_PER_PARTITION="${SEED_REPLAY_TRAIN_PER_PARTITION:-2000}"
EXPAND_TRAIN_PER_PARTITION="${EXPAND_TRAIN_PER_PARTITION:-2000}"
FRONTIER_ROW_PROFILE="${FRONTIER_ROW_PROFILE:-uniform}"
INITIAL_MAX_B_DIGITS="${INITIAL_MAX_B_DIGITS:-8}"
EXPAND_B_DIGITS="${EXPAND_B_DIGITS:-2}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-4}"
HELDOUT_PER_PARTITION="${HELDOUT_PER_PARTITION:-200}"
FRONTIER_MIN_A_DIGITS="${FRONTIER_MIN_A_DIGITS:-1}"
FRONTIER_MAX_A_DIGITS="${FRONTIER_MAX_A_DIGITS:-6}"
FRONTIER_MIN_B_DIGITS="${FRONTIER_MIN_B_DIGITS:-2}"
SAVE_MODEL="${SAVE_MODEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_self_improvement_${TS}}"

case "${BASELINE}" in
  short_only) MODE_ARGS=(--pseudo-label-mode short_only) ;;
  direct) MODE_ARGS=(--pseudo-label-mode direct) ;;
  compose) MODE_ARGS=(--pseudo-label-mode compose) ;;
  compose_corrupt) MODE_ARGS=(--pseudo-label-mode compose_corrupt --corruption-rate 0.10) ;;
  *)
    echo "[ERROR] Unknown BASELINE=${BASELINE}" >&2
    exit 1
    ;;
esac

BASE_OUT="${OUT_ROOT}/${BASELINE}"
mkdir -p "${BASE_OUT}"

run_or_dry() {
  self_print_command_stdout "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

cmd=(
  "${PYTHON_BIN}" -m self.rectangular_multiplication_self_improvement
  --seed-model "${SEED_MODEL}"
  --output-dir "${BASE_OUT}"
  --recipe "${RECIPE_NAME}"
  --format-version "${FORMAT_VERSION}"
  --frontier-min-a-digits "${FRONTIER_MIN_A_DIGITS}"
  --frontier-max-a-digits "${FRONTIER_MAX_A_DIGITS}"
  --frontier-min-b-digits "${FRONTIER_MIN_B_DIGITS}"
  --initial-max-b-digits "${INITIAL_MAX_B_DIGITS}"
  --num-expand-rounds "${NUM_EXPAND_ROUNDS}"
  --expand-b-digits "${EXPAND_B_DIGITS}"
  --seed-replay-train-per-partition "${SEED_REPLAY_TRAIN_PER_PARTITION}"
  --expand-train-per-partition "${EXPAND_TRAIN_PER_PARTITION}"
  --frontier-row-profile "${FRONTIER_ROW_PROFILE}"
  --heldout-per-partition "${HELDOUT_PER_PARTITION}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --num-epochs 1
  --logging-steps 50
  --decode-max-new-tokens 48
  --bf16
  --bucket-train-batches-by-partition
  --seed "${SEED}"
  "${MODE_ARGS[@]}"
)

if [[ -n "${LEARNING_RATE}" ]]; then
  cmd+=(--learning-rate "${LEARNING_RATE}")
fi

if [[ -n "${MAX_STEPS}" ]]; then
  cmd+=(--max-steps "${MAX_STEPS}")
fi

if self_parse_bool "${SAVE_MODEL}"; then
  cmd+=(--save-model)
else
  cmd+=(--no-save-model)
fi

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Baseline" "${BASELINE}" \
  "Seed model" "${SEED_MODEL}" \
  "Initial max b digits" "${INITIAL_MAX_B_DIGITS}" \
  "Expand b digits" "${EXPAND_B_DIGITS}" \
  "Num expand rounds" "${NUM_EXPAND_ROUNDS}" \
  "Seed replay per partition" "${SEED_REPLAY_TRAIN_PER_PARTITION}" \
  "Frontier train per partition" "${EXPAND_TRAIN_PER_PARTITION}" \
  "Frontier row profile" "${FRONTIER_ROW_PROFILE}" \
  "Learning rate" "${LEARNING_RATE:-recipe_default}" \
  "Max steps" "${MAX_STEPS:-recipe_default}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Dry run" "${DRY_RUN}" \
  "Slurm job id" "${SLURM_JOB_ID:-unknown}" \
  "Hostname" "$(hostname)"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Dry run: skipping CUDA probes."
else
  nvidia-smi -L || true
  "${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
fi

run_or_dry "${cmd[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] Status: dry_run"
else
  echo "[INFO] Status: completed"
  echo "[INFO] Summary: ${BASE_OUT}/self_improvement_results.json"
fi

#!/usr/bin/env bash
#SBATCH --job-name=mult-rect-seed
#SBATCH --output=artifacts/logs/mult-rect-seed-%j.out
#SBATCH --error=artifacts/logs/mult-rect-seed-%j.err
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

MODEL_NAME="meta/models/tiny_gpt2_8l_384d"
RECIPE_NAME="multiplication_self_improve_v1"
FORMAT_VERSION="symbolic_v1"
PARTITIONS_SPEC="${PARTITIONS_SPEC:-1x1,1x2,1x3,1x4,1x5,1x6,2x1,3x1,4x1,5x1,6x1}"

# User-facing overrides: LR, TRAIN_PER_PARTITION, MAX_STEPS, TRAIN_BATCH_SIZE,
# EVAL_BATCH_SIZE, SEED, OUT_ROOT, SAVE_MODEL.
# Internal wrapper overrides: HELDOUT_PER_PARTITION, PARTITIONS_SPEC,
# SKIP_TRAIN_EVAL.
SEED="${SEED:-0}"
LR="${LR:-5e-5}"
TRAIN_PER_PARTITION="${TRAIN_PER_PARTITION:-50000}"
HELDOUT_PER_PARTITION="${HELDOUT_PER_PARTITION:-200}"
MAX_STEPS="${MAX_STEPS:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SKIP_TRAIN_EVAL="${SKIP_TRAIN_EVAL:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_seed_mig_${TS}}"

mkdir -p "${OUT_ROOT}"

run_or_dry() {
  self_print_command_stdout "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

cmd=(
  "${PYTHON_BIN}" -m self.rectangular_multiplication_recipe_seed_fit
  --model-name "${MODEL_NAME}"
  --output-dir "${OUT_ROOT}"
  --recipe "${RECIPE_NAME}"
  --format-version "${FORMAT_VERSION}"
  --partitions "${PARTITIONS_SPEC}"
  --train-per-partition "${TRAIN_PER_PARTITION}"
  --heldout-per-partition "${HELDOUT_PER_PARTITION}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --num-epochs 1
  --learning-rate "${LR}"
  --logging-steps 50
  --max-steps "${MAX_STEPS}"
  --decode-max-new-tokens 48
  --bf16
  --init-from-scratch
  --bucket-train-batches-by-partition
  --seed "${SEED}"
)

if self_parse_bool "${SAVE_MODEL}"; then
  cmd+=(--save-model)
else
  cmd+=(--no-save-model)
fi

if self_parse_bool "${SKIP_TRAIN_EVAL}"; then
  cmd+=(--skip-train-eval)
fi

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Model name" "${MODEL_NAME}" \
  "Recipe" "${RECIPE_NAME}" \
  "Partitions" "${PARTITIONS_SPEC}" \
  "Train per partition" "${TRAIN_PER_PARTITION}" \
  "Heldout per partition" "${HELDOUT_PER_PARTITION}" \
  "Learning rate" "${LR}" \
  "Max steps" "${MAX_STEPS}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Save model" "${SAVE_MODEL}" \
  "Skip train eval" "${SKIP_TRAIN_EVAL}" \
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
  echo "[INFO] Summary: ${OUT_ROOT}/summary.json"
fi

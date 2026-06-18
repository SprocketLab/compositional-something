#!/usr/bin/env bash
#SBATCH --job-name=mult-rect-square-diag
#SBATCH --output=artifacts/logs/mult-rect-square-diag-%j.out
#SBATCH --error=artifacts/logs/mult-rect-square-diag-%j.err
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
SEED_PARTITIONS_SPEC="${SEED_PARTITIONS_SPEC:-1x1,1x2,1x3,2x1,2x2,2x3,3x1,3x2,3x3}"
FRONTIER_PARTITIONS_SPEC="${FRONTIER_PARTITIONS_SPEC:-4x4,4x5,4x6,4x7,4x8,5x4,5x5,5x6,5x7,5x8,6x4,6x5,6x6,6x7,6x8,7x4,7x5,7x6,7x7,7x8,8x4,8x5,8x6,8x7,8x8}"

SEED_MODEL="${SEED_MODEL:-}"
MODE="${MODE:-compose}"
SEED="${SEED:-0}"
TRAIN_PER_PARTITION="${TRAIN_PER_PARTITION:-2000}"
HELDOUT_PER_PARTITION="${HELDOUT_PER_PARTITION:-100}"
MAX_STEPS="${MAX_STEPS:-3000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SAVE_MODEL="${SAVE_MODEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_square_compose_diag_mig_${TS}}"

if [[ -z "${SEED_MODEL}" ]]; then
  echo "[ERROR] SEED_MODEL is required." >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

run_or_dry() {
  self_print_command_stdout "$@"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "$@"
}

cmd=(
  "${PYTHON_BIN}" -m self.diagnostics.rectangular_multiplication_compose_diagnostic
  --seed-model "${SEED_MODEL}"
  --output-dir "${OUT_ROOT}"
  --recipe "${RECIPE_NAME}"
  --format-version "${FORMAT_VERSION}"
  --seed-partitions "${SEED_PARTITIONS_SPEC}"
  --frontier-partitions "${FRONTIER_PARTITIONS_SPEC}"
  --train-per-partition "${TRAIN_PER_PARTITION}"
  --heldout-per-partition "${HELDOUT_PER_PARTITION}"
  --mode "${MODE}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --num-epochs 1
  --logging-steps 50
  --max-steps "${MAX_STEPS}"
  --decode-max-new-tokens 48
  --seed "${SEED}"
)

if self_parse_bool "${SAVE_MODEL}"; then
  cmd+=(--save-model)
else
  cmd+=(--no-save-model)
fi

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Seed model" "${SEED_MODEL}" \
  "Recipe" "${RECIPE_NAME}" \
  "Mode" "${MODE}" \
  "Seed partitions" "${SEED_PARTITIONS_SPEC}" \
  "Frontier partitions" "${FRONTIER_PARTITIONS_SPEC}" \
  "Train per partition" "${TRAIN_PER_PARTITION}" \
  "Heldout per partition" "${HELDOUT_PER_PARTITION}" \
  "Max steps" "${MAX_STEPS}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Save model" "${SAVE_MODEL}" \
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

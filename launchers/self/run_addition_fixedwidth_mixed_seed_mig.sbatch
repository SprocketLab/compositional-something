#!/usr/bin/env bash
#SBATCH --job-name=add-fw-seed
#SBATCH --output=artifacts/logs/add-fw-seed-%j.out
#SBATCH --error=artifacts/logs/add-fw-seed-%j.err
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
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

self_resolve_python

# Supported experiment overrides: LR, TRAIN_PER_DIGIT, MAX_STEPS,
# TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, SEED, OUT_ROOT, SAVE_MODEL,
# ADDITION_SAMPLING_MODE.
SEED="${SEED:-0}"
LR="${LR:-5e-4}"
TRAIN_PER_DIGIT="${TRAIN_PER_DIGIT:-50000}"
MAX_STEPS="${MAX_STEPS:-10000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SAVE_MODEL="${SAVE_MODEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_ACCURACY_THRESHOLD="${TARGET_ACCURACY_THRESHOLD:-0.95}"
ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE:-balanced_visible_lengths}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_fixedwidth_mixed_seed_${TS}}"
SEED_MODEL_LINK="${ROOT_DIR}/artifacts/models/addition_fixedwidth_mixed_seed_best"

mkdir -p "${OUT_ROOT}" "${ROOT_DIR}/artifacts/models"

cmd=(
  "${PYTHON_BIN}" -m self.seed_fit_experiment
  --task addition
  --model-name recipe_scratch
  --output-dir "${OUT_ROOT}"
  --seed "${SEED}"
  --recipe arithmetic_self_improve_v1
  --init-from-scratch
  --addition-width-mode fixed_width_mixed_prompt
  --addition-sampling-mode "${ADDITION_SAMPLING_MODE}"
  --addition-composition-path-mode fixed_binary
  --initial-min-size 3
  --initial-max-size 7
  --initial-train-per-size "${TRAIN_PER_DIGIT}"
  --initial-eval-per-size 200
  --expand-num-size 1
  --expand-train-per-size 0
  --eval-per-size 0
  --composed-eval-per-size 0
  --num-expand-rounds 0
  --pseudo-label-mode none
  --num-epochs 1
  --learning-rate "${LR}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --logging-steps 100
  --max-steps "${MAX_STEPS}"
  --decode-max-new-tokens 48
  --target-accuracy-threshold "${TARGET_ACCURACY_THRESHOLD}"
  --bucket-train-batches-by-size
  --evaluate-train
  --bf16
)

if self_parse_bool "${SAVE_MODEL}"; then
  cmd+=(--save-model)
fi

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Stable seed symlink" "${SEED_MODEL_LINK}" \
  "Addition sampling mode" "${ADDITION_SAMPLING_MODE}" \
  "Dry run" "${DRY_RUN}"
self_print_command_stdout "${cmd[@]}"

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; command not executed."
  exit 0
fi

"${cmd[@]}"

"${PYTHON_BIN}" - "${OUT_ROOT}/seed_fit_results.json" "${TARGET_ACCURACY_THRESHOLD}" <<'PY'
import json
import sys

path = sys.argv[1]
threshold = float(sys.argv[2])
with open(path, "r", encoding="utf-8") as handle:
    payload = json.load(handle)
validation_min = payload.get("validation_min_per_size_accuracy")
test_min = payload.get("test_min_per_size_accuracy")
if validation_min is None or test_min is None:
    raise SystemExit(f"Missing validation/test min accuracy in {path}")
if validation_min < threshold or test_min < threshold:
    raise SystemExit(
        "Seed did not meet threshold: "
        f"validation_min={validation_min:.4f}, test_min={test_min:.4f}, threshold={threshold:.4f}"
    )
print(
    "[INFO] Seed passed threshold: "
    f"validation_min={validation_min:.4f}, test_min={test_min:.4f}, threshold={threshold:.4f}",
    flush=True,
)
PY

if self_parse_bool "${SAVE_MODEL}"; then
  if [[ ! -d "${OUT_ROOT}/model" ]]; then
    echo "[ERROR] Expected saved model at ${OUT_ROOT}/model" >&2
    exit 1
  fi
  ln -sfn "${OUT_ROOT}/model" "${SEED_MODEL_LINK}"
  echo "[INFO] Updated ${SEED_MODEL_LINK} -> ${OUT_ROOT}/model"
fi

echo "[INFO] Finished fixed-width mixed-prompt addition seed."

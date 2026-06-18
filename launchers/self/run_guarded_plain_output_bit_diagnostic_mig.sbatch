#!/usr/bin/env bash
#SBATCH --job-name=guarded-bit-diag
#SBATCH --output=artifacts/logs/guarded-bit-diag-%j.out
#SBATCH --error=artifacts/logs/guarded-bit-diag-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=mig
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --time=48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

self_resolve_python

TASK="${TASK:-run_length}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/guarded_plain_output_bit_diagnostic_${TS}}"
SCRATCH_MODEL_NAME="${SCRATCH_MODEL_NAME:-${ROOT_DIR}/meta/models/tiny_gpt2_8l_384d}"
SEED_MODEL="${SEED_MODEL:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SEED_MAX_STEPS="${SEED_MAX_STEPS:-}"
ROUND_MAX_STEPS="${ROUND_MAX_STEPS:-}"
ROUND_WARMUP_STEPS="${ROUND_WARMUP_STEPS:-}"
INITIAL_MIN_BITS="${INITIAL_MIN_BITS:-8}"
INITIAL_MAX_BITS="${INITIAL_MAX_BITS:-12}"
INITIAL_TRAIN_PER_BIT="${INITIAL_TRAIN_PER_BIT:-20000}"
INITIAL_EVAL_PER_BIT="${INITIAL_EVAL_PER_BIT:-100}"
EXPAND_NUM_BITS="${EXPAND_NUM_BITS:-8}"
FRONTIER_MIN_BITS="${FRONTIER_MIN_BITS:-18}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-1}"
EXPAND_TRAIN_PER_BIT="${EXPAND_TRAIN_PER_BIT:-2000}"
EVAL_PER_BIT="${EVAL_PER_BIT:-100}"
COMPOSED_EVAL_PER_BIT="${COMPOSED_EVAL_PER_BIT:-100}"
SYMBOL_ALPHABET_SIZE="${SYMBOL_ALPHABET_SIZE:-2}"
TARGET_MODE="${TARGET_MODE:-symbol_run_pair}"
BIT_COMPOSITION_PATH_MODE="${BIT_COMPOSITION_PATH_MODE:-random}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SAVE_MODEL_POLICY="${SAVE_MODEL_POLICY:-all_rounds}"

case "${TASK}" in
  run_length)
    TASK_MODULE="self.run_length_self_improvement"
    GUARDED_RULE="run_length_no_boundary_continue"
    SEED_GATE="0.97"
    ;;
  *)
    echo "[ERROR] Unsupported TASK=${TASK}" >&2
    exit 1
    ;;
esac

TASK_OUT="${OUT_ROOT}/${TASK}"
SEED_OUT="${TASK_OUT}/seed"
DIRECT_OUT="${TASK_OUT}/pilot/direct"
GUARDED_OUT="${TASK_OUT}/pilot/guarded_compose"
SEED_MODEL_PATH="${SEED_MODEL:-${SEED_OUT}/model}"

run_cmd() {
  local -a cmd=("$@")
  self_print_command_stdout "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

check_seed_gate() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  local results_path="${SEED_OUT}/seed_fit_results.json"
  PYTHONPATH=. "${PYTHON_BIN}" - <<'PY' "${results_path}" "${TASK}" "${SEED_GATE}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
threshold = float(sys.argv[3])
payload = json.loads(path.read_text())
validation_min = payload.get("validation_min_per_size_accuracy")
test_min = payload.get("test_min_per_size_accuracy")
if validation_min is None or test_min is None or validation_min < threshold or test_min < threshold:
    raise SystemExit(
        f"[ERROR] Seed gate failed for {task}: validation_min={validation_min} test_min={test_min} threshold={threshold}"
    )
print(
    f"[INFO] Seed gate passed for {task}: validation_min={validation_min:.4f} test_min={test_min:.4f}",
    flush=True,
)
PY
}

seed_cmd=(
  "${PYTHON_BIN}" -m self.seed_fit_experiment
  --task "${TASK}"
  --model-name "${SCRATCH_MODEL_NAME}"
  --output-dir "${SEED_OUT}"
  --format-version legacy
  --target-mode "${TARGET_MODE}"
  --recipe algorithmic_self_improve_v1
  --init-from-scratch
  --symbol-alphabet-size "${SYMBOL_ALPHABET_SIZE}"
  --initial-min-size "${INITIAL_MIN_BITS}"
  --initial-max-size "${INITIAL_MAX_BITS}"
  --initial-train-per-size "${INITIAL_TRAIN_PER_BIT}"
  --initial-eval-per-size "${INITIAL_EVAL_PER_BIT}"
  --bucket-train-batches-by-size
  --save-model
  --bf16
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --seed "${SEED}"
)
if [[ -n "${SEED_MAX_STEPS}" ]]; then
  seed_cmd+=(--max-steps "${SEED_MAX_STEPS}")
fi

pilot_common=(
  --model-name "${SEED_MODEL_PATH}"
  --format-version legacy
  --target-mode "${TARGET_MODE}"
  --compose-arity exact2
  --bit-composition-path-mode "${BIT_COMPOSITION_PATH_MODE}"
  --recipe algorithmic_self_improve_v1
  --treat-seed-as-round-zero
  --symbol-alphabet-size "${SYMBOL_ALPHABET_SIZE}"
  --initial-min-bits "${INITIAL_MIN_BITS}"
  --initial-max-bits "${INITIAL_MAX_BITS}"
  --initial-train-per-bit "${INITIAL_TRAIN_PER_BIT}"
  --initial-eval-per-bit "${INITIAL_EVAL_PER_BIT}"
  --frontier-min-bits "${FRONTIER_MIN_BITS}"
  --num-expand-rounds "${NUM_EXPAND_ROUNDS}"
  --expand-num-bits "${EXPAND_NUM_BITS}"
  --expand-train-per-bit "${EXPAND_TRAIN_PER_BIT}"
  --eval-per-bit "${EVAL_PER_BIT}"
  --composed-eval-per-bit "${COMPOSED_EVAL_PER_BIT}"
  --bucket-train-batches-by-bits
  --bf16
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --seed "${SEED}"
)
if [[ "${SAVE_MODEL}" == "1" ]]; then
  pilot_common+=(--save-model-policy "${SAVE_MODEL_POLICY}")
else
  pilot_common+=(--save-model-policy none)
fi
if [[ -n "${ROUND_MAX_STEPS}" ]]; then
  pilot_common+=(--max-steps "${ROUND_MAX_STEPS}")
fi
if [[ -n "${ROUND_WARMUP_STEPS}" ]]; then
  pilot_common+=(--self-improve-warmup-steps "${ROUND_WARMUP_STEPS}")
fi

direct_cmd=(
  "${PYTHON_BIN}" -m "${TASK_MODULE}"
  --output-dir "${DIRECT_OUT}"
  --pseudo-label-mode direct
  --guarded-compose-rule none
  "${pilot_common[@]}"
)

guarded_cmd=(
  "${PYTHON_BIN}" -m "${TASK_MODULE}"
  --output-dir "${GUARDED_OUT}"
  --pseudo-label-mode compose
  --guarded-compose-rule "${GUARDED_RULE}"
  "${pilot_common[@]}"
)

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Task" "${TASK}" \
  "Output root" "${OUT_ROOT}" \
  "Scratch model" "${SCRATCH_MODEL_NAME}" \
  "Seed model override" "${SEED_MODEL:-<none>}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Seed range" "${INITIAL_MIN_BITS}..${INITIAL_MAX_BITS}" \
  "Frontier range" "${FRONTIER_MIN_BITS}..$((FRONTIER_MIN_BITS + EXPAND_NUM_BITS - 1))" \
  "Symbol alphabet size" "${SYMBOL_ALPHABET_SIZE}" \
  "Target mode" "${TARGET_MODE}" \
  "Bit composition path mode" "${BIT_COMPOSITION_PATH_MODE}" \
  "Num expand rounds" "${NUM_EXPAND_ROUNDS}" \
  "Round warmup steps" "${ROUND_WARMUP_STEPS:-recipe_default}" \
  "Save pilot checkpoints" "${SAVE_MODEL}" \
  "Save model policy" "${SAVE_MODEL_POLICY}" \
  "Dry run" "${DRY_RUN}" \
  "Slurm Job ID" "${SLURM_JOB_ID:-unknown}" \
  "Hostname" "$(hostname)"
nvidia-smi -L || true

if [[ -n "${SEED_MODEL}" ]]; then
  if [[ ! -d "${SEED_MODEL}" ]]; then
    echo "[ERROR] SEED_MODEL does not exist: ${SEED_MODEL}" >&2
    exit 1
  fi
  echo "[INFO] Reusing existing seed checkpoint from ${SEED_MODEL}"
else
  run_cmd "${seed_cmd[@]}"
  check_seed_gate
fi
run_cmd "${direct_cmd[@]}"
run_cmd "${guarded_cmd[@]}"

echo "[INFO] Finished guarded plain-output diagnostic for ${TASK}."

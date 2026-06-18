#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_fixedwidth_mixed_fullpack_${TS}}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/addition_fixedwidth_mixed_seed_best}"
ONLY_BASELINE="${BASELINE:-}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-5000}"
EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-10000}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-8}"
EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS:-2}"
ADDITION_WIDTH_MODE="${ADDITION_WIDTH_MODE:-fixed_width_mixed_prompt}"
ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE:-balanced_visible_lengths}"
ADDITION_COMPOSITION_PATH_MODE="${ADDITION_COMPOSITION_PATH_MODE:-fixed_binary}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --recipe arithmetic_self_improve_v1
  --treat-seed-as-round-zero
  --seed-range-train-mode direct_pseudo
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 0
  --initial-eval-per-digit 200
  --addition-width-mode "${ADDITION_WIDTH_MODE}"
  --addition-sampling-mode "${ADDITION_SAMPLING_MODE}"
  --addition-composition-path-mode "${ADDITION_COMPOSITION_PATH_MODE}"
  --num-expand-rounds "${NUM_EXPAND_ROUNDS}"
  --expand-num-digits "${EXPAND_NUM_DIGITS}"
  --seed-replay-train-per-digit "${SEED_REPLAY_TRAIN_PER_DIGIT}"
  --expand-train-per-digit "${EXPAND_TRAIN_PER_DIGIT}"
  --eval-per-digit 100
  --composed-eval-per-digit 50
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --decode-max-new-tokens 48
  --composed-refresh-mode dynamic
  --bucket-train-batches-by-digits
  --bf16
  --resume
  --seed "${SEED}"
  --early-stop-patience 2
  --early-stop-expanded-eval-threshold 0.01
  --early-stop-frontier-train-threshold 0.50
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

run_baseline() {
  local baseline="$1"
  shift
  if [[ -n "${ONLY_BASELINE}" && "${baseline}" != "${ONLY_BASELINE}" ]]; then
    return 0
  fi

  local out_dir="${OUT_ROOT}/${baseline}"
  local cmd=(
    "${PYTHON_BIN}" -m self.legacy.addition_self_improvement
    --model-name "${SEED_MODEL}"
    --output-dir "${out_dir}"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  echo
  echo "[INFO] Starting baseline=${baseline}"
  echo "[INFO] Output dir: ${out_dir}"
  self_print_command_stdout "${cmd[@]}"

  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] DRY_RUN=1; command not executed."
    return 0
  fi

  "${cmd[@]}"
}

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Seed model" "${SEED_MODEL}" \
  "Baseline selector" "${ONLY_BASELINE:-all}" \
  "Addition width mode" "${ADDITION_WIDTH_MODE}" \
  "Addition sampling mode" "${ADDITION_SAMPLING_MODE}" \
  "Addition composition path mode" "${ADDITION_COMPOSITION_PATH_MODE}" \
  "Dry run" "${DRY_RUN}" \
  "Schedule" "rounds=${NUM_EXPAND_ROUNDS} expand_num_digits=${EXPAND_NUM_DIGITS} seed_replay_train_per_digit=${SEED_REPLAY_TRAIN_PER_DIGIT} expand_train_per_digit=${EXPAND_TRAIN_PER_DIGIT}"

run_baseline short_only --pseudo-label-mode none
run_baseline direct --pseudo-label-mode direct
run_baseline with_carry --pseudo-label-mode compose --composed-strategy with_carry
run_baseline with_carry_filtered --pseudo-label-mode compose --composed-strategy with_carry_filtered --composition-error-percent 0
if self_parse_bool "${INCLUDE_COMPOSE_CORRUPT:-0}"; then
  run_baseline compose_corrupt --pseudo-label-mode compose_corrupt --composed-strategy with_carry --corruption-rate 0.10
fi

echo
echo "[INFO] Finished fixed-width mixed-prompt addition fullpack."
echo "[INFO] Final output root: ${OUT_ROOT}"

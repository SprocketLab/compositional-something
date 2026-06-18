#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_recipe_focused_${TS}}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/addition_recipe_seed_best}"
BASELINE="${BASELINE:-with_carry_filtered}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-5000}"
EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-5000}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-1}"
EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS:-5}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --recipe arithmetic_self_improve_v1
  --model-name "${SEED_MODEL}"
  --output-dir "${OUT_ROOT}/${BASELINE}"
  --treat-seed-as-round-zero
  --seed-range-train-mode direct_pseudo
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 0
  --initial-eval-per-digit 200
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
)

case "${BASELINE}" in
  short_only)
    BASELINE_ARGS=(--pseudo-label-mode none)
    ;;
  direct)
    BASELINE_ARGS=(--pseudo-label-mode direct)
    ;;
  with_carry)
    BASELINE_ARGS=(--pseudo-label-mode compose --composed-strategy with_carry)
    ;;
  with_carry_filtered)
    BASELINE_ARGS=(--pseudo-label-mode compose --composed-strategy with_carry_filtered --composition-error-percent 0)
    ;;
  compose_corrupt)
    BASELINE_ARGS=(--pseudo-label-mode compose_corrupt --composed-strategy with_carry --corruption-rate 0.10)
    ;;
  *)
    echo "[ERROR] Unsupported BASELINE=${BASELINE}" >&2
    exit 1
    ;;
esac

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

CMD=("${PYTHON_BIN}" -m self.self_improvement "${COMMON_ARGS[@]}" "${BASELINE_ARGS[@]}")

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Seed model: ${SEED_MODEL}"
echo "[INFO] Baseline: ${BASELINE}"
echo "[INFO] Dry run: ${DRY_RUN}"
self_print_command_stdout "${CMD[@]}"

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; command not executed."
  exit 0
fi

"${CMD[@]}"

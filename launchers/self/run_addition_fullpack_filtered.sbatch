#!/usr/bin/env bash
#SBATCH --job-name=addition-fullpack
#SBATCH --output=artifacts/logs/addition-fullpack-%j.out
#SBATCH --error=artifacts/logs/addition-fullpack-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1g.10gb:1
#SBATCH --partition=mig
#SBATCH --time=144:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

self_resolve_python

SEED_MODEL_LINK="${ROOT_DIR}/artifacts/models/addition_tiny_seed_best"
SEED_MODEL_TARGET="${SEED_MODEL_TARGET:-${ROOT_DIR}/artifacts/runs/addition_tiny_stage2_lr_2e_4_20260418_122516/model}"
TOKENIZER_MODE="${TOKENIZER_MODE:-fixed_char}"
MODEL_NAME="${MODEL_NAME:-${SEED_MODEL_LINK}}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/self_improvement_addition_fullpack_${TS}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-2500}"
EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-5000}"
ONLY_BASELINE="${BASELINE:-}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --tokenizer-mode "${TOKENIZER_MODE}"
  --treat-seed-as-round-zero
  --seed-range-train-mode direct_pseudo
  --bf16
  --resume
  --seed "${SEED}"
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 0
  --initial-eval-per-digit 100
  --num-expand-rounds 8
  --expand-num-digits 5
  --seed-replay-train-per-digit "${SEED_REPLAY_TRAIN_PER_DIGIT}"
  --expand-train-per-digit "${EXPAND_TRAIN_PER_DIGIT}"
  --eval-per-digit 100
  --composed-eval-per-digit 50
  --num-epochs "${NUM_EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --weight-decay 0
  --logging-steps 100
  --decode-max-new-tokens 48
  --composed-refresh-mode dynamic
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

self_print_context \
  "Output root" "${OUT_ROOT}" \
  "Python" "${PYTHON_BIN}" \
  "Model" "${MODEL_NAME}" \
  "Tokenizer mode" "${TOKENIZER_MODE}" \
  "Seed model link" "${SEED_MODEL_LINK}" \
  "Seed model target" "${SEED_MODEL_TARGET}" \
  "Seed" "${SEED}" \
  "Dry run" "${DRY_RUN}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Baseline selector" "${ONLY_BASELINE:-all}" \
  "Slurm job id" "${SLURM_JOB_ID:-unknown}" \
  "Hostname" "$(hostname)"
if [[ "${MODEL_NAME}" == "${SEED_MODEL_LINK}" ]]; then
  mkdir -p "$(dirname "${SEED_MODEL_LINK}")"
  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] DRY_RUN=1; would update ${SEED_MODEL_LINK} -> ${SEED_MODEL_TARGET}"
  else
    ln -sfn "${SEED_MODEL_TARGET}" "${SEED_MODEL_LINK}"
  fi
fi

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; skipping CUDA and model/tokenizer preflight."
else
  nvidia-smi -L || true
  self_print_torch_cuda_info
  self_preflight_model_snapshot "${MODEL_NAME}" "${TOKENIZER_MODE}"
fi

run_baseline() {
  local baseline="$1"
  shift
  if [[ -n "${ONLY_BASELINE}" && "${baseline}" != "${ONLY_BASELINE}" ]]; then
    return 0
  fi
  local out_dir="${OUT_ROOT}/addition/${baseline}"
  mkdir -p "${out_dir}"

  local cmd=(
    "${PYTHON_BIN}" -m self.self_improvement
    --output-dir "${out_dir}"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  echo
  echo "[INFO] $(date --iso-8601=seconds) starting baseline=${baseline}"
  echo "[INFO] Output dir: ${out_dir}"
  self_print_command_stdout "${cmd[@]}"

  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] DRY_RUN=1; command not executed."
    return 0
  fi

  "${cmd[@]}"
  echo "[INFO] $(date --iso-8601=seconds) finished baseline=${baseline}"
}

run_baseline short_only --pseudo-label-mode none
run_baseline direct --pseudo-label-mode direct
run_baseline with_carry --pseudo-label-mode compose --composed-strategy with_carry
run_baseline with_carry_filtered --pseudo-label-mode compose --composed-strategy with_carry_filtered --composition-error-percent 0
run_baseline compose_corrupt --pseudo-label-mode compose_corrupt --composed-strategy with_carry --corruption-rate 0.10

echo
echo "[INFO] Finished addition fullpack."
echo "[INFO] Final output root: ${OUT_ROOT}"

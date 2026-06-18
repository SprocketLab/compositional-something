#!/usr/bin/env bash
#SBATCH --job-name=addition-seed
#SBATCH --output=artifacts/logs/addition-seed-%j.out
#SBATCH --error=artifacts/logs/addition-seed-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=all
#SBATCH --time=48:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

self_resolve_python

MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_seed_shared_135m_50kpd_${TS}}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

CMD=(
  "${PYTHON_BIN}" -m self.self_improvement
  --model-name "${MODEL_NAME}"
  --output-dir "${OUT_ROOT}"
  --bf16
  --resume
  --keep-checkpoints
  --seed "${SEED}"
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 50000
  --initial-eval-per-digit 100
  --num-expand-rounds 0
  --expand-num-digits 2
  --expand-train-per-digit 1200
  --eval-per-digit 100
  --composed-eval-per-digit 50
  --pseudo-label-mode none
  --num-epochs 3
  --learning-rate 5e-5
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
  --gradient-accumulation-steps 1
  --weight-decay 0
  --logging-steps 100
  --decode-max-new-tokens 48
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${EXTRA_ARGS})
fi

self_print_context \
  "Output root" "${OUT_ROOT}" \
  "Shared checkpoint path (after completion)" "${OUT_ROOT}/seed_model" \
  "Python" "${PYTHON_BIN}" \
  "Model" "${MODEL_NAME}" \
  "Seed" "${SEED}" \
  "Dry run" "${DRY_RUN}" \
  "Train batch size" "${TRAIN_BATCH_SIZE}" \
  "Eval batch size" "${EVAL_BATCH_SIZE}" \
  "Slurm job id" "${SLURM_JOB_ID:-unknown}" \
  "Hostname" "$(hostname)"
nvidia-smi -L || true

self_print_command_stdout "${CMD[@]}"

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; command not executed."
  exit 0
fi

"${CMD[@]}"

ln -sfn "${OUT_ROOT}/round_00" "${OUT_ROOT}/seed_model"
echo "[INFO] Seed checkpoint ready at ${OUT_ROOT}/seed_model"

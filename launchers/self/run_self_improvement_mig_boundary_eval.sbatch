#!/usr/bin/env bash
#SBATCH --job-name=selfimp-boundary
#SBATCH --output=artifacts/logs/selfimp-boundary-%j.out
#SBATCH --error=artifacts/logs/selfimp-boundary-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --time=02:59:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launchers/self/lib/self_common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/launchers/self"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
# shellcheck source=launchers/self/lib/self_common.sh
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root
self_resolve_python

DEFAULT_LOCAL_MODEL="${ROOT_DIR}/artifacts/models/SmolLM2-360M"
DEFAULT_HUB_MODEL="HuggingFaceTB/SmolLM2-360M"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  DEFAULT_MODEL="${DEFAULT_LOCAL_MODEL}"
else
  DEFAULT_MODEL="${DEFAULT_HUB_MODEL}"
fi
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL}}"
SEED="${SEED:-42}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/self_improvement/mig_boundary_eval_${TS}}"

# Defaults are tuned for a MIG slice; override with EXTRA_ARGS if needed.
COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --seed "${SEED}"
  --bf16
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 2
  --gradient-accumulation-steps 8
  --num-epochs 1
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 1200
  --initial-eval-per-digit 50
  --num-expand-rounds 8
  --expand-num-digits 2
  --expand-train-per-digit 1200
  --eval-per-digit 100
  --composed-eval-per-digit 50
  --pseudo-label-mode compose
  --decode-max-new-tokens 48
  --logging-steps 20
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL_NAME}"
self_print_torch_cuda_info
self_preflight_model_snapshot "${MODEL_NAME}"

echo "[INFO] Run 1/2: with_carry (no filtering)"
"${PYTHON_BIN}" -m self.self_improvement \
  --output-dir "${OUT_ROOT}/with_carry" \
  --composed-strategy with_carry \
  "${COMMON_ARGS[@]}"

echo "[INFO] Run 2/2: with_carry_filtered (0% boundary-carry retention)"
"${PYTHON_BIN}" -m self.experiments.composition_error_sweep \
  --composition-error-percent 0 \
  -- \
  --output-dir "${OUT_ROOT}/with_carry_filtered_p0" \
  "${COMMON_ARGS[@]}"

echo "[INFO] Finished MIG boundary-carry slice experiment."

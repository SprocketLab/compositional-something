#!/usr/bin/env bash
#SBATCH --job-name=selfimp-workshop
#SBATCH --output=artifacts/logs/selfimp-workshop-%j.out
#SBATCH --error=artifacts/logs/selfimp-workshop-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --qos=gpu-short
#SBATCH --time=08:00:00

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
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/self_improvement/workshop_small_model_${TS}}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --output-dir "${OUT_ROOT}"
  --seed "${SEED}"
  --bf16
  --per-device-train-batch-size 1
  --per-device-eval-batch-size 2
  --gradient-accumulation-steps 8
  --num-epochs 1
  --initial-min-digits 3
  --initial-max-digits 7
  --initial-train-per-digit 2000
  --initial-eval-per-digit 100
  --num-expand-rounds 8
  --expand-num-digits 2
  --expand-train-per-digit 1200
  --eval-per-digit 100
  --composed-eval-per-digit 50
  --pseudo-label-mode compose
  --composed-strategy with_carry
  --composed-refresh-mode dynamic
  --decode-max-new-tokens 48
  --logging-steps 20
)

if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  COMMON_ARGS+=(${EXTRA_ARGS})
fi

echo "[INFO] Output dir: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Slurm Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
nvidia-smi -L || true

"${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
MODEL_NAME="${MODEL_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from transformers import AutoConfig
from transformers import AutoTokenizer

model_name = os.environ["MODEL_NAME"]
model_path = Path(model_name)
if model_path.exists() and model_path.is_dir():
    has_weights = any(
        candidate.exists()
        for candidate in (
            model_path / "model.safetensors",
            model_path / "pytorch_model.bin",
        )
    ) or any(model_path.glob("model-*.safetensors")) or any(model_path.glob("pytorch_model-*.bin"))
    if not has_weights:
        raise FileNotFoundError(
            f"Local model directory {model_name} does not contain model weights. "
            "Set MODEL_NAME to a complete local snapshot or a Hugging Face model id."
        )
AutoConfig.from_pretrained(model_name, trust_remote_code=True)
AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f"[INFO] model_preflight_ok={model_name}")
PY

"${PYTHON_BIN}" -m self.self_improvement "${COMMON_ARGS[@]}"

echo "[INFO] Finished workshop small-model self-improvement run."

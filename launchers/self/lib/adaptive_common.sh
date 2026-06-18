#!/usr/bin/env bash
# Shared setup helpers for adaptive self-improvement Slurm launchers.

_ADAPTIVE_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/self/lib/self_common.sh
source "${_ADAPTIVE_COMMON_DIR}/self_common.sh"
unset _ADAPTIVE_COMMON_DIR

adaptive_cd_repo_root() {
  self_cd_repo_root
}

adaptive_setup_runtime_env() {
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"
  export HF_HOME="${HF_HOME:-$(dirname "${ROOT_DIR}")/hf_cache}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
  export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_XET_CACHE}"
}

adaptive_resolve_python() {
  self_resolve_python
}

adaptive_source_config_file() {
  self_source_config_file "${1:-}" "adaptive config"
}

adaptive_source_config_files() {
  local raw="${ADAPTIVE_CONFIG_FILES:-${ADAPTIVE_CONFIG_FILE:-}}"
  self_source_config_files "${raw}" "adaptive config"
}

adaptive_print_worker_context() {
  local label="$1"
  echo "[INFO] ${label}"
  echo "[INFO] Python: ${PYTHON_BIN}"
  echo "[INFO] Slurm Job ID: ${SLURM_JOB_ID:-unknown}"
  echo "[INFO] Hostname: $(hostname)"
  nvidia-smi -L || true
}

adaptive_print_torch_probe() {
  "${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
}

adaptive_set_sbatch_defaults() {
  self_set_sbatch_defaults "$@"
}

adaptive_add_sbatch_resources() {
  self_add_sbatch_resources "$@"
}

adaptive_print_command() {
  self_print_command "$@"
}

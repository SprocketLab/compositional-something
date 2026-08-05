# Per-cluster settings. Edit this one file after a move; the *_portable.slurm
# launchers read everything from here.
#
# Sourced by launchers, so keep it free of side effects.

# Repo root. Default assumes this file lives at <repo>/reports/composition_screen/.
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# Python with torch/transformers/peft. `which python` after activating the env.
export PY="${PY:-python}"

# Where HuggingFace caches models. Needs ~40 GB if you mirror the current cache,
# or ~9 GB for Qwen3.5-4B alone.
export HF_HOME="${HF_HOME:-$REPO/../hf_cache}"

# Slurm. PARTITION and GRES differ per site; check `sinfo -s`.
export PARTITION="${PARTITION:-gpu}"
export GRES="${GRES:-gpu:1}"
export CPUS="${CPUS:-8}"
export MEM="${MEM:-64G}"

# Base model. Resolved from the cache by name so a re-download that lands on a
# different snapshot hash does not break the launcher.
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-4B}"
resolve_model() {
  local repo_dir="$HF_HOME/hub/models--${MODEL_NAME/\//--}"
  local snap
  snap="$(ls -d "$repo_dir"/snapshots/* 2>/dev/null | head -1)"
  if [ -z "$snap" ]; then
    echo "MODEL NOT CACHED: $MODEL_NAME under $repo_dir" >&2
    echo "  fetch with: HF_HOME=$HF_HOME $PY -c \"from huggingface_hub import snapshot_download; snapshot_download('$MODEL_NAME')\"" >&2
    return 1
  fi
  echo "$snap"
}

# 48 GB cards need this; see GPU_MEMORY_NOTES.md. Set to 0 on cards with room.
export GRAD_CKPT="${GRAD_CKPT:-1}"

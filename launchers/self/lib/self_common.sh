#!/usr/bin/env bash
# Shared setup and Slurm helpers for self-improvement launchers.

self_cd_repo_root() {
  local source_index
  local source_path
  local script_dir
  if [[ -n "${ROOT_DIR:-}" ]]; then
    :
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="${SLURM_SUBMIT_DIR}"
  else
    source_index=$((${#BASH_SOURCE[@]} - 1))
    source_path="${BASH_SOURCE[${source_index}]}"
    script_dir="$(cd "$(dirname "${source_path}")" && pwd)"
    ROOT_DIR="$(cd "${script_dir}/../.." && pwd)"
  fi
  export ROOT_DIR
  cd "${ROOT_DIR}"
  mkdir -p "${ROOT_DIR}/artifacts/logs"
}

self_resolve_python() {
  local torch_env_path
  torch_env_path="${TORCH_ENV_PATH:-${HOME}/.conda/envs/torch-env}"
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    :
  elif [[ -x "${torch_env_path}/bin/python" ]]; then
    PYTHON_BIN="${torch_env_path}/bin/python"
  else
    PYTHON_BIN="python"
  fi
  export PYTHON_BIN
}

self_set_sbatch_defaults() {
  local default_partition="$1"
  local default_gres="$2"
  local default_cpus="$3"
  local default_mem="$4"
  local default_time="$5"
  SBATCH_PARTITION="${SBATCH_PARTITION:-${default_partition}}"
  SBATCH_GRES="${SBATCH_GRES:-${default_gres}}"
  SBATCH_CPUS="${SBATCH_CPUS:-${default_cpus}}"
  SBATCH_MEM="${SBATCH_MEM:-${default_mem}}"
  SBATCH_TIME="${SBATCH_TIME:-${default_time}}"
  export SBATCH_PARTITION SBATCH_GRES SBATCH_CPUS SBATCH_MEM SBATCH_TIME
}

self_add_sbatch_resources() {
  local array_name="$1"
  eval "${array_name}+=(--partition \"\${SBATCH_PARTITION}\" --gres \"\${SBATCH_GRES}\" --cpus-per-task \"\${SBATCH_CPUS}\" --mem \"\${SBATCH_MEM}\" --time \"\${SBATCH_TIME}\")"
  if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then
    eval "${array_name}+=(--constraint \"\${SBATCH_CONSTRAINT}\")"
  fi
}

self_add_sbatch_explicit_resources() {
  local array_name="$1"
  local partition="$2"
  local gres="$3"
  local cpus="$4"
  local mem="$5"
  local time_limit="$6"
  eval "${array_name}+=(--partition \"\${partition}\" --cpus-per-task \"\${cpus}\" --mem \"\${mem}\" --time \"\${time_limit}\")"
  if [[ -n "${gres}" ]]; then
    eval "${array_name}+=(--gres \"\${gres}\")"
  fi
  if [[ -n "${SBATCH_CONSTRAINT:-}" ]]; then
    eval "${array_name}+=(--constraint \"\${SBATCH_CONSTRAINT}\")"
  fi
}

self_print_command() {
  printf '[INFO] Command:' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

self_parse_bool() {
  local value
  value="$(echo "${1}" | tr '[:upper:]' '[:lower:]')"
  case "${value}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

self_add_dry_run_arg() {
  local array_name="$1"
  local dry_run_value="${2:-${DRY_RUN:-0}}"
  if self_parse_bool "${dry_run_value}"; then
    eval "${array_name}+=(--dry-run)"
  fi
}

self_print_context() {
  if (( $# % 2 != 0 )); then
    echo "[ERROR] self_print_context expects label/value pairs." >&2
    return 2
  fi
  while (( $# > 0 )); do
    echo "[INFO] $1: $2"
    shift 2
  done
}

self_print_prefixed_command_stdout() {
  local label="$1"
  shift
  printf '[INFO] %s:' "${label}"
  printf ' %q' "$@"
  printf '\n'
}

self_print_command_stdout() {
  self_print_prefixed_command_stdout "Command" "$@"
}

self_print_prefixed_command() {
  local label="$1"
  shift
  printf '[INFO] %s:' "${label}" >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

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

self_source_config_file() {
  local config_path="${1:-}"
  local label="${2:-config}"
  if [[ -z "${config_path}" ]]; then
    return 0
  fi
  if [[ "${config_path}" != /* ]]; then
    config_path="${ROOT_DIR:-$(pwd)}/${config_path}"
  fi
  if [[ ! -f "${config_path}" ]]; then
    echo "[ERROR] Missing ${label} file: ${config_path}" >&2
    return 2
  fi
  # shellcheck source=/dev/null
  source "${config_path}"
  echo "[INFO] Loaded ${label}: ${config_path}"
}

self_source_config_files() {
  local raw="${1:-}"
  local label="${2:-config}"
  local config_path
  if [[ -z "${raw}" ]]; then
    return 0
  fi
  IFS=':' read -r -a _self_config_paths <<< "${raw}"
  for config_path in "${_self_config_paths[@]}"; do
    self_source_config_file "${config_path}" "${label}"
  done
  unset _self_config_paths
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

self_submit_wrapped_job() {
  local job_name="$1"
  local stdout_log="$2"
  local stderr_log="$3"
  local partition="$4"
  local gres="$5"
  local cpus="$6"
  local mem="$7"
  local time_limit="$8"
  local dependency="${9:-}"
  shift 9 || true
  local wrap_cmd="$*"
  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --output "${stdout_log}"
    --error "${stderr_log}"
  )
  self_add_sbatch_explicit_resources sbatch_cmd "${partition}" "${gres}" "${cpus}" "${mem}" "${time_limit}"
  if [[ -n "${dependency}" ]]; then
    sbatch_cmd+=(--dependency "${dependency}")
  fi
  sbatch_cmd+=(--wrap "${wrap_cmd}")
  self_print_command "${sbatch_cmd[@]}"
  if self_parse_bool "${DRY_RUN:-0}"; then
    echo "dryrun-${job_name}"
  else
    "${sbatch_cmd[@]}" | cut -d';' -f1
  fi
}

self_submit_sbatch_command() {
  local dry_run_job_id="$1"
  shift
  self_print_prefixed_command "Submit" "$@"
  if self_parse_bool "${DRY_RUN:-0}"; then
    echo "${dry_run_job_id}"
  else
    "$@"
  fi
}

self_submit_sbatch_script() {
  local dry_run_job_id="$1"
  local job_name="$2"
  local stdout_log="$3"
  local stderr_log="$4"
  local export_vars="$5"
  shift 5
  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --output "${stdout_log}"
    --error "${stderr_log}"
  )
  if [[ -n "${export_vars}" ]]; then
    sbatch_cmd+=(--export "${export_vars}")
  fi
  sbatch_cmd+=("$@")
  self_print_command "${sbatch_cmd[@]}"
  if self_parse_bool "${DRY_RUN:-0}"; then
    echo "${dry_run_job_id}"
  else
    "${sbatch_cmd[@]}" | cut -d';' -f1
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

self_shell_quote() {
  printf "%q" "$1"
}

self_wrap_env_command() {
  if (( $# < 2 )); then
    echo "[ERROR] self_wrap_env_command expects a command path followed by zero or more NAME=VALUE pairs." >&2
    return 2
  fi
  local command_path="$1"
  shift
  local env_pair
  local env_name
  local env_value
  local wrapped
  wrapped="cd $(self_shell_quote "${ROOT_DIR}") && "
  for env_pair in "$@"; do
    env_name="${env_pair%%=*}"
    env_value="${env_pair#*=}"
    wrapped+="${env_name}=$(self_shell_quote "${env_value}") "
  done
  wrapped+="bash $(self_shell_quote "${command_path}")"
  printf '%s' "${wrapped}"
}

self_wrap_repo_command() {
  if (( $# < 1 )); then
    echo "[ERROR] self_wrap_repo_command expects a command and optional arguments." >&2
    return 2
  fi
  local arg
  local wrapped
  wrapped="cd $(self_shell_quote "${ROOT_DIR}") && PYTHONPATH=."
  for arg in "$@"; do
    wrapped+=" $(self_shell_quote "${arg}")"
  done
  printf '%s' "${wrapped}"
}

self_submit_wrapped_resource_job() {
  local dry_run_job_id="$1"
  local job_name="$2"
  local stdout_log="$3"
  local stderr_log="$4"
  local wrapped_cmd="$5"
  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --output "${stdout_log}"
    --error "${stderr_log}"
  )
  self_add_sbatch_resources sbatch_cmd
  sbatch_cmd+=(--wrap "${wrapped_cmd}")
  self_print_command "${sbatch_cmd[@]}"
  if self_parse_bool "${DRY_RUN:-0}"; then
    echo "${dry_run_job_id}"
  else
    "${sbatch_cmd[@]}"
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

self_print_python_launcher_context() {
  local out_root="$1"
  shift
  self_print_context \
    "Root dir" "${ROOT_DIR}" \
    "Python" "${PYTHON_BIN}" \
    "Output root" "${out_root}" \
    "$@"
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

self_print_and_run_command_stdout() {
  self_print_command_stdout "$@"
  "$@"
}

self_print_prefixed_command() {
  local label="$1"
  shift
  printf '[INFO] %s:' "${label}" >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
}

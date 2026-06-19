#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root
adaptive_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/adaptive_condition_pilots_ailab_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
RUNNER="${RUNNER:-${ROOT_DIR}/launchers/self/run_adaptive_condition_ailab.sbatch}"
PLAN_LOG_PATH="${PLAN_LOG_PATH:-${ROOT_DIR}/plan/260603-self-improvement-init.md}"
DRY_RUN="${DRY_RUN:-0}"

adaptive_set_sbatch_defaults "ailab" "gpu:h200:1" "4" "48G" "01:00:00"
MIN_EXAMPLES_PER_SIZE="${MIN_EXAMPLES_PER_SIZE:-1}"
MAX_EXAMPLES_PER_SIZE="${MAX_EXAMPLES_PER_SIZE:-128}"
FRONTIER_POLICY="${FRONTIER_POLICY:-fixed}"
FRONTIER_MIN_COUNT="${FRONTIER_MIN_COUNT:-1}"
FRONTIER_MAX_ACCURACY="${FRONTIER_MAX_ACCURACY:-0.85}"
FRONTIER_MAX_WIDTH="${FRONTIER_MAX_WIDTH:-1}"
FRONTIER_PREFER_LARGER_WEIGHT="${FRONTIER_PREFER_LARGER_WEIGHT:-0.01}"
ENFORCE_SELECTED_FRONTIER="${ENFORCE_SELECTED_FRONTIER:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

submit_condition() {
  local job_key="$1"
  local task="$2"
  local condition="$3"
  local fixture="$4"
  local source_min_allowed="$5"
  local source_max_allowed="$6"
  local frontier_min_allowed="$7"
  local frontier_max_allowed="$8"
  local out_dir="${OUT_ROOT}/${job_key}"
  local export_vars
  local -a export_pairs=(
    "TASK=${task}"
    "CONDITION=${condition}"
    "FIXTURE=${fixture}"
    "OUT_DIR=${out_dir}"
    "SOURCE_MIN_ALLOWED=${source_min_allowed}"
    "SOURCE_MAX_ALLOWED=${source_max_allowed}"
    "FRONTIER_MIN_ALLOWED=${frontier_min_allowed}"
    "FRONTIER_MAX_ALLOWED=${frontier_max_allowed}"
    "FRONTIER_POLICY=${FRONTIER_POLICY}"
    "FRONTIER_MIN_COUNT=${FRONTIER_MIN_COUNT}"
    "FRONTIER_MAX_ACCURACY=${FRONTIER_MAX_ACCURACY}"
    "FRONTIER_MAX_WIDTH=${FRONTIER_MAX_WIDTH}"
    "FRONTIER_PREFER_LARGER_WEIGHT=${FRONTIER_PREFER_LARGER_WEIGHT}"
    "ENFORCE_SELECTED_FRONTIER=${ENFORCE_SELECTED_FRONTIER}"
    "MIN_EXAMPLES_PER_SIZE=${MIN_EXAMPLES_PER_SIZE}"
    "MAX_EXAMPLES_PER_SIZE=${MAX_EXAMPLES_PER_SIZE}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "PLAN_LOG_PATH=${PLAN_LOG_PATH}"
  )
  if [[ -n "${FRONTIER_DIAGNOSTICS_PATH:-}" ]]; then
    export_pairs+=("FRONTIER_DIAGNOSTICS_PATH=${FRONTIER_DIAGNOSTICS_PATH}")
  fi
  export_vars="$(self_sbatch_export_all "${export_pairs[@]}")"
  local -a resource_args=()
  adaptive_add_sbatch_resources resource_args
  self_submit_sbatch_script \
    "dryrun-adapt-${job_key}" \
    "adapt-${job_key}" \
    "${LOG_DIR}/adapt-${job_key}-%j.out" \
    "${LOG_DIR}/adapt-${job_key}-%j.err" \
    "${export_vars}" \
    "${resource_args[@]}" \
    "${RUNNER}"
}

ADDITION_CONFIG_FIXTURE="${ADDITION_CONFIG_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_addition_config_fixture.jsonl}"
ADDITION_PROGRAM_FIXTURE="${ADDITION_PROGRAM_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_addition_program_fixture.jsonl}"
RUN_LENGTH_CONFIG_FIXTURE="${RUN_LENGTH_CONFIG_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_run_length_config_fixture.jsonl}"
RUN_LENGTH_PROGRAM_FIXTURE="${RUN_LENGTH_PROGRAM_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_run_length_program_fixture.jsonl}"

ADDITION_CONFIG_JOB_ID="$(
  submit_condition \
    "addition-config" \
    "addition" \
    "config" \
    "${ADDITION_CONFIG_FIXTURE}" \
    3 \
    7 \
    8 \
    31
)"
ADDITION_PROGRAM_JOB_ID="$(
  submit_condition \
    "addition-program" \
    "addition" \
    "program" \
    "${ADDITION_PROGRAM_FIXTURE}" \
    3 \
    7 \
    8 \
    31
)"
RUN_LENGTH_CONFIG_JOB_ID="$(
  submit_condition \
    "run-length-config" \
    "run_length" \
    "config" \
    "${RUN_LENGTH_CONFIG_FIXTURE}" \
    8 \
    16 \
    17 \
    48
)"
RUN_LENGTH_PROGRAM_JOB_ID="$(
  submit_condition \
    "run-length-program" \
    "run_length" \
    "program" \
    "${RUN_LENGTH_PROGRAM_FIXTURE}" \
    8 \
    16 \
    17 \
    48
)"

MANIFEST="${OUT_ROOT}/submission_manifest.json"
"${PYTHON_BIN}" -m self.launcher_manifests adaptive-condition \
  --manifest "${MANIFEST}" \
  --out-root "${OUT_ROOT}" \
  --partition "${SBATCH_PARTITION}" \
  --gres "${SBATCH_GRES}" \
  --time-limit "${SBATCH_TIME}" \
  --cpus-per-task "${SBATCH_CPUS}" \
  --mem "${SBATCH_MEM}" \
  --frontier-policy "${FRONTIER_POLICY}" \
  --frontier-min-count "${FRONTIER_MIN_COUNT}" \
  --frontier-max-accuracy "${FRONTIER_MAX_ACCURACY}" \
  --frontier-max-width "${FRONTIER_MAX_WIDTH}" \
  --frontier-prefer-larger-weight "${FRONTIER_PREFER_LARGER_WEIGHT}" \
  --enforce-selected-frontier "${ENFORCE_SELECTED_FRONTIER}" \
  --frontier-diagnostics-path "${FRONTIER_DIAGNOSTICS_PATH:-}" \
  --addition-config-job-id "${ADDITION_CONFIG_JOB_ID}" \
  --addition-program-job-id "${ADDITION_PROGRAM_JOB_ID}" \
  --run-length-config-job-id "${RUN_LENGTH_CONFIG_JOB_ID}" \
  --run-length-program-job-id "${RUN_LENGTH_PROGRAM_JOB_ID}"

echo "[INFO] Submitted split adaptive condition jobs."
echo "[INFO] Manifest: ${MANIFEST}"

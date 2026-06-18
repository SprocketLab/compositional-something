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

adaptive_set_sbatch_defaults "ailab" "gpu:h200:1" "4" "96G" "01:00:00"
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
  export_vars="ALL,TASK=${task},CONDITION=${condition},FIXTURE=${fixture},OUT_DIR=${out_dir},SOURCE_MIN_ALLOWED=${source_min_allowed},SOURCE_MAX_ALLOWED=${source_max_allowed},FRONTIER_MIN_ALLOWED=${frontier_min_allowed},FRONTIER_MAX_ALLOWED=${frontier_max_allowed},FRONTIER_POLICY=${FRONTIER_POLICY},FRONTIER_MIN_COUNT=${FRONTIER_MIN_COUNT},FRONTIER_MAX_ACCURACY=${FRONTIER_MAX_ACCURACY},FRONTIER_MAX_WIDTH=${FRONTIER_MAX_WIDTH},FRONTIER_PREFER_LARGER_WEIGHT=${FRONTIER_PREFER_LARGER_WEIGHT},ENFORCE_SELECTED_FRONTIER=${ENFORCE_SELECTED_FRONTIER},MIN_EXAMPLES_PER_SIZE=${MIN_EXAMPLES_PER_SIZE},MAX_EXAMPLES_PER_SIZE=${MAX_EXAMPLES_PER_SIZE},PYTHON_BIN=${PYTHON_BIN},PLAN_LOG_PATH=${PLAN_LOG_PATH}"
  if [[ -n "${FRONTIER_DIAGNOSTICS_PATH:-}" ]]; then
    export_vars="${export_vars},FRONTIER_DIAGNOSTICS_PATH=${FRONTIER_DIAGNOSTICS_PATH}"
  fi
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
"${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${OUT_ROOT}" "${SBATCH_PARTITION}" "${SBATCH_GRES}" "${SBATCH_TIME}" "${SBATCH_CPUS}" "${SBATCH_MEM}" "${FRONTIER_POLICY}" "${FRONTIER_MIN_COUNT}" "${FRONTIER_MAX_ACCURACY}" "${FRONTIER_MAX_WIDTH}" "${FRONTIER_PREFER_LARGER_WEIGHT}" "${ENFORCE_SELECTED_FRONTIER}" "${FRONTIER_DIAGNOSTICS_PATH:-}" "${ADDITION_CONFIG_JOB_ID}" "${ADDITION_PROGRAM_JOB_ID}" "${RUN_LENGTH_CONFIG_JOB_ID}" "${RUN_LENGTH_PROGRAM_JOB_ID}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_root = sys.argv[2]
payload = {
    "out_root": out_root,
    "slurm": {
        "partition": sys.argv[3],
        "gres": sys.argv[4],
        "time": sys.argv[5],
        "cpus_per_task": sys.argv[6],
        "mem": sys.argv[7],
        "frontier_policy": sys.argv[8],
        "frontier_min_count": sys.argv[9],
        "frontier_max_accuracy": sys.argv[10],
        "frontier_max_width": sys.argv[11],
        "frontier_prefer_larger_weight": sys.argv[12],
        "enforce_selected_frontier": sys.argv[13],
        "frontier_diagnostics_path": sys.argv[14] or None,
    },
    "jobs": {
        "addition_config": {
            "job_id": sys.argv[15],
            "task": "addition",
            "condition": "config",
            "output_dir": f"{out_root}/addition-config",
        },
        "addition_program": {
            "job_id": sys.argv[16],
            "task": "addition",
            "condition": "program",
            "output_dir": f"{out_root}/addition-program",
        },
        "run_length_config": {
            "job_id": sys.argv[17],
            "task": "run_length",
            "condition": "config",
            "output_dir": f"{out_root}/run-length-config",
        },
        "run_length_program": {
            "job_id": sys.argv[18],
            "task": "run_length",
            "condition": "program",
            "output_dir": f"{out_root}/run-length-program",
        },
    },
    "scope_note": (
        "These are split adaptive proposal/preflight condition jobs. "
        "They do not yet run temporary LoRA self-edit training/evaluation loops."
    ),
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

echo "[INFO] Submitted split adaptive condition jobs."
echo "[INFO] Manifest: ${MANIFEST}"

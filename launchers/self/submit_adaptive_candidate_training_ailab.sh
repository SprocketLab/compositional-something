#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/adaptive_candidate_training_ailab_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT_DIR}/launchers/self/run_adaptive_candidate_training_ailab.sbatch}"
TASKS="${TASKS:-addition run_length}"
CONDITIONS="${CONDITIONS:-config}"
OUTCOME_TRACE_TARGET_MODES="${OUTCOME_TRACE_TARGET_MODES:-numeric}"
PROPOSAL_GRPO_ZERO_VARIANCE_MODES="${PROPOSAL_GRPO_ZERO_VARIANCE_MODES:-fixed_baseline}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
NUM_CANDIDATES_LIST="${NUM_CANDIDATES_LIST:-${NUM_CANDIDATES}}"
SBATCH_MEM="${SBATCH_MEM:-48G}"
ADAPTIVE_CONFIG_EXPORT="${ADAPTIVE_CONFIG_FILES:-${ADAPTIVE_CONFIG_FILE:-}}"

adaptive_resolve_python

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

submit_cell() {
  local task="$1"
  local condition="$2"
  local outcome_mode="$3"
  local zero_variance="$4"
  local num_candidates="$5"
  local task_slug
  task_slug="${task//_/-}"
  local condition_slug
  condition_slug="${condition//_/-}"
  local outcome_slug
  outcome_slug="${outcome_mode//_/-}"
  local zero_slug
  zero_slug="${zero_variance//_/-}"
  local out_dir="${OUT_ROOT}/${task}-${condition}-${outcome_slug}-n${num_candidates}-grpo-${zero_slug}"
  self_submit_sbatch_script \
    "dryrun-adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${zero_slug}" \
    "adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${zero_slug}" \
    "${LOG_DIR}/adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${zero_slug}-%j.out" \
    "${LOG_DIR}/adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${zero_slug}-%j.err" \
    "$(self_sbatch_export_all \
      "TASK=${task}" \
      "CONDITION=${condition}" \
      "OUTCOME_TRACE_TARGET_MODE=${outcome_mode}" \
      "PROPOSAL_GRPO_ZERO_VARIANCE=${zero_variance}" \
      "NUM_CANDIDATES=${num_candidates}" \
      "OUT_DIR=${out_dir}" \
      "ADAPTIVE_CONFIG_FILES=${ADAPTIVE_CONFIG_EXPORT}")" \
    --mem "${SBATCH_MEM}" \
    "${SBATCH_SCRIPT}"
}

declare -a MANIFEST_ARGS=()
for task in ${TASKS}; do
  for condition in ${CONDITIONS}; do
    for outcome_mode in ${OUTCOME_TRACE_TARGET_MODES}; do
      for zero_variance in ${PROPOSAL_GRPO_ZERO_VARIANCE_MODES}; do
        for num_candidates in ${NUM_CANDIDATES_LIST}; do
          outcome_slug="${outcome_mode//_/-}"
          zero_slug="${zero_variance//_/-}"
          job_id="$(submit_cell "${task}" "${condition}" "${outcome_mode}" "${zero_variance}" "${num_candidates}")"
          MANIFEST_ARGS+=(
            "${task}"
            "${condition}"
            "${outcome_mode}"
            "${zero_variance}"
            "${num_candidates}"
            "${job_id}"
            "${OUT_ROOT}/${task}-${condition}-${outcome_slug}-n${num_candidates}-grpo-${zero_slug}"
          )
        done
      done
    done
  done
done

MANIFEST="${OUT_ROOT}/submission_manifest.json"
"${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${OUT_ROOT}" "${TASKS}" "${CONDITIONS}" "${OUTCOME_TRACE_TARGET_MODES}" "${PROPOSAL_GRPO_ZERO_VARIANCE_MODES}" "${NUM_CANDIDATES_LIST}" "${ADAPTIVE_CONFIG_EXPORT}" "${MANIFEST_ARGS[@]}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
out_root = sys.argv[2]
tasks = sys.argv[3].split()
conditions = sys.argv[4].split()
outcome_modes = sys.argv[5].split()
zero_variance_modes = sys.argv[6].split()
num_candidates_list = [int(value) for value in sys.argv[7].split()]
adaptive_config_files = sys.argv[8]
items = sys.argv[9:]
jobs = {}
for index in range(0, len(items), 7):
    task, condition, outcome_mode, zero_variance, num_candidates, job_id, output_dir = items[index : index + 7]
    jobs[f"{task}-{condition}-{outcome_mode}-n{num_candidates}-grpo-{zero_variance}"] = {
        "task": task,
        "condition": condition,
        "outcome_trace_target_mode": outcome_mode,
        "proposal_grpo_zero_variance": zero_variance,
        "num_candidates": int(num_candidates),
        "job_id": job_id,
        "output_dir": output_dir,
        "status": "submitted",
    }
payload = {
    "out_root": out_root,
    "tasks": tasks,
    "conditions": conditions,
    "outcome_trace_target_modes": outcome_modes,
    "proposal_grpo_zero_variance_modes": zero_variance_modes,
    "num_candidates_list": num_candidates_list,
    "adaptive_config_files": adaptive_config_files,
    "jobs": jobs,
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

echo "[INFO] Submitted adaptive candidate-training jobs."
echo "[INFO] Manifest: ${MANIFEST}"

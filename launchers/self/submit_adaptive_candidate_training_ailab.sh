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
"${PYTHON_BIN}" -m self.launcher_manifests adaptive-candidate \
  --manifest "${MANIFEST}" \
  --out-root "${OUT_ROOT}" \
  --tasks "${TASKS}" \
  --conditions "${CONDITIONS}" \
  --outcome-trace-target-modes "${OUTCOME_TRACE_TARGET_MODES}" \
  --proposal-grpo-zero-variance-modes "${PROPOSAL_GRPO_ZERO_VARIANCE_MODES}" \
  --num-candidates-list "${NUM_CANDIDATES_LIST}" \
  --adaptive-config-files "${ADAPTIVE_CONFIG_EXPORT}" \
  --job-fields "${MANIFEST_ARGS[@]}"

echo "[INFO] Submitted adaptive candidate-training jobs."
echo "[INFO] Manifest: ${MANIFEST}"

#!/usr/bin/env bash
#SBATCH --job-name=adaptive-condition
#SBATCH --output=artifacts/logs/adaptive-condition-%j.out
#SBATCH --error=artifacts/logs/adaptive-condition-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root

: "${TASK:?TASK must be set to addition or run_length}"
: "${CONDITION:?CONDITION must be set to config or program}"
: "${FIXTURE:?FIXTURE must point to a proposal fixture jsonl file}"
: "${OUT_DIR:?OUT_DIR must point to the output directory for this condition}"

mkdir -p "${OUT_DIR}"
adaptive_setup_runtime_env
adaptive_resolve_python

CURRENT_FRONTIER_MIN="${CURRENT_FRONTIER_MIN:-0}"
CURRENT_FRONTIER_MAX="${CURRENT_FRONTIER_MAX:-0}"
SOURCE_MIN_ALLOWED="${SOURCE_MIN_ALLOWED:-${CURRENT_FRONTIER_MIN}}"
SOURCE_MAX_ALLOWED="${SOURCE_MAX_ALLOWED:-${CURRENT_FRONTIER_MAX}}"
FRONTIER_MIN_ALLOWED="${FRONTIER_MIN_ALLOWED:-1}"
FRONTIER_MAX_ALLOWED="${FRONTIER_MAX_ALLOWED:-32}"
FRONTIER_POLICY="${FRONTIER_POLICY:-fixed}"
FRONTIER_MIN_COUNT="${FRONTIER_MIN_COUNT:-1}"
FRONTIER_MAX_ACCURACY="${FRONTIER_MAX_ACCURACY:-0.85}"
FRONTIER_MAX_WIDTH="${FRONTIER_MAX_WIDTH:-1}"
FRONTIER_PREFER_LARGER_WEIGHT="${FRONTIER_PREFER_LARGER_WEIGHT:-0.01}"
ENFORCE_SELECTED_FRONTIER="${ENFORCE_SELECTED_FRONTIER:-0}"
MIN_EXAMPLES_PER_SIZE="${MIN_EXAMPLES_PER_SIZE:-1}"
MAX_EXAMPLES_PER_SIZE="${MAX_EXAMPLES_PER_SIZE:-128}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
PROGRAM_TIMEOUT_SECONDS="${PROGRAM_TIMEOUT_SECONDS:-1.0}"
PLAN_LOG_PATH="${PLAN_LOG_PATH:-${ROOT_DIR}/plan/260603-self-improvement-init.md}"
AGGREGATE_METRICS_JSON="${AGGREGATE_METRICS_JSON:-{\"frontier_accuracy\":0.42,\"final_accuracy\":0.90,\"slice\":\"ailab_split_condition\"}}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"
FRONTIER_ARGS=(
  --frontier-policy "${FRONTIER_POLICY}"
  --frontier-min-count "${FRONTIER_MIN_COUNT}"
  --frontier-max-accuracy "${FRONTIER_MAX_ACCURACY}"
  --frontier-max-width "${FRONTIER_MAX_WIDTH}"
  --frontier-prefer-larger-weight "${FRONTIER_PREFER_LARGER_WEIGHT}"
)
if [[ -n "${FRONTIER_DIAGNOSTICS_PATH:-}" ]]; then
  FRONTIER_ARGS+=(--frontier-diagnostics-path "${FRONTIER_DIAGNOSTICS_PATH}")
fi
if [[ -n "${FRONTIER_DIAGNOSTICS_JSON:-}" ]]; then
  FRONTIER_ARGS+=(--frontier-diagnostics-json "${FRONTIER_DIAGNOSTICS_JSON}")
fi
if [[ "${ENFORCE_SELECTED_FRONTIER}" == "1" ]]; then
  FRONTIER_ARGS+=(--enforce-selected-frontier)
fi

self_print_context \
  "Task/condition" "${TASK}/${CONDITION}" \
  "Output dir" "${OUT_DIR}" \
  "Fixture" "${FIXTURE}"
adaptive_print_worker_context "Adaptive proposal condition"
adaptive_print_torch_probe

if [[ "${RUN_COMPILE_CHECK}" == "1" ]]; then
  echo "[INFO] Compile check"
  "${PYTHON_BIN}" -m py_compile \
    self/adaptive/proposals/__init__.py \
    self/adaptive/sandbox/program_sandbox.py \
    self/adaptive_self_improvement.py
fi

echo "[INFO] Run adaptive proposal condition"
"${PYTHON_BIN}" -m self.experiments.adaptive_self_improvement \
  --task "${TASK}" \
  --condition "${CONDITION}" \
  --proposal-fixture-jsonl "${FIXTURE}" \
  --output-dir "${OUT_DIR}" \
  --dry-run-proposals \
  --source-min-allowed "${SOURCE_MIN_ALLOWED}" \
  --source-max-allowed "${SOURCE_MAX_ALLOWED}" \
  --frontier-min-allowed "${FRONTIER_MIN_ALLOWED}" \
  --frontier-max-allowed "${FRONTIER_MAX_ALLOWED}" \
  "${FRONTIER_ARGS[@]}" \
  --min-examples-per-size "${MIN_EXAMPLES_PER_SIZE}" \
  --max-examples-per-size "${MAX_EXAMPLES_PER_SIZE}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --program-timeout-seconds "${PROGRAM_TIMEOUT_SECONDS}" \
  --aggregate-metrics-json "${AGGREGATE_METRICS_JSON}" \
  --plan-log-path "${PLAN_LOG_PATH}"

echo "[INFO] Finished adaptive proposal condition."

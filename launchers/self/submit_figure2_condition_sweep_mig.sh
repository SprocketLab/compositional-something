#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure2_condition_sweep_${TS}}"
SELECTION_JSON="${SELECTION_JSON:-${ROOT_DIR}/artifacts/paper/paper_schedule_selection.json}"
PAPER_SCHEDULE_ENV="${PAPER_SCHEDULE_ENV:-${ROOT_DIR}/artifacts/paper/paper_schedule_selection.env}"
FIGURE_DIR="${FIGURE_DIR:-${ROOT_DIR}/icmlw26_comp-self-improvement/figures}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${PYTHON_BIN}"
  -m
  self.figure2_condition_sweep
  submit
  --out-root "${OUT_ROOT}"
  --selection-json "${SELECTION_JSON}"
  --paper-schedule-env "${PAPER_SCHEDULE_ENV}"
  --figure-dir "${FIGURE_DIR}"
  --log-dir "${LOG_DIR}"
  --python-bin "${PYTHON_BIN}"
)

self_add_dry_run_arg cmd "${DRY_RUN}"

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Selection JSON" "${SELECTION_JSON}" \
  "Paper schedule env" "${PAPER_SCHEDULE_ENV}" \
  "Figure dir" "${FIGURE_DIR}" \
  "Log dir" "${LOG_DIR}" \
  "Dry run" "${DRY_RUN}"
self_print_command_stdout "${cmd[@]}"

"${cmd[@]}"

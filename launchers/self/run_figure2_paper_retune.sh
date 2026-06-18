#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure2_paper_retune_${TS}}"
SELECTION_JSON="${SELECTION_JSON:-${OUT_ROOT}/paper_schedule_selection.json}"
PAPER_SCHEDULE_ENV="${PAPER_SCHEDULE_ENV:-${OUT_ROOT}/paper_schedule_selection.env}"
FIGURE_DIR="${FIGURE_DIR:-${ROOT_DIR}/icmlw26_comp-self-improvement/figures}"
SEED="${SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_LENGTH_FORCE_RERUN="${RUN_LENGTH_FORCE_RERUN:-0}"
SKIP_RENDER="${SKIP_RENDER:-0}"

cmd=(
  "${PYTHON_BIN}"
  -m
  self.figure2_paper_retune
  --output-dir "${OUT_ROOT}"
  --selection-json "${SELECTION_JSON}"
  --paper-schedule-env "${PAPER_SCHEDULE_ENV}"
  --figure-dir "${FIGURE_DIR}"
  --seed "${SEED}"
)

if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  cmd+=(--train-batch-size "${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
  cmd+=(--eval-batch-size "${EVAL_BATCH_SIZE}")
fi
if self_parse_bool "${RUN_LENGTH_FORCE_RERUN}"; then
  cmd+=(--run-length-force-rerun)
fi
if self_parse_bool "${SKIP_RENDER}"; then
  cmd+=(--skip-render)
fi
if self_parse_bool "${DRY_RUN}"; then
  cmd+=(--dry-run)
fi
if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Selection JSON: ${SELECTION_JSON}"
echo "[INFO] Paper schedule env: ${PAPER_SCHEDULE_ENV}"
echo "[INFO] Figure dir: ${FIGURE_DIR}"
self_print_command_stdout "${cmd[@]}"

"${cmd[@]}"

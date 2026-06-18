#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_tune_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/multiplication_rectangular_seed_best}"
STAGE1_MANIFEST="${STAGE1_MANIFEST:-${OUT_ROOT}/stage1_manifest.json}"
STAGE1_SELECTION="${STAGE1_SELECTION:-${OUT_ROOT}/stage1_selection.json}"
STAGE2_SELECTION="${STAGE2_SELECTION:-${OUT_ROOT}/stage2_selection.json}"
STAGE3_SELECTION="${STAGE3_SELECTION:-${OUT_ROOT}/stage3_selection.json}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${LOG_DIR}"

cmd=(
  "${PYTHON_BIN}" -m self.experiments.multiplication_rectangular_tune
  submit
  --out-root "${OUT_ROOT}"
  --log-dir "${LOG_DIR}"
  --seed-model "${SEED_MODEL}"
  --python-bin "${PYTHON_BIN}"
  --stage1-manifest "${STAGE1_MANIFEST}"
  --stage1-selection "${STAGE1_SELECTION}"
  --stage2-selection "${STAGE2_SELECTION}"
  --stage3-selection "${STAGE3_SELECTION}"
)

self_add_dry_run_arg cmd "${DRY_RUN}"

self_print_command_stdout "${cmd[@]}"
"${cmd[@]}"

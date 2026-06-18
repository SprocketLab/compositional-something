#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure3_seed_quality_sweep_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${PYTHON_BIN}"
  -m
  self.figure3_seed_quality_sweep
  submit
  --out-root "${OUT_ROOT}"
  --log-dir "${LOG_DIR}"
  --python-bin "${PYTHON_BIN}"
)

if self_parse_bool "${DRY_RUN}"; then
  cmd+=(--dry-run)
fi

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Log dir: ${LOG_DIR}"
echo "[INFO] Dry run: ${DRY_RUN}"
self_print_command_stdout "${cmd[@]}"

"${cmd[@]}"

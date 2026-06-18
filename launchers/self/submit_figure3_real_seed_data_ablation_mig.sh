#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure3_real_seed_data_ablation_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${PYTHON_BIN}"
  -m
  self.figure3_real_seed_data_ablation
  submit
  --out-root "${OUT_ROOT}"
  --log-dir "${LOG_DIR}"
  --python-bin "${PYTHON_BIN}"
)

self_add_dry_run_arg cmd "${DRY_RUN}"

self_print_python_launcher_context "${OUT_ROOT}" \
  "Log dir" "${LOG_DIR}" \
  "Dry run" "${DRY_RUN}"

self_print_and_run_command_stdout "${cmd[@]}"

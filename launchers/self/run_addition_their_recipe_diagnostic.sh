#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_their_recipe_diag_${TS}}"
DEVICE_TARGET="${DEVICE_TARGET:-local_a100_40gb}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
MAX_STEPS="${MAX_STEPS:-}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  "${PYTHON_BIN}" -m self.diagnostics.addition_recipe_diagnostic
  --output-dir "${OUT_ROOT}"
  --recipe arithmetic_self_improve_v1
  --device-target "${DEVICE_TARGET}"
)

if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  CMD+=(--per-device-train-batch-size "${TRAIN_BATCH_SIZE}")
fi
if [[ -n "${EVAL_BATCH_SIZE}" ]]; then
  CMD+=(--per-device-eval-batch-size "${EVAL_BATCH_SIZE}")
fi
if [[ -n "${MAX_STEPS}" ]]; then
  CMD+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=(${EXTRA_ARGS})
fi
if self_parse_bool "${DRY_RUN}"; then
  CMD+=(--dry-run)
fi

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Device target" "${DEVICE_TARGET}" \
  "Dry run" "${DRY_RUN}"
self_print_command_stdout "${CMD[@]}"

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; command not executed."
  exit 0
fi

"${CMD[@]}"

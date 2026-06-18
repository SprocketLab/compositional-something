#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_recipe_recovery_${TS}}"
DRY_RUN="${DRY_RUN:-0}"
DEVICE_TARGET="${DEVICE_TARGET:-local_a100_40gb}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
SEED_MODEL_LINK="${ROOT_DIR}/artifacts/models/addition_recipe_seed_best"

DIAG_LAUNCHER="${ROOT_DIR}/launchers/self/run_addition_their_recipe_diagnostic.sh"
FOCUSED_LAUNCHER="${ROOT_DIR}/launchers/self/run_addition_recipe_focused.sh"
FULLPACK_LAUNCHER="${ROOT_DIR}/launchers/self/run_addition_recipe_fullpack.sh"

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Output root" "${OUT_ROOT}" \
  "Device target" "${DEVICE_TARGET}" \
  "Dry run" "${DRY_RUN}"

env \
  DRY_RUN="${DRY_RUN}" \
  OUT_ROOT="${OUT_ROOT}/diagnostic" \
  DEVICE_TARGET="${DEVICE_TARGET}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  bash "${DIAG_LAUNCHER}"

if self_parse_bool "${DRY_RUN}"; then
  env \
    DRY_RUN=1 \
    OUT_ROOT="${OUT_ROOT}/focused" \
    BASELINE=with_carry_filtered \
    SEED_MODEL="${SEED_MODEL_LINK}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash "${FOCUSED_LAUNCHER}"

  env \
    DRY_RUN=1 \
    OUT_ROOT="${OUT_ROOT}/fullpack" \
    SEED_MODEL="${SEED_MODEL_LINK}" \
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    bash "${FULLPACK_LAUNCHER}"
  exit 0
fi

python - <<'PY' "${OUT_ROOT}/diagnostic/summary.json"
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary = json.loads(summary_path.read_text(encoding="utf-8"))

seed_worst = float(summary["seed"]["worst_case_heldout_min_per_digit_accuracy"])
stage_89 = summary["frontier_8_9"]["heldout_eval"]
digit9 = float(stage_89["per_digit_accuracy"].get("9", 0.0) or 0.0)
stage_812 = summary["frontier_8_12"]["heldout_eval"]
overall_812 = float(stage_812["accuracy"] or 0.0)
per_digit_812 = {int(key): float(value or 0.0) for key, value in stage_812["per_digit_accuracy"].items()}

missing = [digit for digit in range(8, 13) if per_digit_812.get(digit, 0.0) <= 0.0]
print(
    f"[INFO] Diagnostic gates: seed_worst={seed_worst:.4f} digit9={digit9:.4f} "
    f"overall_8_12={overall_812:.4f} zero_digits={missing}"
)
if seed_worst < 0.95:
    raise SystemExit("Seed recipe gate failed.")
if digit9 < 0.40:
    raise SystemExit("8..9 frontier gate failed.")
if overall_812 < 0.20 or missing:
    raise SystemExit("8..12 frontier gate failed.")
PY

SEED_MODEL_TARGET="$(python - <<'PY' "${OUT_ROOT}/diagnostic/summary.json"
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(summary["seed"]["model_dir"])
PY
)"
mkdir -p "$(dirname "${SEED_MODEL_LINK}")"
ln -sfn "${SEED_MODEL_TARGET}" "${SEED_MODEL_LINK}"
echo "[INFO] Updated recipe seed link: ${SEED_MODEL_LINK} -> ${SEED_MODEL_TARGET}"

env \
  OUT_ROOT="${OUT_ROOT}/focused" \
  BASELINE=with_carry_filtered \
  SEED_MODEL="${SEED_MODEL_LINK}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  bash "${FOCUSED_LAUNCHER}"

python - <<'PY' "${OUT_ROOT}/focused/with_carry_filtered/round_01/metrics.json"
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
seed_eval = float(metrics.get("seed_eval_accuracy") or 0.0)
frontier_train = float(metrics.get("frontier_train_accuracy") or 0.0)
expanded_eval = float(metrics.get("expanded_eval_accuracy") or 0.0)
per_digit = {int(key): float(value or 0.0) for key, value in metrics.get("per_digit_accuracy", {}).items()}
nonzero_expanded = sum(1 for digit in range(8, 13) if per_digit.get(digit, 0.0) > 0.0)
print(
    f"[INFO] Focused round-1 gates: seed_eval={seed_eval:.4f} frontier_train={frontier_train:.4f} "
    f"expanded_eval={expanded_eval:.4f} nonzero_expanded={nonzero_expanded}"
)
if seed_eval < 0.95:
    raise SystemExit("Focused seed replay gate failed.")
if frontier_train < 0.90:
    raise SystemExit("Focused frontier-train gate failed.")
if expanded_eval < 0.10:
    raise SystemExit("Focused expanded-eval gate failed.")
if nonzero_expanded < 3:
    raise SystemExit("Focused nonzero-expanded-digit gate failed.")
PY

env \
  OUT_ROOT="${OUT_ROOT}/fullpack" \
  SEED_MODEL="${SEED_MODEL_LINK}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  bash "${FULLPACK_LAUNCHER}"

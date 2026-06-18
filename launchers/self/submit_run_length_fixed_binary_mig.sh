#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/run_length_fixed_binary_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"

RUN_LENGTH_FIXED_BINARY_DEFAULT_CONFIG="${SCRIPT_DIR}/config/run_length_fixed_binary.env"
self_source_config_file "${RUN_LENGTH_FIXED_BINARY_DEFAULT_CONFIG}" "run-length fixed-binary default config"
if [[ -n "${RUN_LENGTH_FIXED_BINARY_CONFIG:-}" ]]; then
  self_source_config_file "${RUN_LENGTH_FIXED_BINARY_CONFIG}" "run-length fixed-binary override config"
fi

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

PAPER_ROOT="${OUT_ROOT}/paper_default"
PAPER_CMD="cd ${ROOT_DIR} && env PYTHON_BIN=${PYTHON_BIN} OUT_ROOT=${PAPER_ROOT} STAGE=${RUN_LENGTH_FIXED_BINARY_PAPER_STAGE} TASKS=${RUN_LENGTH_FIXED_BINARY_PAPER_TASKS} RUN_LENGTH_SEED_MODEL=${RUN_LENGTH_SEED_MODEL} RUN_LENGTH_NUM_EXPAND_ROUNDS=${RUN_LENGTH_FIXED_BINARY_PAPER_EXPAND_ROUNDS} RUN_LENGTH_EXPAND_NUM_BITS=${RUN_LENGTH_FIXED_BINARY_PAPER_EXPAND_NUM_BITS} RUN_LENGTH_EXPAND_TRAIN_PER_BIT=${RUN_LENGTH_FIXED_BINARY_PAPER_EXPAND_TRAIN_PER_BIT} TRAIN_BATCH_SIZE=${RUN_LENGTH_FIXED_BINARY_PAPER_TRAIN_BATCH_SIZE} EVAL_BATCH_SIZE=${RUN_LENGTH_FIXED_BINARY_PAPER_EVAL_BATCH_SIZE} BIT_COMPOSITION_PATH_MODE=${RUN_LENGTH_FIXED_BINARY_PAPER_COMPOSITION_PATH_MODE} RUN_PILOT_GATE=${RUN_LENGTH_FIXED_BINARY_PAPER_RUN_PILOT_GATE} DRY_RUN=0 bash ${ROOT_DIR}/launchers/self/run_figure2_recipe_aggressive.sh"
PAPER_JOB_ID="$(
  self_submit_wrapped_job \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_JOB_NAME}" \
    "${LOG_DIR}/rl-fb-paper-%j.out" \
    "${LOG_DIR}/rl-fb-paper-%j.err" \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_PARTITION}" \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_GRES}" \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_CPUS}" \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_MEM}" \
    "${RUN_LENGTH_FIXED_BINARY_PAPER_TIME}" \
    "" \
    "${PAPER_CMD}"
)"

ALPHA_ROOT="${OUT_ROOT}/alpha10_guarded"
TEMPLATE_OUT="${ALPHA_ROOT}/${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_REL}"
TEMPLATE_CMD="cd ${ROOT_DIR} && PYTHONPATH=. ${PYTHON_BIN} -m self.run_length_self_improvement --output-dir ${TEMPLATE_OUT} --model-name ${ALPHA10_SEED_MODEL} --format-version legacy --target-mode ${RUN_LENGTH_FIXED_BINARY_ALPHA_TARGET_MODE} --compose-arity ${RUN_LENGTH_FIXED_BINARY_ALPHA_COMPOSE_ARITY} --bit-composition-path-mode ${RUN_LENGTH_FIXED_BINARY_ALPHA_COMPOSITION_PATH_MODE} --recipe ${RUN_LENGTH_FIXED_BINARY_ALPHA_RECIPE} --treat-seed-as-round-zero --symbol-alphabet-size ${RUN_LENGTH_FIXED_BINARY_ALPHA_SYMBOL_ALPHABET_SIZE} --initial-min-bits ${RUN_LENGTH_FIXED_BINARY_ALPHA_INITIAL_MIN_BITS} --initial-max-bits ${RUN_LENGTH_FIXED_BINARY_ALPHA_INITIAL_MAX_BITS} --initial-train-per-bit ${RUN_LENGTH_FIXED_BINARY_ALPHA_INITIAL_TRAIN_PER_BIT} --initial-eval-per-bit ${RUN_LENGTH_FIXED_BINARY_ALPHA_INITIAL_EVAL_PER_BIT} --frontier-min-bits ${RUN_LENGTH_FIXED_BINARY_ALPHA_FRONTIER_MIN_BITS} --num-expand-rounds ${RUN_LENGTH_FIXED_BINARY_ALPHA_NUM_EXPAND_ROUNDS} --expand-num-bits ${RUN_LENGTH_FIXED_BINARY_ALPHA_EXPAND_NUM_BITS} --expand-train-per-bit ${RUN_LENGTH_FIXED_BINARY_ALPHA_EXPAND_TRAIN_PER_BIT} --eval-per-bit ${RUN_LENGTH_FIXED_BINARY_ALPHA_EVAL_PER_BIT} --composed-eval-per-bit ${RUN_LENGTH_FIXED_BINARY_ALPHA_COMPOSED_EVAL_PER_BIT} --pseudo-label-mode ${RUN_LENGTH_FIXED_BINARY_ALPHA_PSEUDO_LABEL_MODE} --guarded-compose-rule ${RUN_LENGTH_FIXED_BINARY_ALPHA_GUARDED_COMPOSE_RULE} --bucket-train-batches-by-bits --bf16 --per-device-train-batch-size ${RUN_LENGTH_FIXED_BINARY_ALPHA_TRAIN_BATCH_SIZE} --per-device-eval-batch-size ${RUN_LENGTH_FIXED_BINARY_ALPHA_EVAL_BATCH_SIZE} --seed ${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_SEED} --save-model-policy all_rounds --self-improve-warmup-steps ${RUN_LENGTH_FIXED_BINARY_ALPHA_WARMUP_STEPS} --resume --stop-after-round 0"
TEMPLATE_JOB_ID="$(
  self_submit_wrapped_job \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_JOB_NAME}" \
    "${LOG_DIR}/rl-fb-a10-template-%j.out" \
    "${LOG_DIR}/rl-fb-a10-template-%j.err" \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_PARTITION}" \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_GRES}" \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_CPUS}" \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_MEM}" \
    "${RUN_LENGTH_FIXED_BINARY_ALPHA_TEMPLATE_TIME}" \
    "" \
    "${TEMPLATE_CMD}"
)"

BEAM_ROOT="${ALPHA_ROOT}/beam"
BEAM_CMD="cd ${ROOT_DIR} && PYTHONPATH=. ${PYTHON_BIN} -m self.experiments.run_length_alpha10_seed_beam --template-run ${TEMPLATE_OUT} --seed-model ${ALPHA10_SEED_MODEL} --out-root ${BEAM_ROOT} --max-round ${RUN_LENGTH_FIXED_BINARY_BEAM_MAX_ROUND} --round-warmup-steps ${RUN_LENGTH_FIXED_BINARY_BEAM_WARMUP_STEPS} --train-batch-size ${RUN_LENGTH_FIXED_BINARY_BEAM_TRAIN_BATCH_SIZE} --eval-batch-size ${RUN_LENGTH_FIXED_BINARY_BEAM_EVAL_BATCH_SIZE} --expand-train-per-bit ${RUN_LENGTH_FIXED_BINARY_BEAM_EXPAND_TRAIN_PER_BIT} --bit-composition-path-mode ${RUN_LENGTH_FIXED_BINARY_BEAM_COMPOSITION_PATH_MODE} --baseline ${RUN_LENGTH_FIXED_BINARY_ALPHA_BASELINE}"
BEAM_JOB_ID="$(
  self_submit_wrapped_job \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_JOB_NAME}" \
    "${LOG_DIR}/rl-fb-a10-beam-%j.out" \
    "${LOG_DIR}/rl-fb-a10-beam-%j.err" \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_PARTITION}" \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_GRES}" \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_CPUS}" \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_MEM}" \
    "${RUN_LENGTH_FIXED_BINARY_BEAM_TIME}" \
    "afterok:${TEMPLATE_JOB_ID}" \
    "${BEAM_CMD}"
)"

MANIFEST="${OUT_ROOT}/run_manifest.json"
"${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${OUT_ROOT}" "${PAPER_ROOT}" "${PAPER_JOB_ID}" "${TEMPLATE_OUT}" "${TEMPLATE_JOB_ID}" "${BEAM_ROOT}" "${BEAM_JOB_ID}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
payload = {
    "out_root": sys.argv[2],
    "paper_default": {
        "output_root": sys.argv[3],
        "job_id": sys.argv[4],
        "results_path": str(Path(sys.argv[3]) / "run_length/pilot/compose/self_improvement_results.json"),
    },
    "alpha10_guarded": {
        "template_output": sys.argv[5],
        "template_job_id": sys.argv[6],
        "beam_root": sys.argv[7],
        "beam_job_id": sys.argv[8],
        "beam_summary": str(Path(sys.argv[7]) / "beam_summary.json"),
    },
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload, indent=2), flush=True)
PY

echo "[INFO] Submitted fixed-binary run-length jobs."
echo "[INFO] Manifest: ${MANIFEST}"

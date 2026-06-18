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

RUN_LENGTH_SEED_MODEL="${RUN_LENGTH_SEED_MODEL:-${ROOT_DIR}/artifacts/models/run_length_recipe_seed_best}"
ALPHA10_SEED_MODEL="${ALPHA10_SEED_MODEL:-${ROOT_DIR}/artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model}"
CONTROLLER_PARTITION="${CONTROLLER_PARTITION:-cpu}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

submit_job() {
  local job_name="$1"
  local stdout_log="$2"
  local stderr_log="$3"
  local partition="$4"
  local gres="$5"
  local cpus="$6"
  local mem="$7"
  local time_limit="$8"
  local dependency="${9:-}"
  shift 9 || true
  local wrap_cmd="$*"
  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --output "${stdout_log}"
    --error "${stderr_log}"
  )
  self_add_sbatch_explicit_resources sbatch_cmd "${partition}" "${gres}" "${cpus}" "${mem}" "${time_limit}"
  if [[ -n "${dependency}" ]]; then
    sbatch_cmd+=(--dependency "${dependency}")
  fi
  sbatch_cmd+=(--wrap "${wrap_cmd}")
  self_print_command "${sbatch_cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dryrun-${job_name}"
  else
    "${sbatch_cmd[@]}" | cut -d';' -f1
  fi
}

PAPER_ROOT="${OUT_ROOT}/paper_default"
PAPER_CMD="cd ${ROOT_DIR} && env PYTHON_BIN=${PYTHON_BIN} OUT_ROOT=${PAPER_ROOT} STAGE=pilot TASKS=run_length RUN_LENGTH_SEED_MODEL=${RUN_LENGTH_SEED_MODEL} RUN_LENGTH_NUM_EXPAND_ROUNDS=8 RUN_LENGTH_EXPAND_NUM_BITS=4 RUN_LENGTH_EXPAND_TRAIN_PER_BIT=1200 TRAIN_BATCH_SIZE=128 EVAL_BATCH_SIZE=128 BIT_COMPOSITION_PATH_MODE=fixed_binary RUN_PILOT_GATE=0 DRY_RUN=0 bash ${ROOT_DIR}/launchers/self/run_figure2_recipe_aggressive.sh"
PAPER_JOB_ID="$(
  submit_job \
    "rl-fb-paper" \
    "${LOG_DIR}/rl-fb-paper-%j.out" \
    "${LOG_DIR}/rl-fb-paper-%j.err" \
    "mig" \
    "gpu:1g.10gb:1" \
    "1" \
    "64G" \
    "48:00:00" \
    "" \
    "${PAPER_CMD}"
)"

ALPHA_ROOT="${OUT_ROOT}/alpha10_guarded"
TEMPLATE_OUT="${ALPHA_ROOT}/template/run_length/pilot/guarded_compose"
TEMPLATE_CMD="cd ${ROOT_DIR} && PYTHONPATH=. ${PYTHON_BIN} -m self.run_length_self_improvement --output-dir ${TEMPLATE_OUT} --model-name ${ALPHA10_SEED_MODEL} --format-version legacy --target-mode symbol_run_pair --compose-arity exact2 --bit-composition-path-mode fixed_binary --recipe algorithmic_self_improve_v1 --treat-seed-as-round-zero --symbol-alphabet-size 10 --initial-min-bits 6 --initial-max-bits 10 --initial-train-per-bit 50000 --initial-eval-per-bit 100 --frontier-min-bits 12 --num-expand-rounds 7 --expand-num-bits 9 --expand-train-per-bit 2000 --eval-per-bit 100 --composed-eval-per-bit 100 --pseudo-label-mode compose --guarded-compose-rule run_length_no_boundary_continue --bucket-train-batches-by-bits --bf16 --per-device-train-batch-size 256 --per-device-eval-batch-size 256 --seed 42 --save-model-policy all_rounds --self-improve-warmup-steps 500 --resume --stop-after-round 0"
TEMPLATE_JOB_ID="$(
  submit_job \
    "rl-fb-a10-template" \
    "${LOG_DIR}/rl-fb-a10-template-%j.out" \
    "${LOG_DIR}/rl-fb-a10-template-%j.err" \
    "mig" \
    "gpu:1g.10gb:1" \
    "1" \
    "64G" \
    "12:00:00" \
    "" \
    "${TEMPLATE_CMD}"
)"

BEAM_ROOT="${ALPHA_ROOT}/beam"
BEAM_CMD="cd ${ROOT_DIR} && PYTHONPATH=. ${PYTHON_BIN} ${ROOT_DIR}/launchers/self/run_run_length_alpha10_seed_beam_mig.py --template-run ${TEMPLATE_OUT} --seed-model ${ALPHA10_SEED_MODEL} --out-root ${BEAM_ROOT} --max-round 7 --round-warmup-steps 500 --train-batch-size 256 --eval-batch-size 256 --expand-train-per-bit 2000 --bit-composition-path-mode fixed_binary"
BEAM_JOB_ID="$(
  submit_job \
    "rl-fb-a10-beam" \
    "${LOG_DIR}/rl-fb-a10-beam-%j.out" \
    "${LOG_DIR}/rl-fb-a10-beam-%j.err" \
    "${CONTROLLER_PARTITION}" \
    "" \
    "1" \
    "8G" \
    "36:00:00" \
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

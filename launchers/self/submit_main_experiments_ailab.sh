#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root
adaptive_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/main_experiments_ailab_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"
adaptive_set_sbatch_defaults "ailab" "gpu:h200:1" "4" "96G" "72:00:00"

RUN_LENGTH_SCRIPT="${RUN_LENGTH_SCRIPT:-${ROOT_DIR}/launchers/self/run_figure2_recipe_aggressive.sh}"
ADDITION_SCRIPT="${ADDITION_SCRIPT:-${ROOT_DIR}/launchers/self/run_addition_recipe_fullpack.sh}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

submit_job() {
  local job_name="$1"
  local stdout_log="$2"
  local stderr_log="$3"
  shift 3
  local wrap_cmd="$*"
  local -a sbatch_cmd=(
    sbatch
    --parsable
    --job-name "${job_name}"
    --output "${stdout_log}"
    --error "${stderr_log}"
  )
  adaptive_add_sbatch_resources sbatch_cmd
  sbatch_cmd+=(--wrap "${wrap_cmd}")
  adaptive_print_command "${sbatch_cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dryrun-${job_name}"
  else
    "${sbatch_cmd[@]}" | cut -d';' -f1
  fi
}

RUN_LENGTH_OUT="${OUT_ROOT}/run_length_run_state"
RUN_LENGTH_CMD="cd ${ROOT_DIR} && env PYTHON_BIN=${PYTHON_BIN} OUT_ROOT=${RUN_LENGTH_OUT} STAGE=all TASKS=run_length BASELINES='short_only direct compose compose_corrupt' TARGET_MODE=run_state RUN_LENGTH_TARGET_MODE=run_state RUN_LENGTH_BIT_COMPOSITION_PATH_MODE=fixed_binary BIT_COMPOSITION_PATH_MODE=fixed_binary RUN_PILOT_GATE=0 RUN_FULLPACK_ONLY_IF_HEALTHY=0 TRAIN_BATCH_SIZE=512 EVAL_BATCH_SIZE=512 INITIAL_TRAIN_PER_BIT=50000 INITIAL_EVAL_PER_BIT=100 RUN_LENGTH_NUM_EXPAND_ROUNDS=8 RUN_LENGTH_EXPAND_NUM_BITS=4 RUN_LENGTH_EXPAND_TRAIN_PER_BIT=2000 COMPOSED_EVAL_PER_BIT=100 DRY_RUN=0 bash ${RUN_LENGTH_SCRIPT}"
RUN_LENGTH_JOB_ID="$(
  submit_job \
    "main-rl-runstate" \
    "${LOG_DIR}/main-rl-runstate-%j.out" \
    "${LOG_DIR}/main-rl-runstate-%j.err" \
    "${RUN_LENGTH_CMD}"
)"

ADDITION_OUT="${OUT_ROOT}/addition_recipe_fullpack"
ADDITION_CMD="cd ${ROOT_DIR} && env PYTHON_BIN=${PYTHON_BIN} OUT_ROOT=${ADDITION_OUT} SEED_MODEL=${ROOT_DIR}/artifacts/models/addition_recipe_seed_best TRAIN_BATCH_SIZE=1024 EVAL_BATCH_SIZE=1024 SEED_REPLAY_TRAIN_PER_DIGIT=5000 EXPAND_TRAIN_PER_DIGIT=5000 NUM_EXPAND_ROUNDS=8 EXPAND_NUM_DIGITS=3 ADDITION_COMPOSITION_PATH_MODE=fixed_binary DRY_RUN=0 bash ${ADDITION_SCRIPT}"
ADDITION_JOB_ID="$(
  submit_job \
    "main-add-fullpack" \
    "${LOG_DIR}/main-add-fullpack-%j.out" \
    "${LOG_DIR}/main-add-fullpack-%j.err" \
    "${ADDITION_CMD}"
)"

MANIFEST="${OUT_ROOT}/submission_manifest.json"
"${PYTHON_BIN}" - <<'PY' "${MANIFEST}" "${OUT_ROOT}" "${RUN_LENGTH_OUT}" "${RUN_LENGTH_JOB_ID}" "${ADDITION_OUT}" "${ADDITION_JOB_ID}" "${SBATCH_PARTITION}" "${SBATCH_GRES}" "${SBATCH_TIME}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
payload = {
    "out_root": sys.argv[2],
    "slurm": {
        "partition": sys.argv[7],
        "gres": sys.argv[8],
        "time": sys.argv[9],
    },
    "jobs": {
        "run_length_run_state": {
            "job_id": sys.argv[4],
            "output_root": sys.argv[3],
            "target_mode": "run_state",
            "composition_path": "fixed_binary",
            "status": "submitted",
        },
        "addition_recipe_fullpack": {
            "job_id": sys.argv[6],
            "output_root": sys.argv[5],
            "composition_path": "fixed_binary",
            "status": "submitted",
        },
    },
}
manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2), flush=True)
PY

echo "[INFO] Submitted main experiment jobs."
echo "[INFO] Manifest: ${MANIFEST}"

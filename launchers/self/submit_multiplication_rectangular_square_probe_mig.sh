#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

SEED_LAUNCHER="${ROOT_DIR}/launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch"
DIAG_LAUNCHER="${ROOT_DIR}/launchers/self/run_multiplication_rectangular_square_compose_diagnostic_mig.sbatch"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_square_probe_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"

SEED_OUT="${OUT_ROOT}/seed"
DIAG_OUT="${OUT_ROOT}/compose_diag"

SEED_TRAIN_PER_PARTITION="${SEED_TRAIN_PER_PARTITION:-25000}"
SEED_HELDOUT_PER_PARTITION="${SEED_HELDOUT_PER_PARTITION:-200}"
SEED_MAX_STEPS="${SEED_MAX_STEPS:-6000}"
SEED_SKIP_TRAIN_EVAL="${SEED_SKIP_TRAIN_EVAL:-1}"
DIAG_TRAIN_PER_PARTITION="${DIAG_TRAIN_PER_PARTITION:-2000}"
DIAG_HELDOUT_PER_PARTITION="${DIAG_HELDOUT_PER_PARTITION:-100}"
DIAG_MAX_STEPS="${DIAG_MAX_STEPS:-3000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED="${SEED:-0}"
SAVE_MODEL="${SAVE_MODEL:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

seed_log_stem="${LOG_DIR}/mult-rect-square-seed-%j"
seed_job_id="$(
  self_submit_sbatch_script \
    "dryrun_seed" \
    "mult-rect-square-seed" \
    "${seed_log_stem}.out" \
    "${seed_log_stem}.err" \
    "ALL,OUT_ROOT=${SEED_OUT},TRAIN_PER_PARTITION=${SEED_TRAIN_PER_PARTITION},HELDOUT_PER_PARTITION=${SEED_HELDOUT_PER_PARTITION},MAX_STEPS=${SEED_MAX_STEPS},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SEED=${SEED},SAVE_MODEL=${SAVE_MODEL},SKIP_TRAIN_EVAL=${SEED_SKIP_TRAIN_EVAL},DRY_RUN=${DRY_RUN}" \
    "${SEED_LAUNCHER}"
)"
echo "[INFO] seed_job_id=${seed_job_id} output=${SEED_OUT} log=${seed_log_stem}.out"

diag_log_stem="${LOG_DIR}/mult-rect-square-diag-%j"
diag_job_id="$(
  self_submit_sbatch_script \
    "dryrun_diag" \
    "mult-rect-square-diag" \
    "${diag_log_stem}.out" \
    "${diag_log_stem}.err" \
    "ALL,OUT_ROOT=${DIAG_OUT},SEED_MODEL=${SEED_OUT}/model,TRAIN_PER_PARTITION=${DIAG_TRAIN_PER_PARTITION},HELDOUT_PER_PARTITION=${DIAG_HELDOUT_PER_PARTITION},MAX_STEPS=${DIAG_MAX_STEPS},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SEED=${SEED},SAVE_MODEL=${SAVE_MODEL},DRY_RUN=${DRY_RUN}" \
    --dependency "afterok:${seed_job_id}" \
    "${DIAG_LAUNCHER}"
)"
echo "[INFO] diagnostic_job_id=${diag_job_id} output=${DIAG_OUT} log=${diag_log_stem}.out"

echo "[INFO] Seed summary: ${SEED_OUT}/summary.json"
echo "[INFO] Diagnostic summary: ${DIAG_OUT}/summary.json"
echo "[INFO] Status: $([[ "${DRY_RUN}" == "1" ]] && echo dry_run || echo submitted)"

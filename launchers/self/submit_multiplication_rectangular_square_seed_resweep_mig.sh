#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

LAUNCHER="${ROOT_DIR}/launchers/self/run_multiplication_rectangular_square_seed_mig.sbatch"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-10000}"
HELDOUT_PER_PARTITION="${HELDOUT_PER_PARTITION:-200}"
SKIP_TRAIN_EVAL="${SKIP_TRAIN_EVAL:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_square_seed_resweep_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"

TRAIN_COUNTS=(${TRAIN_COUNTS:-50000 100000})
LRS=(${LRS:-2e-5 5e-5 1e-4})

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

lr_tag() {
  echo "${1}" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

for train_count in "${TRAIN_COUNTS[@]}"; do
  for lr in "${LRS[@]}"; do
    tag="$(lr_tag "${lr}")"
    out_dir="${OUT_ROOT}/train_${train_count}/lr_${tag}"
    job_name="mult-rect-square-seed-train-${train_count}-lr-${tag}"
    log_stem="${LOG_DIR}/${job_name}-%j"
    job_id="$(
      self_submit_sbatch_script \
        "dryrun" \
        "${job_name}" \
        "${log_stem}.out" \
        "${log_stem}.err" \
        "ALL,OUT_ROOT=${out_dir},LR=${lr},TRAIN_PER_PARTITION=${train_count},MAX_STEPS=${MAX_STEPS},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SEED=${SEED},SAVE_MODEL=1,HELDOUT_PER_PARTITION=${HELDOUT_PER_PARTITION},SKIP_TRAIN_EVAL=${SKIP_TRAIN_EVAL},DRY_RUN=${DRY_RUN}" \
        "${LAUNCHER}"
    )"
    if self_parse_bool "${DRY_RUN}"; then
      echo "[INFO] square_seed_job=train_${train_count}_lr_${tag} job_id=dryrun output=${out_dir} summary=${out_dir}/summary.json"
    else
      echo "[INFO] square_seed_job=train_${train_count}_lr_${tag} job_id=${job_id} output=${out_dir} summary=${out_dir}/summary.json"
    fi
  done
done

echo "[INFO] Status: $(self_parse_bool "${DRY_RUN}" && echo dry_run || echo submitted)"

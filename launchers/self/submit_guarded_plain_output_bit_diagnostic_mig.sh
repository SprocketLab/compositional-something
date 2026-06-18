#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

JOB_SCRIPT="${ROOT_DIR}/launchers/self/run_guarded_plain_output_bit_diagnostic_mig.sbatch"
LOG_DIR="${ROOT_DIR}/artifacts/logs"
mkdir -p "${LOG_DIR}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/guarded_plain_output_bit_diagnostic_${TS}}"
TASKS="${TASKS:-run_length}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED="${SEED:-42}"
SYMBOL_ALPHABET_SIZE="${SYMBOL_ALPHABET_SIZE:-2}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a TASK_LIST <<< "${TASKS}"

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Job script: ${JOB_SCRIPT}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Tasks: ${TASKS}"
echo "[INFO] Train batch size: ${TRAIN_BATCH_SIZE}"
echo "[INFO] Eval batch size: ${EVAL_BATCH_SIZE}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Symbol alphabet size: ${SYMBOL_ALPHABET_SIZE}"
echo "[INFO] Dry run: ${DRY_RUN}"

for task in "${TASK_LIST[@]}"; do
  cmd=(
    sbatch
    --export=ALL,TASK="${task}",OUT_ROOT="${OUT_ROOT}",TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}",EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}",SEED="${SEED}",SYMBOL_ALPHABET_SIZE="${SYMBOL_ALPHABET_SIZE}",DRY_RUN="${DRY_RUN}"
    "${JOB_SCRIPT}"
  )
  self_print_command_stdout "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[INFO] DRY_RUN=1; sbatch not executed for ${task}."
    continue
  fi
  submit_output="$("${cmd[@]}")"
  echo "${submit_output}"
  job_id="$(awk '{print $4}' <<< "${submit_output}")"
  echo "[INFO] Submitted task=${task} job_id=${job_id} output_dir=${OUT_ROOT}/${task}"
done

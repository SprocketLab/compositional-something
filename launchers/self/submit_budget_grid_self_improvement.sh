#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/self/run_task_self_improvement.sbatch}"
TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/artifacts/runs/self_improvement_budget_grid/${TS}}"
SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
SEED="${SEED:-0}"
MODES="${MODES:-none compose}"
TASKS="${TASKS:-addition run_length multiplication}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_LOCAL_MODEL="${ROOT_DIR}/artifacts/models/SmolLM2-360M"
DEFAULT_HUB_MODEL="HuggingFaceTB/SmolLM2-360M"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_LOCAL_MODEL}}"
else
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_HUB_MODEL}}"
fi

mkdir -p "${BASE_OUT}"

read -r -a MODE_LIST <<< "${MODES}"
read -r -a TASK_LIST <<< "${TASKS}"
BUDGET_LIST=(small medium large)

module_for_task() {
  case "$1" in
    addition) echo "self.self_improvement" ;;
    run_length) echo "self.run_length_self_improvement" ;;
    multiplication) echo "self.multiplication_self_improvement" ;;
    *)
      echo "Unknown task: $1" >&2
      return 1
      ;;
  esac
}

time_for_budget() {
  case "$1" in
    small) echo "${SBATCH_TIME_SMALL:-04:00:00}" ;;
    medium) echo "${SBATCH_TIME_MEDIUM:-08:00:00}" ;;
    large) echo "${SBATCH_TIME_LARGE:-12:00:00}" ;;
    *)
      echo "Unknown budget: $1" >&2
      return 1
      ;;
  esac
}

extra_args_for() {
  local task="$1"
  local budget="$2"
  local mode="$3"

  case "${task}" in
    addition)
      case "${budget}" in
        small)
          echo "--initial-min-digits 3 --initial-max-digits 5 --initial-train-per-digit 500 --initial-eval-per-digit 50 --num-expand-rounds 2 --expand-num-digits 2 --expand-train-per-digit 250 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 80 --pseudo-label-mode ${mode}"
          ;;
        medium)
          echo "--initial-min-digits 3 --initial-max-digits 7 --initial-train-per-digit 2000 --initial-eval-per-digit 50 --num-expand-rounds 4 --expand-num-digits 2 --expand-train-per-digit 1200 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 240 --pseudo-label-mode ${mode}"
          ;;
        large)
          echo "--initial-min-digits 3 --initial-max-digits 7 --initial-train-per-digit 4000 --initial-eval-per-digit 50 --num-expand-rounds 8 --expand-num-digits 2 --expand-train-per-digit 2400 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 480 --pseudo-label-mode ${mode}"
          ;;
        *)
          echo "Unknown budget: ${budget}" >&2
          return 1
          ;;
      esac
      ;;
    run_length)
      case "${budget}" in
        small)
          echo "--initial-min-bits 4 --initial-max-bits 8 --initial-train-per-bit 500 --initial-eval-per-bit 50 --num-expand-rounds 2 --expand-num-bits 4 --expand-train-per-bit 250 --eval-per-bit 100 --composed-eval-per-bit 50 --decode-max-new-tokens 16 --max-steps 80 --pseudo-label-mode ${mode}"
          ;;
        medium)
          echo "--initial-min-bits 4 --initial-max-bits 8 --initial-train-per-bit 2000 --initial-eval-per-bit 50 --num-expand-rounds 4 --expand-num-bits 4 --expand-train-per-bit 1200 --eval-per-bit 100 --composed-eval-per-bit 50 --decode-max-new-tokens 16 --max-steps 240 --pseudo-label-mode ${mode}"
          ;;
        large)
          echo "--initial-min-bits 4 --initial-max-bits 8 --initial-train-per-bit 4000 --initial-eval-per-bit 50 --num-expand-rounds 4 --expand-num-bits 4 --expand-train-per-bit 2400 --eval-per-bit 100 --composed-eval-per-bit 50 --decode-max-new-tokens 16 --max-steps 480 --pseudo-label-mode ${mode}"
          ;;
        *)
          echo "Unknown budget: ${budget}" >&2
          return 1
          ;;
      esac
      ;;
    multiplication)
      case "${budget}" in
        small)
          echo "--block-size 2 --initial-min-digits 2 --initial-max-digits 2 --initial-train-per-digit 2000 --initial-eval-per-digit 50 --num-expand-rounds 2 --expand-num-digits 2 --expand-train-per-digit 600 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 120 --pseudo-label-mode ${mode}"
          ;;
        medium)
          echo "--block-size 2 --initial-min-digits 2 --initial-max-digits 2 --initial-train-per-digit 4000 --initial-eval-per-digit 50 --num-expand-rounds 2 --expand-num-digits 2 --expand-train-per-digit 1200 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 240 --pseudo-label-mode ${mode}"
          ;;
        large)
          echo "--block-size 2 --initial-min-digits 2 --initial-max-digits 2 --initial-train-per-digit 8000 --initial-eval-per-digit 50 --num-expand-rounds 2 --expand-num-digits 2 --expand-train-per-digit 2400 --eval-per-digit 100 --composed-eval-per-digit 50 --decode-max-new-tokens 16 --max-steps 480 --pseudo-label-mode ${mode}"
          ;;
        *)
          echo "Unknown budget: ${budget}" >&2
          return 1
          ;;
      esac
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      return 1
      ;;
  esac
}

SUBMISSION_TSV="${BASE_OUT}/submission_jobs.tsv"
{
  echo -e "job_id\ttask\tmode\tbudget\tout_dir\tpartition\ttime_limit\tmodel_name"
} > "${SUBMISSION_TSV}"

for task in "${TASK_LIST[@]}"; do
  module="$(module_for_task "${task}")"
  for mode in "${MODE_LIST[@]}"; do
    for budget in "${BUDGET_LIST[@]}"; do
      out_dir="${BASE_OUT}/${task}/${mode}/${budget}/seed_${SEED}"
      extra_args="$(extra_args_for "${task}" "${budget}" "${mode}")"
      time_limit="$(time_for_budget "${budget}")"
      job_name="sibg-${task}-${mode}-${budget}"

      if self_parse_bool "${DRY_RUN}"; then
        job_id="DRYRUN-${task}-${mode}-${budget}"
        echo "[DRY RUN] ${job_name}"
      else
        job_id="$(
          TASK_MODULE="${module}" \
          TASK_TAG="${task}" \
          BUDGET_TAG="${budget}" \
          MODE_TAG="${mode}" \
          OUT_ROOT="${out_dir}" \
          MODEL_NAME="${MODEL_NAME}" \
          SEED="${SEED}" \
          EXTRA_ARGS="${extra_args}" \
          sbatch \
            --partition="${SBATCH_PARTITION}" \
            --time="${time_limit}" \
            --job-name="${job_name}" \
            --export=ALL \
            "${JOB_SCRIPT}" | awk '{print $4}'
        )"
        echo "[INFO] Submitted ${job_name} -> ${job_id}"
      fi

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${job_id}" "${task}" "${mode}" "${budget}" "${out_dir}" "${SBATCH_PARTITION}" "${time_limit}" "${MODEL_NAME}" \
        >> "${SUBMISSION_TSV}"
    done
  done
done

cat > "${BASE_OUT}/submission_info.txt" <<EOF
base_out=${BASE_OUT}
job_script=${JOB_SCRIPT}
partition=${SBATCH_PARTITION}
seed=${SEED}
tasks=${TASKS}
modes=${MODES}
budgets=${BUDGET_LIST[*]}
model_name=${MODEL_NAME}
dry_run=${DRY_RUN}
submitted_at=$(date --iso-8601=seconds)
submission_tsv=${SUBMISSION_TSV}
EOF

echo "[INFO] Wrote submission metadata to ${BASE_OUT}/submission_info.txt"
echo "[INFO] Wrote job table to ${SUBMISSION_TSV}"

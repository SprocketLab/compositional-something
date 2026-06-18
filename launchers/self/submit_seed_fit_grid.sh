#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/self/run_seed_fit_experiment.sbatch}"
TS="$(date +%Y%m%d_%H%M%S)"
BASE_OUT="${BASE_OUT:-${ROOT_DIR}/artifacts/runs/seed_fit_grid/${TS}}"
SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
SEED="${SEED:-0}"
TASKS="${TASKS:-run_length multiplication}"
DRY_RUN="${DRY_RUN:-0}"
FORMAT_VERSION="${FORMAT_VERSION:-legacy}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"

DEFAULT_LOCAL_MODEL="${ROOT_DIR}/artifacts/models/SmolLM2-360M"
DEFAULT_HUB_MODEL="HuggingFaceTB/SmolLM2-360M"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_LOCAL_MODEL}}"
else
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_HUB_MODEL}}"
fi

mkdir -p "${BASE_OUT}"

read -r -a TASK_LIST <<< "${TASKS}"

train_sizes_for_task() {
  case "$1" in
    run_length) echo "${RUN_LENGTH_TRAIN_SIZES:-500 1000 2000 4000}" ;;
    multiplication) echo "${MULTIPLICATION_TRAIN_SIZES:-2000 4000 8000 12000}" ;;
    *)
      echo "Unknown task: $1" >&2
      return 1
      ;;
  esac
}

steps_for_task() {
  case "$1" in
    run_length) echo "${RUN_LENGTH_STEP_BUDGETS:-960 3840 7680}" ;;
    multiplication) echo "${MULTIPLICATION_STEP_BUDGETS:-960 3840 7680}" ;;
    *)
      echo "Unknown task: $1" >&2
      return 1
      ;;
  esac
}

time_for_steps() {
  case "$1" in
    960) echo "${SBATCH_TIME_960:-08:00:00}" ;;
    3840) echo "${SBATCH_TIME_3840:-20:00:00}" ;;
    7680) echo "${SBATCH_TIME_7680:-36:00:00}" ;;
    15360) echo "${SBATCH_TIME_15360:-48:00:00}" ;;
    30720) echo "${SBATCH_TIME_30720:-72:00:00}" ;;
    *)
      echo "Unknown step budget: $1" >&2
      return 1
      ;;
  esac
}

extra_args_for() {
  local task="$1"
  local train_per_size="$2"
  local max_steps="$3"

  case "${task}" in
    run_length)
      echo "--format-version ${FORMAT_VERSION} --initial-min-size 4 --initial-max-size 8 --initial-train-per-size ${train_per_size} --initial-eval-per-size 100 --expand-num-size 4 --decode-max-new-tokens 16 --max-steps ${max_steps}"
      ;;
    multiplication)
      echo "--format-version ${FORMAT_VERSION} --initial-min-size 2 --initial-max-size 3 --initial-train-per-size ${train_per_size} --initial-eval-per-size 100 --expand-num-size 2 --block-size 2 --decode-max-new-tokens 16 --max-steps ${max_steps}"
      ;;
    *)
      echo "Unknown task: ${task}" >&2
      return 1
      ;;
  esac
}

SUBMISSION_TSV="${BASE_OUT}/submission_jobs.tsv"
{
  echo -e "job_id\ttask\ttrain_per_size\tmax_steps\tout_dir\tpartition\ttime_limit\tmodel_name"
} > "${SUBMISSION_TSV}"

for task in "${TASK_LIST[@]}"; do
  read -r -a TRAIN_SIZE_LIST <<< "$(train_sizes_for_task "${task}")"
  read -r -a STEP_LIST <<< "$(steps_for_task "${task}")"
  for train_per_size in "${TRAIN_SIZE_LIST[@]}"; do
    for max_steps in "${STEP_LIST[@]}"; do
      data_tag="train_${train_per_size}"
      step_tag="steps_${max_steps}"
      out_dir="${BASE_OUT}/${task}/${data_tag}/${step_tag}/seed_${SEED}"
      extra_args="$(extra_args_for "${task}" "${train_per_size}" "${max_steps}")"
      time_limit="$(time_for_steps "${max_steps}")"
      job_name="sfit-${task}-n${train_per_size}-s${max_steps}"

      if self_parse_bool "${DRY_RUN}"; then
        job_id="DRYRUN-${task}-${train_per_size}-${max_steps}"
        echo "[DRY RUN] ${job_name}"
      else
        job_id="$(
          TASK_NAME="${task}" \
          TASK_TAG="${task}" \
          DATA_TAG="${data_tag}" \
          STEP_TAG="${step_tag}" \
          OUT_ROOT="${out_dir}" \
          MODEL_NAME="${MODEL_NAME}" \
          SEED="${SEED}" \
          TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
          EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
          GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
          NUM_EPOCHS="${NUM_EPOCHS}" \
          LOGGING_STEPS="${LOGGING_STEPS}" \
          MAX_STEPS="${max_steps}" \
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
        "${job_id}" "${task}" "${train_per_size}" "${max_steps}" "${out_dir}" "${SBATCH_PARTITION}" "${time_limit}" "${MODEL_NAME}" \
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
model_name=${MODEL_NAME}
format_version=${FORMAT_VERSION}
train_batch_size=${TRAIN_BATCH_SIZE}
eval_batch_size=${EVAL_BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
num_epochs=${NUM_EPOCHS}
logging_steps=${LOGGING_STEPS}
dry_run=${DRY_RUN}
submitted_at=$(date --iso-8601=seconds)
submission_tsv=${SUBMISSION_TSV}
EOF

echo "[INFO] Wrote submission metadata to ${BASE_OUT}/submission_info.txt"
echo "[INFO] Wrote job table to ${SUBMISSION_TSV}"

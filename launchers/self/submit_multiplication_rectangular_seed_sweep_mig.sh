#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root
self_resolve_python

LAUNCHER="${ROOT_DIR}/launchers/self/run_multiplication_rectangular_seed_mig.sbatch"
MODEL_LINK="${ROOT_DIR}/artifacts/models/multiplication_rectangular_seed_best"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
MAX_STEPS="${MAX_STEPS:-10000}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_seed_search_${TS}}"

STAGE0_TRAIN_PER_PARTITION=10
STAGE0_MAX_STEPS=1000
STAGE1_TRAIN_COUNTS=(25000 50000)
STAGE1_LRS=(2e-5 5e-5 1e-4)
STAGE3_TRAIN_PER_PARTITION=100000

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
mkdir -p "$(dirname "${MODEL_LINK}")"

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Log dir" "${LOG_DIR}" \
  "Launcher" "${LAUNCHER}" \
  "Stable model link" "${MODEL_LINK}" \
  "Dry run" "${DRY_RUN}"

write_status() {
  local status="$1"
  printf '%s\n' "${status}" > "${OUT_ROOT}/status.txt"
  echo "[INFO] Status: ${status}"
}

lr_tag() {
  echo "${1}" | sed 's/-/m/g; s/\./p/g; s/+//g'
}

submit_job() {
  local out_dir="$1"
  local lr="$2"
  local train_per_partition="$3"
  local max_steps="$4"
  local heldout_per_partition="$5"
  local save_model="$6"
  local out_slug="${out_dir#${OUT_ROOT}/}"
  out_slug="$(echo "${out_slug}" | sed 's#[^A-Za-z0-9_-]#-#g; s#_#-#g')"
  local job_name="mult-rect-seed-${out_slug}"
  local log_stem="${LOG_DIR}/${job_name}-%j"

  self_submit_sbatch_script \
    "dryrun" \
    "${job_name}" \
    "${log_stem}.out" \
    "${log_stem}.err" \
    "ALL,OUT_ROOT=${out_dir},LR=${lr},TRAIN_PER_PARTITION=${train_per_partition},MAX_STEPS=${max_steps},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SEED=${SEED},SAVE_MODEL=${save_model},HELDOUT_PER_PARTITION=${heldout_per_partition}" \
    "${LAUNCHER}"
}

job_state() {
  local job_id="$1"
  local state
  state="$(sacct -X -n -j "${job_id}" -o State 2>/dev/null | awk 'NF {print $1; exit}')"
  if [[ -n "${state}" ]]; then
    echo "${state}"
    return 0
  fi
  if squeue -h -j "${job_id}" 2>/dev/null | grep -q .; then
    echo "PENDING"
    return 0
  fi
  echo "UNKNOWN"
}

wait_for_jobs() {
  local -a job_ids=("$@")
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  local remaining=${#job_ids[@]}
  while (( remaining > 0 )); do
    remaining=0
    for job_id in "${job_ids[@]}"; do
      local state
      state="$(job_state "${job_id}")"
      case "${state}" in
        COMPLETED)
          ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|PREEMPTED|NODE_FAIL|BOOT_FAIL|DEADLINE|REVOKED)
          echo "[ERROR] Job ${job_id} finished with state=${state}" >&2
          return 1
          ;;
        *)
          remaining=$((remaining + 1))
          ;;
      esac
    done
    if (( remaining > 0 )); then
      sleep "${WAIT_INTERVAL_SECONDS}"
    fi
  done
}

read_metrics() {
  local summary_path="$1"
  "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

train = payload["results"]["train"]
validation = payload["results"]["validation"]
test = payload["results"]["test"]
print(
    f"{train.get('min_partition_accuracy')}\t"
    f"{validation.get('min_partition_accuracy')}\t"
    f"{validation.get('accuracy')}\t"
    f"{test.get('min_partition_accuracy')}"
)
PY
}

select_best_stage1() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import sys
from pathlib import Path

best = None
for raw_path in sys.argv[1:]:
    summary_path = Path(raw_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config_path = summary_path.with_name("config_args.json")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validation = payload["results"]["validation"]
    test = payload["results"]["test"]
    key = (
        float(validation["min_partition_accuracy"]),
        float(validation["accuracy"]),
        float(test["min_partition_accuracy"]),
    )
    if best is None or key > best[0]:
        best = (key, str(summary_path), payload, config)

if best is None:
    raise SystemExit("No stage-1 summaries were provided.")

payload = best[2]
config = best[3]
print(
    f"{best[1]}\t"
    f"{config['learning_rate']}\t"
    f"{config['train_per_partition']}\t"
    f"{payload['results']['validation']['min_partition_accuracy']}\t"
    f"{payload['results']['validation']['accuracy']}\t"
    f"{payload['results']['test']['min_partition_accuracy']}"
)
PY
}

write_best_summary() {
  local stage_name="$1"
  local summary_path="$2"
  local model_dir="$3"
  local out_path="${OUT_ROOT}/best_seed_summary.json"
  "${PYTHON_BIN}" - "${stage_name}" "${summary_path}" "${model_dir}" "${out_path}" <<'PY'
import json
import sys
from pathlib import Path

stage_name = sys.argv[1]
summary_path = Path(sys.argv[2])
model_dir = sys.argv[3]
out_path = Path(sys.argv[4])
payload = json.loads(summary_path.read_text(encoding="utf-8"))
record = {
    "winning_stage": stage_name,
    "summary_path": str(summary_path),
    "model_dir": model_dir,
    "train_examples": payload["train_examples"],
    "validation_examples": payload["validation_examples"],
    "test_examples": payload["test_examples"],
    "recipe": payload["recipe"],
    "format_version": payload["format_version"],
    "partitions": payload["partitions"],
    "training": payload["training"],
    "results": payload["results"],
}
out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
PY
}

cleanup_model_dir() {
  local path="$1"
  if [[ -n "${path}" && -d "${path}" ]]; then
    rm -rf "${path}"
  fi
}

stage0_dir="${OUT_ROOT}/stage0_overfit"
stage0_summary="${stage0_dir}/summary.json"
stage0_job="$(submit_job "${stage0_dir}" "5e-5" "${STAGE0_TRAIN_PER_PARTITION}" "${STAGE0_MAX_STEPS}" "0" "0")"
echo "[INFO] Stage 0 job_id=${stage0_job} output=${stage0_dir} summary=${stage0_summary}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for train_count in "${STAGE1_TRAIN_COUNTS[@]}"; do
    for lr in "${STAGE1_LRS[@]}"; do
      tag="$(lr_tag "${lr}")"
      out_dir="${OUT_ROOT}/stage1/train_${train_count}/lr_${tag}"
      echo "[INFO] Stage 1 dry-run job=train_${train_count}_lr_${tag} output=${out_dir} summary=${out_dir}/summary.json"
    done
  done
  write_status "dry_run"
  exit 0
fi

wait_for_jobs "${stage0_job}"
stage0_train_min="$(read_metrics "${stage0_summary}" | cut -f1)"
echo "[INFO] Stage 0 train_min_partition_accuracy=${stage0_train_min}"
if [[ "${stage0_train_min}" != "1.0" ]]; then
  write_status "stage0_failed"
  echo "[ERROR] Stage 0 overfit sanity failed: expected train min partition accuracy 1.0" >&2
  exit 1
fi

declare -a stage1_job_ids=()
declare -a stage1_summary_paths=()
for train_count in "${STAGE1_TRAIN_COUNTS[@]}"; do
  for lr in "${STAGE1_LRS[@]}"; do
    tag="$(lr_tag "${lr}")"
    out_dir="${OUT_ROOT}/stage1/train_${train_count}/lr_${tag}"
    summary_path="${out_dir}/summary.json"
    job_id="$(submit_job "${out_dir}" "${lr}" "${train_count}" "${MAX_STEPS}" "200" "0")"
    stage1_job_ids+=("${job_id}")
    stage1_summary_paths+=("${summary_path}")
    echo "[INFO] Stage 1 job_id=${job_id} train_per_partition=${train_count} lr=${lr} output=${out_dir} summary=${summary_path}"
  done
done

wait_for_jobs "${stage1_job_ids[@]}"

best_stage1="$(select_best_stage1 "${stage1_summary_paths[@]}")"
best_stage1_path="$(echo "${best_stage1}" | cut -f1)"
best_lr="$(echo "${best_stage1}" | cut -f2)"
best_train_per_partition="$(echo "${best_stage1}" | cut -f3)"
best_validation_min="$(echo "${best_stage1}" | cut -f4)"
best_validation_accuracy="$(echo "${best_stage1}" | cut -f5)"
best_test_min="$(echo "${best_stage1}" | cut -f6)"
printf '%s\n' "${best_stage1_path}" > "${OUT_ROOT}/stage1_best_result_path.txt"

echo "[INFO] Stage 1 winner summary=${best_stage1_path}"
echo "[INFO] Stage 1 winner lr=${best_lr} train_per_partition=${best_train_per_partition} validation_min=${best_validation_min} validation_acc=${best_validation_accuracy} test_min=${best_test_min}"

if ! "${PYTHON_BIN}" - "${best_validation_min}" <<'PY'
import sys
value = float(sys.argv[1])
raise SystemExit(0 if value >= 0.95 else 1)
PY
then
  write_status "non_viable_stage1"
  echo "[ERROR] No Stage 1 run reached validation min partition accuracy >= 0.95" >&2
  exit 1
fi

stage2_dir="${OUT_ROOT}/stage2/winner"
stage2_summary="${stage2_dir}/summary.json"
stage2_model_dir="${stage2_dir}/model"
stage2_job="$(submit_job "${stage2_dir}" "${best_lr}" "${best_train_per_partition}" "${MAX_STEPS}" "200" "1")"
echo "[INFO] Stage 2 job_id=${stage2_job} output=${stage2_dir} summary=${stage2_summary}"
wait_for_jobs "${stage2_job}"

stage2_metrics="$(read_metrics "${stage2_summary}")"
stage2_train_min="$(echo "${stage2_metrics}" | cut -f1)"
stage2_validation_min="$(echo "${stage2_metrics}" | cut -f2)"
stage2_test_min="$(echo "${stage2_metrics}" | cut -f4)"
echo "[INFO] Stage 2 train_min=${stage2_train_min} validation_min=${stage2_validation_min} test_min=${stage2_test_min}"

if "${PYTHON_BIN}" - "${stage2_test_min}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.95 else 1)
PY
then
  ln -sfn "${stage2_model_dir}" "${MODEL_LINK}"
  write_best_summary "stage2" "${stage2_summary}" "${stage2_model_dir}"
  write_status "completed_stage2"
  echo "[INFO] Updated ${MODEL_LINK} -> ${stage2_model_dir}"
  exit 0
fi

if "${PYTHON_BIN}" - "${stage2_test_min}" "${stage2_train_min}" <<'PY'
import sys
test_min = float(sys.argv[1])
train_min = float(sys.argv[2])
close_enough = 0.90 <= test_min <= 0.949
raise SystemExit(0 if close_enough and train_min >= 0.98 else 1)
PY
then
  stage3_dir="${OUT_ROOT}/stage3/train_${STAGE3_TRAIN_PER_PARTITION}"
  stage3_summary="${stage3_dir}/summary.json"
  stage3_model_dir="${stage3_dir}/model"
  stage3_job="$(submit_job "${stage3_dir}" "${best_lr}" "${STAGE3_TRAIN_PER_PARTITION}" "${MAX_STEPS}" "200" "1")"
  echo "[INFO] Stage 3 job_id=${stage3_job} output=${stage3_dir} summary=${stage3_summary}"
  wait_for_jobs "${stage3_job}"

  stage3_metrics="$(read_metrics "${stage3_summary}")"
  stage3_test_min="$(echo "${stage3_metrics}" | cut -f4)"
  echo "[INFO] Stage 3 test_min=${stage3_test_min}"

  if "${PYTHON_BIN}" - "${stage3_test_min}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) >= 0.95 else 1)
PY
  then
    cleanup_model_dir "${stage2_model_dir}"
    ln -sfn "${stage3_model_dir}" "${MODEL_LINK}"
    write_best_summary "stage3" "${stage3_summary}" "${stage3_model_dir}"
    write_status "completed_stage3"
    echo "[INFO] Updated ${MODEL_LINK} -> ${stage3_model_dir}"
    exit 0
  fi

  write_status "stage3_below_target"
  echo "[ERROR] Stage 3 did not reach test min partition accuracy >= 0.95" >&2
  exit 1
fi

write_status "stage2_below_target"
echo "[ERROR] Stage 2 was below target and did not qualify for Stage 3 escalation" >&2
exit 1

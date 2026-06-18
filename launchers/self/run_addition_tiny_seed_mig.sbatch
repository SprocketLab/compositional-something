#!/usr/bin/env bash
#SBATCH --job-name=add-tiny-seed
#SBATCH --output=artifacts/logs/add-tiny-seed-%j.out
#SBATCH --error=artifacts/logs/add-tiny-seed-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=mig
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MALLOC_CONF="${MALLOC_CONF:-background_thread:false}"

self_resolve_python

MODEL_NAME="meta/models/tiny_gpt2_8l_384d"
# Supported overrides: LR, TRAIN_PER_DIGIT, MAX_STEPS, TRAIN_BATCH_SIZE,
# EVAL_BATCH_SIZE, SEED, OUT_ROOT, SAVE_MODEL.
SEED="${SEED:-0}"
LR_OVERRIDE="${LR:-}"
TRAIN_PER_DIGIT="${TRAIN_PER_DIGIT:-50000}"
MAX_STEPS="${MAX_STEPS:-48000}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
SAVE_MODEL="${SAVE_MODEL:-1}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/addition_tiny_seed_mig_${TS}}"

STAGE1_TRAIN_PER_DIGIT=20000
STAGE1_EVAL_PER_DIGIT=200
STAGE1_MAX_STEPS=16000
STAGE2_EVAL_PER_DIGIT=200
STAGE3_TRAIN_PER_DIGIT=$(( TRAIN_PER_DIGIT * 2 ))
STAGE3_MAX_STEPS=$(( MAX_STEPS * 2 ))

mkdir -p "${OUT_ROOT}"

if [[ -n "${LR_OVERRIDE}" ]]; then
  STAGE1_LRS=("${LR_OVERRIDE}")
else
  STAGE1_LRS=("5e-5" "1e-4" "2e-4")
fi

run_or_dry() {
  self_print_command_stdout "$@"
  if self_parse_bool "${DRY_RUN}"; then
    return 0
  fi
  "$@"
}

write_status() {
  local status="$1"
  printf '%s\n' "${status}" > "${OUT_ROOT}/status.txt"
  echo "[INFO] Status: ${status}"
}

seed_fit_common_args() {
  local output_dir="$1"
  local learning_rate="$2"
  local min_digits="$3"
  local max_digits="$4"
  local train_per_digit="$5"
  local eval_per_digit="$6"
  local max_steps="$7"

  local -a cmd=(
    "${PYTHON_BIN}" -m self.seed_fit_experiment
    --task addition
    --model-name "${MODEL_NAME}"
    --output-dir "${output_dir}"
    --seed "${SEED}"
    --bf16
    --init-from-scratch
    --tokenizer-mode fixed_char
    --initial-min-size "${min_digits}"
    --initial-max-size "${max_digits}"
    --initial-train-per-size "${train_per_digit}"
    --initial-eval-per-size "${eval_per_digit}"
    --expand-num-size 1
    --expand-train-per-size 0
    --eval-per-size 0
    --composed-eval-per-size 0
    --num-expand-rounds 0
    --pseudo-label-mode none
    --num-epochs 1
    --learning-rate "${learning_rate}"
    --per-device-train-batch-size "${TRAIN_BATCH_SIZE}"
    --per-device-eval-batch-size "${EVAL_BATCH_SIZE}"
    --gradient-accumulation-steps 1
    --weight-decay 0
    --logging-steps 100
    --max-steps "${max_steps}"
    --decode-max-new-tokens 48
    --evaluate-train
  )
  printf '%s\0' "${cmd[@]}"
}

run_seed_fit_stage() {
  local output_dir="$1"
  local learning_rate="$2"
  local min_digits="$3"
  local max_digits="$4"
  local train_per_digit="$5"
  local eval_per_digit="$6"
  local max_steps="$7"
  local threshold="$8"
  local save_model_flag="$9"

  local -a cmd=()
  while IFS= read -r -d '' arg; do
    cmd+=("${arg}")
  done < <(seed_fit_common_args "${output_dir}" "${learning_rate}" "${min_digits}" "${max_digits}" "${train_per_digit}" "${eval_per_digit}" "${max_steps}")

  cmd+=(--target-accuracy-threshold "${threshold}")
  if self_parse_bool "${save_model_flag}"; then
    cmd+=(--save-model)
  fi

  run_or_dry "${cmd[@]}"
}

read_overfit_accuracy() {
  local result_path="$1"
  "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["eval_accuracy"])
PY
}

read_seed_metrics() {
  local result_path="$1"
  "${PYTHON_BIN}" - "${result_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

train_min = payload["results"].get("train", {}).get("min_per_size_accuracy")
validation_min = payload.get("validation_min_per_size_accuracy")
test_min = payload.get("test_min_per_size_accuracy")
print(f"{train_min}\t{validation_min}\t{test_min}")
PY
}

select_best_stage1_result() {
  "${PYTHON_BIN}" - "$@" <<'PY'
import json
import sys

best = None
for raw_path in sys.argv[1:]:
    with open(raw_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validation_min = float(payload["validation_min_per_size_accuracy"])
    test_min = float(payload["test_min_per_size_accuracy"])
    learning_rate = float(payload["training"]["learning_rate"])
    key = (validation_min, test_min, -learning_rate)
    if best is None or key > best[0]:
        best = (key, raw_path, payload)

if best is None:
    raise SystemExit("No stage-1 results were provided.")

payload = best[2]
print(
    f"{best[1]}\t"
    f"{payload['training']['learning_rate']}\t"
    f"{payload['validation_min_per_size_accuracy']}\t"
    f"{payload['test_min_per_size_accuracy']}"
)
PY
}

cleanup_saved_models() {
  local winner_model_dir="${1:-}"
  shift || true
  local candidate
  for candidate in "$@"; do
    if [[ -z "${candidate}" || ! -d "${candidate}" ]]; then
      continue
    fi
    if [[ -n "${winner_model_dir}" && "${candidate}" == "${winner_model_dir}" ]]; then
      continue
    fi
    rm -rf "${candidate}"
  done
}

echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Seed: ${SEED}"
echo "[INFO] Dry run: ${DRY_RUN}"
echo "[INFO] Train batch size: ${TRAIN_BATCH_SIZE}"
echo "[INFO] Eval batch size: ${EVAL_BATCH_SIZE}"
echo "[INFO] Stage 1 learning rates: ${STAGE1_LRS[*]}"
echo "[INFO] Stage 2 train_per_digit=${TRAIN_PER_DIGIT} max_steps=${MAX_STEPS}"
echo "[INFO] Stage 3 train_per_digit=${STAGE3_TRAIN_PER_DIGIT} max_steps=${STAGE3_MAX_STEPS}"
echo "[INFO] Save model: ${SAVE_MODEL}"
echo "[INFO] Slurm job id: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] Dry run: skipping CUDA and scratch-model config probes."
else
  nvidia-smi -L || true

  "${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"
  MODEL_NAME="${MODEL_NAME}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
from transformers import AutoConfig

model_name = os.environ["MODEL_NAME"]
model_path = Path(model_name)
if not model_path.exists():
    raise FileNotFoundError(f"Scratch model config directory not found: {model_name}")
AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(f"[INFO] scratch_model_config_ok={model_name}")
PY
fi

STAGE0_ROOT="${OUT_ROOT}/stage0_smoke"
STAGE0_RESULT="${STAGE0_ROOT}/addition/base/overfit_result.json"

echo "[INFO] Stage 0: addition/base overfit sanity on MIG."
run_or_dry \
  "${PYTHON_BIN}" -m self.check_self_improvement_overfit \
  --tasks addition \
  --settings base \
  --model-name "${MODEL_NAME}" \
  --output-dir "${STAGE0_ROOT}" \
  --num-examples 10 \
  --max-steps 200 \
  --num-epochs 1 \
  --learning-rate 1e-4 \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps 1 \
  --bf16 \
  --init-from-scratch \
  --tokenizer-mode fixed_char

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] Dry run: printing Stage 1 sweep commands."
  mkdir -p "${OUT_ROOT}/stage1"
  for candidate_lr in "${STAGE1_LRS[@]}"; do
    stage_tag="$(echo "${candidate_lr}" | tr '+.' 'pm' | tr '-' '_')"
    stage_dir="${OUT_ROOT}/stage1/lr_${stage_tag}"
    run_seed_fit_stage "${stage_dir}" "${candidate_lr}" 3 5 "${STAGE1_TRAIN_PER_DIGIT}" "${STAGE1_EVAL_PER_DIGIT}" "${STAGE1_MAX_STEPS}" 0.90 0
  done
  write_status "dry_run"
  exit 0
fi

STAGE0_ACCURACY="$(read_overfit_accuracy "${STAGE0_RESULT}")"
echo "[INFO] Stage 0 eval_accuracy=${STAGE0_ACCURACY}"
if [[ "${STAGE0_ACCURACY}" != "1.0" ]]; then
  write_status "stage0_smoke_failed"
  echo "[ERROR] Stage 0 failed: expected eval_accuracy=1.0 before running the seed sweep."
  exit 1
fi

declare -a STAGE1_RESULT_PATHS=()
mkdir -p "${OUT_ROOT}/stage1"

for candidate_lr in "${STAGE1_LRS[@]}"; do
  stage_tag="$(echo "${candidate_lr}" | tr '+.' 'pm' | tr '-' '_')"
  stage_dir="${OUT_ROOT}/stage1/lr_${stage_tag}"
  echo "[INFO] Stage 1: lr=${candidate_lr}"
  run_seed_fit_stage "${stage_dir}" "${candidate_lr}" 3 5 "${STAGE1_TRAIN_PER_DIGIT}" "${STAGE1_EVAL_PER_DIGIT}" "${STAGE1_MAX_STEPS}" 0.90 0
  STAGE1_RESULT_PATHS+=("${stage_dir}/seed_fit_results.json")
done

IFS=$'\t' read -r BEST_STAGE1_RESULT BEST_LR BEST_STAGE1_VAL BEST_STAGE1_TEST < <(
  select_best_stage1_result "${STAGE1_RESULT_PATHS[@]}"
)

printf '%s\n' "${BEST_LR}" > "${OUT_ROOT}/stage1/best_lr.txt"
printf '%s\n' "${BEST_STAGE1_RESULT}" > "${OUT_ROOT}/stage1/best_result_path.txt"
echo "[INFO] Stage 1 selected lr=${BEST_LR} validation_min=${BEST_STAGE1_VAL} test_min=${BEST_STAGE1_TEST}"

if ! "${PYTHON_BIN}" - "${BEST_STAGE1_VAL}" <<'PY'
import sys

value = float(sys.argv[1])
raise SystemExit(0 if value >= 0.90 else 1)
PY
then
  write_status "non_viable_stage1"
  echo "[WARN] No Stage 1 run reached validation_min_per_size_accuracy >= 0.90 across digits 3..5."
  exit 0
fi

STAGE2_DIR="${OUT_ROOT}/stage2"
echo "[INFO] Stage 2: lr=${BEST_LR}"
run_seed_fit_stage "${STAGE2_DIR}" "${BEST_LR}" 3 7 "${TRAIN_PER_DIGIT}" "${STAGE2_EVAL_PER_DIGIT}" "${MAX_STEPS}" 0.95 "${SAVE_MODEL}"

IFS=$'\t' read -r STAGE2_TRAIN_MIN STAGE2_VAL_MIN STAGE2_TEST_MIN < <(
  read_seed_metrics "${STAGE2_DIR}/seed_fit_results.json"
)
echo "[INFO] Stage 2 train_min=${STAGE2_TRAIN_MIN} validation_min=${STAGE2_VAL_MIN} test_min=${STAGE2_TEST_MIN}"

STAGE2_MODEL_DIR="${STAGE2_DIR}/model"
STAGE3_DIR="${OUT_ROOT}/stage3"
STAGE3_MODEL_DIR="${STAGE3_DIR}/model"
WINNER_STAGE=""
WINNER_MODEL_DIR=""

if "${PYTHON_BIN}" - "${STAGE2_TEST_MIN}" <<'PY'
import sys

value = float(sys.argv[1])
raise SystemExit(0 if value >= 0.95 else 1)
PY
then
  WINNER_STAGE="stage2"
  WINNER_MODEL_DIR="${STAGE2_MODEL_DIR}"
else
  if "${PYTHON_BIN}" - "${STAGE2_TRAIN_MIN}" "${STAGE2_VAL_MIN}" "${STAGE2_TEST_MIN}" <<'PY'
import sys

train_min = float(sys.argv[1])
validation_min = float(sys.argv[2])
test_min = float(sys.argv[3])
gate_metric = min(validation_min, test_min)
raise SystemExit(0 if gate_metric >= 0.90 and gate_metric < 0.95 and train_min >= 0.98 else 1)
PY
  then
    echo "[INFO] Stage 3: escalating seed data and steps."
    run_seed_fit_stage "${STAGE3_DIR}" "${BEST_LR}" 3 7 "${STAGE3_TRAIN_PER_DIGIT}" "${STAGE2_EVAL_PER_DIGIT}" "${STAGE3_MAX_STEPS}" 0.95 "${SAVE_MODEL}"
    IFS=$'\t' read -r STAGE3_TRAIN_MIN STAGE3_VAL_MIN STAGE3_TEST_MIN < <(
      read_seed_metrics "${STAGE3_DIR}/seed_fit_results.json"
    )
    echo "[INFO] Stage 3 train_min=${STAGE3_TRAIN_MIN} validation_min=${STAGE3_VAL_MIN} test_min=${STAGE3_TEST_MIN}"
    if "${PYTHON_BIN}" - "${STAGE3_TEST_MIN}" <<'PY'
import sys

value = float(sys.argv[1])
raise SystemExit(0 if value >= 0.95 else 1)
PY
    then
      WINNER_STAGE="stage3"
      WINNER_MODEL_DIR="${STAGE3_MODEL_DIR}"
    else
      write_status "stage3_below_threshold"
    fi
  else
    if "${PYTHON_BIN}" - "${STAGE2_TRAIN_MIN}" <<'PY'
import sys

value = float(sys.argv[1])
raise SystemExit(0 if value < 0.98 else 1)
PY
    then
      write_status "stage2_capacity_limited"
    else
      write_status "stage2_not_close_enough_for_escalation"
    fi
  fi
fi

if [[ -z "${WINNER_STAGE}" ]]; then
  cleanup_saved_models "" "${STAGE2_MODEL_DIR}" "${STAGE3_MODEL_DIR}"
  echo "[WARN] No seed model met the Stage 2/3 acceptance threshold."
  exit 0
fi

if ! self_parse_bool "${SAVE_MODEL}"; then
  write_status "${WINNER_STAGE}_success_no_saved_model"
  echo "[WARN] ${WINNER_STAGE} met the threshold, but SAVE_MODEL=${SAVE_MODEL} prevented saving a reusable checkpoint."
  exit 0
fi

if [[ ! -d "${WINNER_MODEL_DIR}" ]]; then
  write_status "${WINNER_STAGE}_missing_model_dir"
  echo "[ERROR] Expected winner model directory is missing: ${WINNER_MODEL_DIR}"
  exit 1
fi

cleanup_saved_models "${WINNER_MODEL_DIR}" "${STAGE2_MODEL_DIR}" "${STAGE3_MODEL_DIR}"
self_update_symlink "${WINNER_MODEL_DIR}" "${OUT_ROOT}/seed_model"
printf '%s\n' "${WINNER_MODEL_DIR}" > "${OUT_ROOT}/winner_model_path.txt"
printf '%s\n' "${WINNER_STAGE}" > "${OUT_ROOT}/winner_stage.txt"
write_status "${WINNER_STAGE}_success"
echo "[INFO] Winner: ${WINNER_STAGE}"
echo "[INFO] Seed checkpoint ready at ${OUT_ROOT}/seed_model"

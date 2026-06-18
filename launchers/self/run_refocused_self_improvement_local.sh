#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/self/lib/self_common.sh
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

PRINT_ONLY=0
RUN_SCOPE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --only-addition)
      RUN_SCOPE="addition"
      shift
      ;;
    --only-bit)
      RUN_SCOPE="bit"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BASE_OUT="${BASE_OUT:-${ROOT_DIR}/artifacts/runs/self_improvement_refocus_20260416}"
ADDITION_MODEL_NAME="${ADDITION_MODEL_NAME:-HuggingFaceTB/SmolLM2-360M}"
BIT_MODEL_NAME="${BIT_MODEL_NAME:-${ROOT_DIR}/meta/models/tiny_gpt2_8l_384d}"
ADDITION_GPUS="${ADDITION_GPUS:-0 1 2 3 4}"
BIT_GPUS="${BIT_GPUS:-5 6 7}"

read -r -a ADDITION_GPU_LIST <<< "${ADDITION_GPUS}"
read -r -a BIT_GPU_LIST <<< "${BIT_GPUS}"
if [[ "${RUN_SCOPE}" != "bit" && ${#ADDITION_GPU_LIST[@]} -lt 5 ]]; then
  echo "Need at least 5 GPUs in ADDITION_GPUS to run the 5 addition baselines." >&2
  exit 1
fi
if [[ "${RUN_SCOPE}" != "addition" && ${#BIT_GPU_LIST[@]} -lt 1 ]]; then
  echo "Need at least 1 GPU in BIT_GPUS." >&2
  exit 1
fi

mkdir -p "${BASE_OUT}"

RUN_LABELS=()
RUN_OUT_DIRS=()
RUN_COMMANDS=()
RUN_POOLS=()

append_run() {
  local pool="$1"
  local label="$2"
  local out_dir="$3"
  shift 3
  RUN_POOLS+=("${pool}")
  RUN_LABELS+=("${label}")
  RUN_OUT_DIRS+=("${out_dir}")
  RUN_COMMANDS+=("$(printf '%q ' "$@")")
}

COMMON_ENV=(
  PYTORCH_NVML_BASED_CUDA_CHECK=1
  OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  TOKENIZERS_PARALLELISM=false
)

add_addition_runs() {
  local task_out="${BASE_OUT}/addition"
  local -a common_args=(
    python -m self.legacy.addition_self_improvement
    --model-name "${ADDITION_MODEL_NAME}"
    --bf16
    --skip-save-model
    --resume
    --initial-min-digits 3
    --initial-max-digits 7
    --initial-train-per-digit 20000
    --initial-eval-per-digit 100
    --num-expand-rounds 8
    --expand-num-digits 2
    --expand-train-per-digit 1200
    --eval-per-digit 100
    --composed-eval-per-digit 50
    --num-epochs 3
    --learning-rate 5e-5
    --per-device-train-batch-size 8
    --per-device-eval-batch-size 16
    --gradient-accumulation-steps 1
    --weight-decay 0
    --logging-steps 100
    --decode-max-new-tokens 48
    --composed-refresh-mode dynamic
  )

  local baseline
  for baseline in short_only direct with_carry with_carry_filtered compose_corrupt; do
    local out_dir="${task_out}/${baseline}"
    local -a mode_args=()
    case "${baseline}" in
      short_only) mode_args=(--pseudo-label-mode none) ;;
      direct) mode_args=(--pseudo-label-mode direct) ;;
      with_carry) mode_args=(--pseudo-label-mode compose --composed-strategy with_carry) ;;
      with_carry_filtered)
        mode_args=(--pseudo-label-mode compose --composed-strategy with_carry_filtered --composition-error-percent 0)
        ;;
      compose_corrupt)
        mode_args=(--pseudo-label-mode compose_corrupt --composed-strategy with_carry --corruption-rate 0.10)
        ;;
      *)
        echo "Unknown addition baseline: ${baseline}" >&2
        exit 1
        ;;
    esac
    append_run addition "${baseline}" "${out_dir}" "${common_args[@]}" --output-dir "${out_dir}" "${mode_args[@]}"
  done
}

add_bit_runs() {
  local task_name="$1"
  local module_name="$2"
  local task_out="${BASE_OUT}/${task_name}"
  local -a common_args=(
    python -m "${module_name}"
    --model-name "${BIT_MODEL_NAME}"
    --init-from-scratch
    --tokenizer-mode fixed_char
    --bf16
    --skip-save-model
    --resume
    --format-version legacy
    --initial-min-bits 8
    --initial-max-bits 16
    --initial-train-per-bit 100000
    --initial-eval-per-bit 50
    --num-expand-rounds 4
    --expand-num-bits 4
    --expand-train-per-bit 1200
    --eval-per-bit 50
    --composed-eval-per-bit 50
    --num-epochs 3
    --learning-rate 1e-4
    --per-device-train-batch-size 32
    --per-device-eval-batch-size 64
    --gradient-accumulation-steps 1
    --weight-decay 0
    --logging-steps 100
    --decode-max-new-tokens 16
    --composed-refresh-mode dynamic
    --reserve-heldout-first
    --reserve-shared-eval-first
  )

  local baseline
  for baseline in short_only direct compose compose_corrupt; do
    local out_dir="${task_out}/${baseline}"
    local -a mode_args=()
    case "${baseline}" in
      short_only) mode_args=(--pseudo-label-mode none) ;;
      direct) mode_args=(--pseudo-label-mode direct) ;;
      compose) mode_args=(--pseudo-label-mode compose) ;;
      compose_corrupt) mode_args=(--pseudo-label-mode compose_corrupt --corruption-rate 0.10) ;;
      *)
        echo "Unknown ${task_name} baseline: ${baseline}" >&2
        exit 1
        ;;
    esac
    append_run bit "${task_name}/${baseline}" "${out_dir}" "${common_args[@]}" --output-dir "${out_dir}" "${mode_args[@]}"
  done
}

if [[ "${RUN_SCOPE}" != "bit" ]]; then
  add_addition_runs
fi
if [[ "${RUN_SCOPE}" != "addition" ]]; then
  add_bit_runs "run_length" "self.legacy.run_length_self_improvement"
fi

MANIFEST_PATH="${BASE_OUT}/manifest.tsv"
{
  printf "index\tpool\tlabel\toutput_dir\tcommand\n"
  for idx in "${!RUN_LABELS[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${idx}" \
      "${RUN_POOLS[$idx]}" \
      "${RUN_LABELS[$idx]}" \
      "${RUN_OUT_DIRS[$idx]}" \
      "${RUN_COMMANDS[$idx]}"
  done
} > "${MANIFEST_PATH}"

if self_parse_bool "${PRINT_ONLY}"; then
  echo "[INFO] Manifest written to ${MANIFEST_PATH}"
  for idx in "${!RUN_LABELS[@]}"; do
    printf "[%02d] [%s] %s -> %s\n" \
      "${idx}" \
      "${RUN_POOLS[$idx]}" \
      "${RUN_LABELS[$idx]}" \
      "${RUN_OUT_DIRS[$idx]}"
    printf "  %s\n" "${RUN_COMMANDS[$idx]}"
  done
  exit 0
fi

run_worker() {
  local gpu="$1"
  shift
  local index
  for index in "$@"; do
    local label="${RUN_LABELS[$index]}"
    local out_dir="${RUN_OUT_DIRS[$index]}"
    local cmd="${RUN_COMMANDS[$index]}"
    local log_path="${out_dir}/launcher.log"
    mkdir -p "${out_dir}"
    printf "%s\n" "${cmd}" > "${out_dir}/launch_command.txt"
    {
      echo "[INFO] $(date --iso-8601=seconds) starting ${label} on GPU ${gpu}"
      echo "[INFO] output_dir=${out_dir}"
      echo "[INFO] command=${cmd}"
    } | tee "${log_path}"
    (
      cd "${ROOT_DIR}"
      env CUDA_VISIBLE_DEVICES="${gpu}" "${COMMON_ENV[@]}" bash -lc "${cmd}"
    ) >> "${log_path}" 2>&1
    {
      echo "[INFO] $(date --iso-8601=seconds) finished ${label} on GPU ${gpu}"
    } >> "${log_path}"
  done
}

declare -a ADDITION_INDEXES=()
declare -a BIT_INDEXES=()
for idx in "${!RUN_LABELS[@]}"; do
  if [[ "${RUN_POOLS[$idx]}" == "addition" ]]; then
    ADDITION_INDEXES+=("${idx}")
  else
    BIT_INDEXES+=("${idx}")
  fi
done

PIDS=()

if [[ ${#ADDITION_INDEXES[@]} -gt 0 ]]; then
  for gpu_slot in "${!ADDITION_INDEXES[@]}"; do
    idx="${ADDITION_INDEXES[$gpu_slot]}"
    gpu="${ADDITION_GPU_LIST[$gpu_slot]}"
    run_worker "${gpu}" "${idx}" &
    PIDS+=("$!")
  done
fi

if [[ ${#BIT_INDEXES[@]} -gt 0 ]]; then
  for bit_slot in "${!BIT_GPU_LIST[@]}"; do
    eval "BIT_QUEUE_${bit_slot}=()"
  done

  for i in "${!BIT_INDEXES[@]}"; do
    slot=$((i % ${#BIT_GPU_LIST[@]}))
    eval "BIT_QUEUE_${slot}+=(\"${BIT_INDEXES[$i]}\")"
  done

  for bit_slot in "${!BIT_GPU_LIST[@]}"; do
    gpu="${BIT_GPU_LIST[$bit_slot]}"
    eval "queue=(\"\${BIT_QUEUE_${bit_slot}[@]}\")"
    if [[ ${#queue[@]} -eq 0 ]]; then
      continue
    fi
    run_worker "${gpu}" "${queue[@]}" &
    PIDS+=("$!")
  done
fi

echo "[INFO] Launching ${#RUN_LABELS[@]} runs."
if [[ ${#ADDITION_INDEXES[@]} -gt 0 ]]; then
  echo "[INFO] Addition GPUs: ${ADDITION_GPU_LIST[*]}"
fi
if [[ ${#BIT_INDEXES[@]} -gt 0 ]]; then
  echo "[INFO] Run-length bit-string GPUs: ${BIT_GPU_LIST[*]}"
fi
echo "[INFO] Manifest written to ${MANIFEST_PATH}"

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[INFO] Completed refocused self-improvement batch."

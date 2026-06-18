#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/self/lib/self_common.sh
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

PRINT_ONLY=0
GPU_IDS_CLI=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --gpus)
      GPU_IDS_CLI="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

BASE_OUT="${BASE_OUT:-${ROOT_DIR}/artifacts/runs/workshop_local}"
DEFAULT_LOCAL_MODEL="${ROOT_DIR}/artifacts/models/SmolLM2-360M"
DEFAULT_HUB_MODEL="HuggingFaceTB/SmolLM2-360M"
if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_LOCAL_MODEL}}"
else
  MODEL_NAME="${MODEL_NAME:-${DEFAULT_HUB_MODEL}}"
fi

if [[ -n "${GPU_IDS_CLI}" ]]; then
  GPU_IDS="${GPU_IDS_CLI}"
elif [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS="${GPU_IDS}"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd' ' -)"
else
  GPU_IDS="0"
fi

read -r -a GPU_LIST <<< "${GPU_IDS}"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "No GPUs resolved. Pass --gpus or set GPU_IDS." >&2
  exit 1
fi

mkdir -p "${BASE_OUT}"

COMMON_ARGS=(
  --model-name "${MODEL_NAME}"
  --bf16
  --num-epochs 3
  --learning-rate 5e-5
  --per-device-train-batch-size 4
  --per-device-eval-batch-size 4
  --gradient-accumulation-steps 1
  --weight-decay 0
  --composed-refresh-mode dynamic
  --resume
)

RUN_LABELS=()
RUN_OUT_DIRS=()
RUN_COMMANDS=()

append_run() {
  local label="$1"
  local out_dir="$2"
  shift 2
  RUN_LABELS+=("${label}")
  RUN_OUT_DIRS+=("${out_dir}")
  RUN_COMMANDS+=("$(printf '%q ' "$@")")
}

add_run_length_runs() {
  local task_name="$1"
  local module_name="$2"
  local task_out="${BASE_OUT}/${task_name}"
  local task_args=(
    --output-dir
    ""
    --format-version legacy
    --initial-min-bits 4
    --initial-max-bits 8
    --initial-train-per-bit 2000
    --initial-eval-per-bit 50
    --num-expand-rounds 4
    --expand-num-bits 4
    --expand-train-per-bit 1200
    --eval-per-bit 100
    --composed-eval-per-bit 50
    --decode-max-new-tokens 16
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
        echo "Unknown baseline: ${baseline}" >&2
        exit 1
        ;;
    esac
    local -a argv=(python -m "${module_name}" "${COMMON_ARGS[@]}")
    local -a specific=("${task_args[@]}")
    specific[1]="${out_dir}"
    argv+=("${specific[@]}" "${mode_args[@]}")
    append_run "${task_name}/${baseline}" "${out_dir}" "${argv[@]}"
  done
}

add_addition_runs() {
  local task_out="${BASE_OUT}/addition"
  local addition_args=(
    --output-dir
    ""
    --initial-min-digits 3
    --initial-max-digits 7
    --initial-train-per-digit 2000
    --initial-eval-per-digit 50
    --num-expand-rounds 8
    --expand-num-digits 2
    --expand-train-per-digit 1200
    --eval-per-digit 100
    --composed-eval-per-digit 50
    --decode-max-new-tokens 48
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
        echo "Unknown baseline: ${baseline}" >&2
        exit 1
        ;;
    esac
    local -a argv=(python -m self.self_improvement "${COMMON_ARGS[@]}")
    local -a specific=("${addition_args[@]}")
    specific[1]="${out_dir}"
    argv+=("${specific[@]}" "${mode_args[@]}")
    append_run "addition/${baseline}" "${out_dir}" "${argv[@]}"
  done
}

add_run_length_runs "run_length" "self.run_length_self_improvement"
add_addition_runs

MANIFEST_PATH="${BASE_OUT}/manifest.tsv"
{
  printf "index\ttask\tbaseline\toutput_dir\tcommand\n"
  for idx in "${!RUN_LABELS[@]}"; do
    task="${RUN_LABELS[$idx]%/*}"
    baseline="${RUN_LABELS[$idx]#*/}"
    printf "%s\t%s\t%s\t%s\t%s\n" \
      "${idx}" \
      "${task}" \
      "${baseline}" \
      "${RUN_OUT_DIRS[$idx]}" \
      "${RUN_COMMANDS[$idx]}"
  done
} > "${MANIFEST_PATH}"

if self_parse_bool "${PRINT_ONLY}"; then
  echo "[INFO] Workshop manifest (${#RUN_LABELS[@]} runs)"
  echo "[INFO] Manifest written to ${MANIFEST_PATH}"
  for idx in "${!RUN_LABELS[@]}"; do
    printf "[%02d] %s -> %s\n" "${idx}" "${RUN_LABELS[$idx]}" "${RUN_OUT_DIRS[$idx]}"
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
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTORCH_NVML_BASED_CUDA_CHECK=1 \
      OPENBLAS_NUM_THREADS=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      bash -lc "${cmd}"
    ) >> "${log_path}" 2>&1
    {
      echo "[INFO] $(date --iso-8601=seconds) finished ${label} on GPU ${gpu}"
    } >> "${log_path}"
  done
}

for gpu_index in "${!GPU_LIST[@]}"; do
  eval "GPU_JOB_INDEXES_${gpu_index}=()"
done

for idx in "${!RUN_LABELS[@]}"; do
  worker_slot=$((idx % ${#GPU_LIST[@]}))
  eval "GPU_JOB_INDEXES_${worker_slot}+=(\"${idx}\")"
done

echo "[INFO] Launching ${#RUN_LABELS[@]} runs across GPUs: ${GPU_LIST[*]}"
echo "[INFO] Manifest written to ${MANIFEST_PATH}"

PIDS=()
for gpu_index in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$gpu_index]}"
  eval "worker_jobs=(\"\${GPU_JOB_INDEXES_${gpu_index}[@]}\")"
  if [[ ${#worker_jobs[@]} -eq 0 ]]; then
    continue
  fi
  run_worker "${gpu}" "${worker_jobs[@]}" &
  PIDS+=("$!")
done

for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "[INFO] Completed workshop local batch."

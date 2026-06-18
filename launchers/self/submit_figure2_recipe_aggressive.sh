#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root
self_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/figure2_recipe_aggressive_${TS}}"
JOB_NAME="${JOB_NAME:-fig2-rec-aggr}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_TIME="${SBATCH_TIME:-48:00:00}"
SBATCH_GRES="${SBATCH_GRES:-}"
SBATCH_MEM="${SBATCH_MEM:-64G}"
SBATCH_CPUS="${SBATCH_CPUS:-1}"
SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-}"
STAGE="${STAGE:-all}"
TASKS="${TASKS:-run_length}"
BASELINES="${BASELINES:-short_only direct compose compose_corrupt}"
DEVICE_TARGET="${DEVICE_TARGET:-local_a100_40gb}"
DRY_RUN="${DRY_RUN:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-}"
MAX_STEPS_SEED="${MAX_STEPS_SEED:-}"
MAX_STEPS_ROUND="${MAX_STEPS_ROUND:-}"
SEED="${SEED:-}"
RUN_FULLPACK_ONLY_IF_HEALTHY="${RUN_FULLPACK_ONLY_IF_HEALTHY:-}"
INITIAL_TRAIN_PER_BIT="${INITIAL_TRAIN_PER_BIT:-}"
INITIAL_EVAL_PER_BIT="${INITIAL_EVAL_PER_BIT:-}"
EXPAND_TRAIN_PER_BIT="${EXPAND_TRAIN_PER_BIT:-}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-}"
EXPAND_NUM_BITS="${EXPAND_NUM_BITS:-}"
RUN_LENGTH_NUM_EXPAND_ROUNDS="${RUN_LENGTH_NUM_EXPAND_ROUNDS:-}"
RUN_LENGTH_EXPAND_NUM_BITS="${RUN_LENGTH_EXPAND_NUM_BITS:-}"
COMPOSED_EVAL_PER_BIT="${COMPOSED_EVAL_PER_BIT:-}"

mkdir -p "${ROOT_DIR}/artifacts/logs"
stdout_log="${ROOT_DIR}/artifacts/logs/${JOB_NAME}-%j.out"
stderr_log="${ROOT_DIR}/artifacts/logs/${JOB_NAME}-%j.err"

wrap_cmd=(
  env
  "PYTHON_BIN=${PYTHON_BIN}"
  "OUT_ROOT=${OUT_ROOT}"
  "STAGE=${STAGE}"
  "TASKS=${TASKS}"
  "BASELINES=${BASELINES}"
  "DEVICE_TARGET=${DEVICE_TARGET}"
  "DRY_RUN=${DRY_RUN}"
)

append_optional_env() {
  local name="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    wrap_cmd+=("${name}=${value}")
  fi
}

append_optional_env "TRAIN_BATCH_SIZE" "${TRAIN_BATCH_SIZE}"
append_optional_env "EVAL_BATCH_SIZE" "${EVAL_BATCH_SIZE}"
append_optional_env "MAX_STEPS_SEED" "${MAX_STEPS_SEED}"
append_optional_env "MAX_STEPS_ROUND" "${MAX_STEPS_ROUND}"
append_optional_env "SEED" "${SEED}"
append_optional_env "RUN_FULLPACK_ONLY_IF_HEALTHY" "${RUN_FULLPACK_ONLY_IF_HEALTHY}"
append_optional_env "INITIAL_TRAIN_PER_BIT" "${INITIAL_TRAIN_PER_BIT}"
append_optional_env "INITIAL_EVAL_PER_BIT" "${INITIAL_EVAL_PER_BIT}"
append_optional_env "EXPAND_TRAIN_PER_BIT" "${EXPAND_TRAIN_PER_BIT}"
append_optional_env "NUM_EXPAND_ROUNDS" "${NUM_EXPAND_ROUNDS}"
append_optional_env "EXPAND_NUM_BITS" "${EXPAND_NUM_BITS}"
append_optional_env "RUN_LENGTH_NUM_EXPAND_ROUNDS" "${RUN_LENGTH_NUM_EXPAND_ROUNDS}"
append_optional_env "RUN_LENGTH_EXPAND_NUM_BITS" "${RUN_LENGTH_EXPAND_NUM_BITS}"
append_optional_env "COMPOSED_EVAL_PER_BIT" "${COMPOSED_EVAL_PER_BIT}"
wrap_cmd+=(bash "${ROOT_DIR}/launchers/self/run_figure2_recipe_aggressive.sh")

self_print_prefixed_command_stdout "Submission command" "${wrap_cmd[@]}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Stdout log: ${stdout_log}"
echo "[INFO] Stderr log: ${stderr_log}"

if [[ -z "${SBATCH_PARTITION}" || -z "${SBATCH_GRES}" ]]; then
  case "${DEVICE_TARGET}" in
    local_a100_40gb)
      SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
      SBATCH_GRES="${SBATCH_GRES:-gpu:a100:1}"
      SBATCH_CONSTRAINT="${SBATCH_CONSTRAINT:-a100&gpu40&nomig}"
      ;;
    mig_10gb)
      SBATCH_PARTITION="${SBATCH_PARTITION:-mig}"
      SBATCH_GRES="${SBATCH_GRES:-gpu:1g.10gb:1}"
      ;;
    all_gpu)
      SBATCH_PARTITION="${SBATCH_PARTITION:-all}"
      SBATCH_GRES="${SBATCH_GRES:-gpu:1}"
      ;;
    *)
      echo "[ERROR] Unsupported DEVICE_TARGET=${DEVICE_TARGET}" >&2
      exit 1
      ;;
  esac
fi

echo "[INFO] Resolved partition: ${SBATCH_PARTITION}"
echo "[INFO] Resolved gres: ${SBATCH_GRES}"
if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
  echo "[INFO] Resolved constraint: ${SBATCH_CONSTRAINT}"
fi

if self_parse_bool "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=1; sbatch not executed."
  exit 0
fi

sbatch_args=(
  --parsable
  --partition="${SBATCH_PARTITION}"
  --time="${SBATCH_TIME}"
  --gres="${SBATCH_GRES}"
  --mem="${SBATCH_MEM}"
  --cpus-per-task="${SBATCH_CPUS}"
  --job-name="${JOB_NAME}"
  --output="${stdout_log}"
  --error="${stderr_log}"
)

if [[ -n "${SBATCH_CONSTRAINT}" ]]; then
  sbatch_args+=(--constraint="${SBATCH_CONSTRAINT}")
fi

sbatch_args+=(--wrap="$(printf '%q ' "${wrap_cmd[@]}")")

job_id="$(sbatch "${sbatch_args[@]}")"

echo "[INFO] Submitted ${JOB_NAME} -> ${job_id}"

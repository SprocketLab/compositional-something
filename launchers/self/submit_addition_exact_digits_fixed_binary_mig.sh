#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/addition_exact_digits_fixed_binary_${TS}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/manifest.tsv}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/addition_recipe_seed_best}"
FULLPACK_LAUNCHER="${FULLPACK_LAUNCHER:-${ROOT_DIR}/launchers/self/run_addition_recipe_fullpack.sh}"

self_set_sbatch_defaults "mig" "gpu:1g.10gb:1" "1" "64G" "48:00:00"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-8}"
EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS:-2}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-5000}"
EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-10000}"
ADDITION_COMPOSITION_PATH_MODE="${ADDITION_COMPOSITION_PATH_MODE:-fixed_binary}"
DRY_RUN="${DRY_RUN:-0}"

BASELINES=(short_only direct with_carry with_carry_filtered compose_corrupt)

mkdir -p "${LOG_DIR}"
printf "job_id\tbaseline\tcomposition_path_mode\tout_dir\tresults_path\n" > "${MANIFEST}"

submit_baseline() {
  local baseline="$1"
  local out_dir="${RUN_ROOT}/${baseline}"
  local results_path="${out_dir}/self_improvement_results.json"
  local job_name="add-exfb-${baseline//_/-}"
  local cmd
  cmd="$(
    self_wrap_env_command \
      "${FULLPACK_LAUNCHER}" \
      "OUT_ROOT=${RUN_ROOT}" \
      "SEED_MODEL=${SEED_MODEL}" \
      "BASELINE=${baseline}" \
      "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \
      "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}" \
      "NUM_EXPAND_ROUNDS=${NUM_EXPAND_ROUNDS}" \
      "EXPAND_NUM_DIGITS=${EXPAND_NUM_DIGITS}" \
      "SEED_REPLAY_TRAIN_PER_DIGIT=${SEED_REPLAY_TRAIN_PER_DIGIT}" \
      "EXPAND_TRAIN_PER_DIGIT=${EXPAND_TRAIN_PER_DIGIT}" \
      "ADDITION_COMPOSITION_PATH_MODE=${ADDITION_COMPOSITION_PATH_MODE}" \
      "PYTHONUNBUFFERED=1"
  )"

  local job_id
  if [[ "${DRY_RUN}" == "1" ]]; then
    job_id="DRYRUN-${baseline}"
    echo "[DRYRUN] baseline=${baseline}"
    echo "[DRYRUN] ${cmd}"
  else
    job_id="$(
      self_submit_wrapped_resource_job \
        "DRYRUN-${baseline}" \
        "${job_name}" \
        "${LOG_DIR}/%x-%j.out" \
        "${LOG_DIR}/%x-%j.err" \
        "${cmd}"
    )"
  fi

  printf "%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" \
    "${baseline}" \
    "${ADDITION_COMPOSITION_PATH_MODE}" \
    "${out_dir}" \
    "${results_path}" >> "${MANIFEST}"

  echo "[INFO] job_id=${job_id} baseline=${baseline}"
  echo "[INFO] output=${out_dir}"
  echo "[INFO] results=${results_path}"
}

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Run root" "${RUN_ROOT}" \
  "Seed model" "${SEED_MODEL}" \
  "Launcher" "${FULLPACK_LAUNCHER}" \
  "Logs" "${LOG_DIR}" \
  "Manifest" "${MANIFEST}" \
  "Slurm" "partition=${SBATCH_PARTITION} gres=${SBATCH_GRES} cpus=${SBATCH_CPUS} mem=${SBATCH_MEM} time=${SBATCH_TIME}" \
  "Schedule" "exact_digits composition_path=${ADDITION_COMPOSITION_PATH_MODE} rounds=${NUM_EXPAND_ROUNDS} expand_num_digits=${EXPAND_NUM_DIGITS} seed_replay_train_per_digit=${SEED_REPLAY_TRAIN_PER_DIGIT} expand_train_per_digit=${EXPAND_TRAIN_PER_DIGIT}"

for baseline in "${BASELINES[@]}"; do
  submit_baseline "${baseline}"
done

echo "[INFO] Submitted exact-digit fixed-binary addition pack."
echo "[INFO] Manifest: ${MANIFEST}"

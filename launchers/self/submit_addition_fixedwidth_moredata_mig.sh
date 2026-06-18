#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/addition_fixedwidth_moredata_sweep_${TS}}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"
MANIFEST="${MANIFEST:-${RUN_ROOT}/stage1_manifest.tsv}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/addition_fixedwidth_mixed_seed_best}"
FULLPACK_LAUNCHER="${FULLPACK_LAUNCHER:-${ROOT_DIR}/launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh}"

self_set_sbatch_defaults "mig" "gpu:1g.10gb:1" "1" "64G" "48:00:00"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-5000}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-8}"
EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS:-2}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${LOG_DIR}"
printf "job_id\tschedule_id\tcomposition_path_mode\texpand_train_per_digit\tmax_steps\tself_improve_warmup_steps\tself_improve_stable_steps\tself_improve_decay_steps\tout_root\tresults_path\n" > "${MANIFEST}"

q() {
  printf "%q" "$1"
}

submit_candidate() {
  local path_mode="$1"
  local expand_train_per_digit="$2"
  local max_steps="$3"
  local stable_steps="$4"
  local decay_steps="$5"
  local warmup_steps=0
  local schedule_id="${path_mode}_expand${expand_train_per_digit}_steps${max_steps}"
  local candidate_root="${RUN_ROOT}/stage1/${schedule_id}"
  local results_path="${candidate_root}/with_carry_filtered/self_improvement_results.json"
  local job_name="add-fw-md-${path_mode:0:2}-${expand_train_per_digit}"
  local extra_args="--max-steps ${max_steps} --self-improve-warmup-steps ${warmup_steps} --self-improve-stable-steps ${stable_steps} --self-improve-decay-steps ${decay_steps}"
  local cmd
  cmd="cd $(q "${ROOT_DIR}") && "
  cmd+="OUT_ROOT=$(q "${candidate_root}") "
  cmd+="SEED_MODEL=$(q "${SEED_MODEL}") "
  cmd+="BASELINE=with_carry_filtered "
  cmd+="TRAIN_BATCH_SIZE=$(q "${TRAIN_BATCH_SIZE}") "
  cmd+="EVAL_BATCH_SIZE=$(q "${EVAL_BATCH_SIZE}") "
  cmd+="SEED_REPLAY_TRAIN_PER_DIGIT=$(q "${SEED_REPLAY_TRAIN_PER_DIGIT}") "
  cmd+="EXPAND_TRAIN_PER_DIGIT=$(q "${expand_train_per_digit}") "
  cmd+="NUM_EXPAND_ROUNDS=$(q "${NUM_EXPAND_ROUNDS}") "
  cmd+="EXPAND_NUM_DIGITS=$(q "${EXPAND_NUM_DIGITS}") "
  cmd+="ADDITION_WIDTH_MODE=fixed_width_mixed_prompt "
  cmd+="ADDITION_SAMPLING_MODE=balanced_visible_lengths "
  cmd+="ADDITION_COMPOSITION_PATH_MODE=$(q "${path_mode}") "
  cmd+="INCLUDE_COMPOSE_CORRUPT=0 "
  cmd+="PYTHONUNBUFFERED=1 "
  cmd+="EXTRA_ARGS=$(q "${extra_args}") "
  cmd+="bash $(q "${FULLPACK_LAUNCHER}")"

  local job_id
  if [[ "${DRY_RUN}" == "1" ]]; then
    job_id="DRYRUN-${schedule_id}"
    echo "[DRYRUN] ${schedule_id}"
    echo "[DRYRUN] extra_args=${extra_args}"
    echo "[DRYRUN] ${cmd}"
  else
    local -a sbatch_cmd=(
      sbatch
      --parsable
      --job-name "${job_name}"
      --output "${LOG_DIR}/%x-%j.out"
      --error "${LOG_DIR}/%x-%j.err"
    )
    self_add_sbatch_resources sbatch_cmd
    sbatch_cmd+=(--wrap "${cmd}")
    self_print_command "${sbatch_cmd[@]}"
    job_id="$("${sbatch_cmd[@]}")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" \
    "${schedule_id}" \
    "${path_mode}" \
    "${expand_train_per_digit}" \
    "${max_steps}" \
    "${warmup_steps}" \
    "${stable_steps}" \
    "${decay_steps}" \
    "${candidate_root}" \
    "${results_path}" >> "${MANIFEST}"

  echo "[INFO] job_id=${job_id} schedule=${schedule_id}"
  echo "[INFO] output_root=${candidate_root}"
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
  "Fixed knobs" "baseline=with_carry_filtered width_mode=fixed_width_mixed_prompt sampling=balanced_visible_lengths rounds=${NUM_EXPAND_ROUNDS} expand_num_digits=${EXPAND_NUM_DIGITS} seed_replay_train_per_digit=${SEED_REPLAY_TRAIN_PER_DIGIT}"

submit_candidate fixed_binary 20000 4500 3500 1000
submit_candidate fixed_binary 40000 6000 5000 1000
submit_candidate random 20000 4500 3500 1000
submit_candidate random 40000 6000 5000 1000

echo "[INFO] Stage 1 fixed-width more-data sweep submitted."
echo "[INFO] Manifest: ${MANIFEST}"

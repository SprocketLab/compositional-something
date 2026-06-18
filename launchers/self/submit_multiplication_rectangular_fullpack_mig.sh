#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

LAUNCHER="${ROOT_DIR}/launchers/self/run_multiplication_rectangular_self_improvement_mig.sbatch"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/models/multiplication_rectangular_seed_best}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_self_improvement_pack_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
BASELINES=(${BASELINES:-short_only direct compose compose_corrupt})
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED_REPLAY_TRAIN_PER_PARTITION="${SEED_REPLAY_TRAIN_PER_PARTITION:-2000}"
EXPAND_TRAIN_PER_PARTITION="${EXPAND_TRAIN_PER_PARTITION:-2000}"
FRONTIER_ROW_PROFILE="${FRONTIER_ROW_PROFILE:-uniform}"
INITIAL_MAX_B_DIGITS="${INITIAL_MAX_B_DIGITS:-8}"
EXPAND_B_DIGITS="${EXPAND_B_DIGITS:-2}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-4}"
HELDOUT_PER_PARTITION="${HELDOUT_PER_PARTITION:-200}"
LEARNING_RATE="${LEARNING_RATE:-}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_MODEL="${SAVE_MODEL:-1}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

if [[ "${DRY_RUN}" != "1" && ! -e "${SEED_MODEL}" ]]; then
  echo "[ERROR] Seed model path does not exist: ${SEED_MODEL}" >&2
  exit 1
fi

if [[ -e "${SEED_MODEL}" ]]; then
  echo "[INFO] Seed model resolved to: $(readlink -f "${SEED_MODEL}")"
else
  echo "[INFO] Seed model will be checked at runtime: ${SEED_MODEL}"
fi

for baseline in "${BASELINES[@]}"; do
  baseline_slug="${baseline//_/-}"
  job_name="mult-rect-si-${baseline_slug}"
  log_stem="${LOG_DIR}/${job_name}-%j"
  job_id="$(
    self_submit_sbatch_script \
      "dryrun" \
      "${job_name}" \
      "${log_stem}.out" \
      "${log_stem}.err" \
      "ALL,OUT_ROOT=${OUT_ROOT},BASELINE=${baseline},SEED_MODEL=${SEED_MODEL},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},SEED_REPLAY_TRAIN_PER_PARTITION=${SEED_REPLAY_TRAIN_PER_PARTITION},EXPAND_TRAIN_PER_PARTITION=${EXPAND_TRAIN_PER_PARTITION},FRONTIER_ROW_PROFILE=${FRONTIER_ROW_PROFILE},INITIAL_MAX_B_DIGITS=${INITIAL_MAX_B_DIGITS},EXPAND_B_DIGITS=${EXPAND_B_DIGITS},NUM_EXPAND_ROUNDS=${NUM_EXPAND_ROUNDS},HELDOUT_PER_PARTITION=${HELDOUT_PER_PARTITION},LEARNING_RATE=${LEARNING_RATE},MAX_STEPS=${MAX_STEPS},SAVE_MODEL=${SAVE_MODEL},SEED=${SEED},DRY_RUN=${DRY_RUN}" \
      "${LAUNCHER}"
  )"
  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] baseline=${baseline} job_id=${job_id} output=${OUT_ROOT}/${baseline} log=${log_stem}.out"
  else
    echo "[INFO] baseline=${baseline} job_id=${job_id} output=${OUT_ROOT}/${baseline} log=${log_stem}.out"
  fi
done

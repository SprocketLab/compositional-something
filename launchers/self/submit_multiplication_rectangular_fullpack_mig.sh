#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/multiplication_rectangular_self_improvement_pack_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"

MULT_RECT_FULLPACK_DEFAULT_CONFIG="${SCRIPT_DIR}/config/multiplication_rectangular_fullpack.env"
self_source_config_file "${MULT_RECT_FULLPACK_DEFAULT_CONFIG}" "multiplication rectangular fullpack default config"
if [[ -n "${MULT_RECT_FULLPACK_CONFIG:-}" ]]; then
  self_source_config_file "${MULT_RECT_FULLPACK_CONFIG}" "multiplication rectangular fullpack override config"
fi

read -r -a BASELINES <<< "${MULT_RECT_FULLPACK_BASELINES_RAW}"
if (( ${#BASELINES[@]} == 0 )); then
  echo "[ERROR] MULT_RECT_FULLPACK_BASELINES_RAW must contain at least one baseline." >&2
  exit 2
fi

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
      "$(self_sbatch_export_all \
        "OUT_ROOT=${OUT_ROOT}" \
        "BASELINE=${baseline}" \
        "SEED_MODEL=${SEED_MODEL}" \
        "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \
        "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}" \
        "SEED_REPLAY_TRAIN_PER_PARTITION=${SEED_REPLAY_TRAIN_PER_PARTITION}" \
        "EXPAND_TRAIN_PER_PARTITION=${EXPAND_TRAIN_PER_PARTITION}" \
        "FRONTIER_ROW_PROFILE=${FRONTIER_ROW_PROFILE}" \
        "INITIAL_MAX_B_DIGITS=${INITIAL_MAX_B_DIGITS}" \
        "EXPAND_B_DIGITS=${EXPAND_B_DIGITS}" \
        "NUM_EXPAND_ROUNDS=${NUM_EXPAND_ROUNDS}" \
        "HELDOUT_PER_PARTITION=${HELDOUT_PER_PARTITION}" \
        "LEARNING_RATE=${LEARNING_RATE}" \
        "MAX_STEPS=${MAX_STEPS}" \
        "SAVE_MODEL=${SAVE_MODEL}" \
        "SEED=${SEED}" \
        "DRY_RUN=${DRY_RUN}")" \
      "${LAUNCHER}"
  )"
  if self_parse_bool "${DRY_RUN}"; then
    echo "[INFO] baseline=${baseline} job_id=${job_id} output=${OUT_ROOT}/${baseline} log=${log_stem}.out"
  else
    echo "[INFO] baseline=${baseline} job_id=${job_id} output=${OUT_ROOT}/${baseline} log=${log_stem}.out"
  fi
done

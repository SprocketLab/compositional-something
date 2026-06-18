#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/self/run_addition_fullpack_filtered.sbatch}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/self_improvement_addition_fullpack_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"

BASELINES=(
  short_only
  direct
  with_carry
  with_carry_filtered
  compose_corrupt
)

self_print_context \
  "Job script" "${JOB_SCRIPT}" \
  "Output root" "${OUT_ROOT}" \
  "Log dir" "${LOG_DIR}" \
  "Dry run" "${DRY_RUN}"

for baseline in "${BASELINES[@]}"; do
  out_dir="${OUT_ROOT}/addition/${baseline}"
  baseline_slug="${baseline//_/-}"
  job_name="addition-fullpack-${baseline_slug}"
  log_stem="${LOG_DIR}/${job_name}-%j"
  if self_parse_bool "${DRY_RUN}"; then
    echo "[DRY_RUN] baseline=${baseline} out_dir=${out_dir}"
  fi

  job_id="$(
    self_submit_sbatch_script \
      "dryrun-${job_name}" \
      "${job_name}" \
      "${log_stem}.out" \
      "${log_stem}.err" \
      "ALL,BASELINE=${baseline},OUT_ROOT=${OUT_ROOT}" \
      "${JOB_SCRIPT}"
  )"
  echo "[INFO] Submitted baseline=${baseline} job_id=${job_id} out_dir=${out_dir}"
done

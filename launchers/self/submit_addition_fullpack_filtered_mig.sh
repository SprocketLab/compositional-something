#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

JOB_SCRIPT="${JOB_SCRIPT:-${ROOT_DIR}/launchers/self/run_addition_fullpack_filtered.sbatch}"
DRY_RUN="${DRY_RUN:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/self_improvement_addition_fullpack_${TS}}"

BASELINES=(
  short_only
  direct
  with_carry
  with_carry_filtered
  compose_corrupt
)

echo "[INFO] Job script: ${JOB_SCRIPT}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Dry run: ${DRY_RUN}"

for baseline in "${BASELINES[@]}"; do
  out_dir="${OUT_ROOT}/addition/${baseline}"
  if self_parse_bool "${DRY_RUN}"; then
    echo "[DRY_RUN] baseline=${baseline} out_dir=${out_dir}"
    echo "  BASELINE=${baseline} OUT_ROOT=${OUT_ROOT} sbatch ${JOB_SCRIPT}"
    continue
  fi

  job_id="$(BASELINE="${baseline}" OUT_ROOT="${OUT_ROOT}" sbatch --parsable "${JOB_SCRIPT}")"
  echo "[INFO] Submitted baseline=${baseline} job_id=${job_id} out_dir=${out_dir}"
done

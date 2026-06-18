#!/usr/bin/env bash
#SBATCH --job-name=adaptive-candidate-worker
#SBATCH --output=artifacts/logs/adaptive-candidate-worker-%A_%a.out
#SBATCH --error=artifacts/logs/adaptive-candidate-worker-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"

: "${CANDIDATE_SPEC_DIR:?CANDIDATE_SPEC_DIR must point to candidate worker specs}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set by sbatch --array}"

adaptive_cd_repo_root
adaptive_setup_runtime_env
adaptive_resolve_python

SPEC_PATH="${CANDIDATE_SPEC_DIR}/candidate_${SLURM_ARRAY_TASK_ID}.json"
if [[ ! -f "${SPEC_PATH}" ]]; then
  echo "[ERROR] Missing candidate worker spec: ${SPEC_PATH}" >&2
  exit 2
fi

echo "[INFO] Candidate worker spec: ${SPEC_PATH}"
echo "[INFO] Array task ID: ${SLURM_ARRAY_TASK_ID}"
adaptive_print_worker_context "Adaptive candidate worker"

"${PYTHON_BIN}" -m self.core.driver \
  --run-candidate-worker \
  --candidate-worker-spec "${SPEC_PATH}"

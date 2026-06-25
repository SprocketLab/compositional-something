#!/usr/bin/env bash
#SBATCH --job-name=adaptive-candidate-controller-worker
#SBATCH --output=artifacts/logs/adaptive-candidate-controller-worker-%j.out
#SBATCH --error=artifacts/logs/adaptive-candidate-controller-worker-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launchers/self/lib/adaptive_common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/launchers/self"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/lib/adaptive_common.sh"

: "${CONTROLLER_WORKER_SPEC:?CONTROLLER_WORKER_SPEC must point to a controller worker spec}"

adaptive_cd_repo_root
adaptive_setup_runtime_env
adaptive_resolve_python

if [[ ! -f "${CONTROLLER_WORKER_SPEC}" ]]; then
  echo "[ERROR] Missing controller worker spec: ${CONTROLLER_WORKER_SPEC}" >&2
  exit 2
fi

echo "[INFO] Controller worker spec: ${CONTROLLER_WORKER_SPEC}"
adaptive_print_worker_context "Adaptive controller worker"

"${PYTHON_BIN}" -m self.adaptive.driver \
  --run-controller-worker \
  --controller-worker-spec "${CONTROLLER_WORKER_SPEC}"

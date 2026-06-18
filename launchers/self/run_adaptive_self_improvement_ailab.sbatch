#!/usr/bin/env bash
#SBATCH --job-name=adaptive-selfimp-pilot
#SBATCH --output=artifacts/logs/adaptive-selfimp-pilot-%j.out
#SBATCH --error=artifacts/logs/adaptive-selfimp-pilot-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root
adaptive_setup_runtime_env
adaptive_resolve_python

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/adaptive_self_improvement_ailab_pilot_${TS}}"
RUN_TESTS="${RUN_TESTS:-1}"

CONFIG_FIXTURE="${CONFIG_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_run_length_config_fixture.jsonl}"
PROGRAM_FIXTURE="${PROGRAM_FIXTURE:-${ROOT_DIR}/tests/fixtures/adaptive_run_length_program_fixture.jsonl}"
AGGREGATE_METRICS_JSON="${AGGREGATE_METRICS_JSON:-{\"frontier_accuracy\":0.42,\"final_accuracy\":0.90,\"slice\":\"ailab_pilot\"}}"

echo "[INFO] Output root: ${OUT_ROOT}"
adaptive_print_worker_context "Adaptive self-improvement pilot"
"${PYTHON_BIN}" -c "import torch; print(f'[INFO] torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

echo "[INFO] Compile check"
"${PYTHON_BIN}" -m py_compile \
  self/self_improvement_tasks.py \
  self/legacy/run_length_bit_cli.py \
  self/seed_fit_experiment.py \
  self/core/proposals.py \
  self/core/program_sandbox.py \
  self/adaptive_self_improvement.py

if [[ "${RUN_TESTS}" == "1" ]]; then
  echo "[INFO] Focused tests"
  "${PYTHON_BIN}" -m pytest \
    tests/test_adaptive_proposals_and_sandbox.py \
    tests/test_adaptive_self_improvement_controller.py \
    tests/test_self_improvement_tasks.py \
    -k 'adaptive or run_state or symbol_run_pair_target_parser or run_length_symbol_pair_guarded_compose_uses_left_tie' \
    -q
fi

echo "[INFO] Run-length config proposal pilot"
"${PYTHON_BIN}" -m self.experiments.adaptive_self_improvement \
  --task run_length \
  --condition config \
  --proposal-fixture-jsonl "${CONFIG_FIXTURE}" \
  --output-dir "${OUT_ROOT}/run_length_config" \
  --dry-run-proposals \
  --current-frontier-min 6 \
  --current-frontier-max 10 \
  --frontier-min-allowed 11 \
  --frontier-max-allowed 14 \
  --min-examples-per-size 1 \
  --max-examples-per-size 128 \
  --aggregate-metrics-json "${AGGREGATE_METRICS_JSON}"

echo "[INFO] Run-length program proposal pilot with one repair"
"${PYTHON_BIN}" -m self.experiments.adaptive_self_improvement \
  --task run_length \
  --condition program \
  --proposal-fixture-jsonl "${PROGRAM_FIXTURE}" \
  --output-dir "${OUT_ROOT}/run_length_program" \
  --dry-run-proposals \
  --current-frontier-min 6 \
  --current-frontier-max 10 \
  --frontier-min-allowed 11 \
  --frontier-max-allowed 14 \
  --min-examples-per-size 1 \
  --max-examples-per-size 128 \
  --aggregate-metrics-json "${AGGREGATE_METRICS_JSON}"

echo "[INFO] Finished adaptive proposal pilot."

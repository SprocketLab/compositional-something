#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/self/lib/self_common.sh
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root
self_resolve_python
self_set_sbatch_defaults "mig" "gpu:1g.10gb:1" "1" "64G" "48:00:00"

RUN_LENGTH_ALPHA10_BASELINE_CONFIG="${RUN_LENGTH_ALPHA10_BASELINE_CONFIG:-${ROOT_DIR}/launchers/self/config/run_length_alpha10_baseline_pack.env}"
self_source_config_file "${RUN_LENGTH_ALPHA10_BASELINE_CONFIG}" "run-length alpha10 baseline config"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/run_length_alpha10_baseline_pack_${TS}}"
SEED_MODEL="${SEED_MODEL:-${ROOT_DIR}/artifacts/runs/run_length_multisymbol_pair_alpha10_seed50k_steps15k_20260423_123229/seed/model}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EXPAND_TRAIN_PER_BIT="${EXPAND_TRAIN_PER_BIT:-2000}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-7}"
EXPAND_NUM_BITS="${EXPAND_NUM_BITS:-9}"
ROUND_WARMUP_STEPS="${ROUND_WARMUP_STEPS:-500}"
SEED="${SEED:-7}"
DRY_RUN="${DRY_RUN:-0}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"

read -r -a BASELINE_ROWS <<< "${RUN_LENGTH_ALPHA10_BASELINE_ROWS_RAW}"
if (( ${#BASELINE_ROWS[@]} == 0 )); then
  echo "[ERROR] Run-length alpha10 baseline rows cannot be empty." >&2
  exit 2
fi
MANIFEST="${OUT_ROOT}/manifest.tsv"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"
printf "baseline\tjob_id\toutput_dir\tresults_path\n" > "${MANIFEST}"

self_print_context \
  "Root dir" "${ROOT_DIR}" \
  "Python" "${PYTHON_BIN}" \
  "Output root" "${OUT_ROOT}" \
  "Seed model" "${SEED_MODEL}" \
  "Config" "${RUN_LENGTH_ALPHA10_BASELINE_CONFIG}" \
  "Baseline rows" "${BASELINE_ROWS[*]}" \
  "Num rounds" "${NUM_EXPAND_ROUNDS}" \
  "Expand num bits" "${EXPAND_NUM_BITS}" \
  "Expand train per bit" "${EXPAND_TRAIN_PER_BIT}" \
  "Round warmup steps" "${ROUND_WARMUP_STEPS}" \
  "Trainer seed" "${SEED}" \
  "Log dir" "${LOG_DIR}" \
  "Dry run" "${DRY_RUN}"

for baseline_row in "${BASELINE_ROWS[@]}"; do
  IFS=: read -r baseline pseudo_label_mode guarded_compose_rule extra_field <<< "${baseline_row}"
  if [[ -z "${baseline}" || -z "${pseudo_label_mode}" || -z "${guarded_compose_rule}" || -n "${extra_field:-}" ]]; then
    echo "[ERROR] Invalid run-length alpha10 baseline row: ${baseline_row}" >&2
    exit 2
  fi
  out_dir="${OUT_ROOT}/${baseline}"
  results_path="${out_dir}/self_improvement_results.json"
  run_cmd=(
    "${PYTHON_BIN}" "-m" "self.legacy.run_length_self_improvement"
    "--output-dir" "${out_dir}"
    "--model-name" "${SEED_MODEL}"
    "--format-version" "legacy"
    "--target-mode" "symbol_run_pair"
    "--compose-arity" "exact2"
    "--bit-composition-path-mode" "random"
    "--recipe" "algorithmic_self_improve_v1"
    "--treat-seed-as-round-zero"
    "--symbol-alphabet-size" "10"
    "--initial-min-bits" "6"
    "--initial-max-bits" "10"
    "--initial-train-per-bit" "50000"
    "--initial-eval-per-bit" "100"
    "--frontier-min-bits" "12"
    "--num-expand-rounds" "${NUM_EXPAND_ROUNDS}"
    "--expand-num-bits" "${EXPAND_NUM_BITS}"
    "--expand-train-per-bit" "${EXPAND_TRAIN_PER_BIT}"
    "--eval-per-bit" "100"
    "--composed-eval-per-bit" "100"
    "--pseudo-label-mode" "${pseudo_label_mode}"
    "--guarded-compose-rule" "${guarded_compose_rule}"
    "--bucket-train-batches-by-bits"
    "--bf16"
    "--per-device-train-batch-size" "${TRAIN_BATCH_SIZE}"
    "--per-device-eval-batch-size" "${EVAL_BATCH_SIZE}"
    "--seed" "${SEED}"
    "--save-model-policy" "all_rounds"
    "--self-improve-warmup-steps" "${ROUND_WARMUP_STEPS}"
    "--resume"
  )
  wrapped_cmd="$(self_wrap_repo_command "${run_cmd[@]}")"
  log_stem="${LOG_DIR}/rl-a10-baseline-${baseline}-%j"
  helper_job_id="$(
    self_submit_wrapped_job \
      "rl-a10-${baseline}" \
      "${log_stem}.out" \
      "${log_stem}.err" \
      "${SBATCH_PARTITION}" \
      "${SBATCH_GRES}" \
      "${SBATCH_CPUS}" \
      "${SBATCH_MEM}" \
      "${SBATCH_TIME}" \
      "" \
      "${wrapped_cmd}"
  )"
  if self_parse_bool "${DRY_RUN}"; then
    job_id="dryrun-${baseline}"
    echo "[INFO] DRY_RUN=1; sbatch not executed for ${baseline}."
  else
    job_id="${helper_job_id}"
    echo "Submitted batch job ${job_id}"
  fi
  printf "%s\t%s\t%s\t%s\n" "${baseline}" "${job_id}" "${out_dir}" "${results_path}" >> "${MANIFEST}"
  echo "[INFO] baseline=${baseline} job_id=${job_id} output_dir=${out_dir}"
done

echo "[INFO] Manifest: ${MANIFEST}"

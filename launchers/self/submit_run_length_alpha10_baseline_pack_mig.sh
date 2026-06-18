#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=launchers/self/lib/self_common.sh
source "${SCRIPT_DIR}/lib/self_common.sh"
self_cd_repo_root
self_resolve_python
self_set_sbatch_defaults "mig" "gpu:1g.10gb:1" "1" "64G" "48:00:00"

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

declare -A MODE_ARGS
MODE_ARGS[direct]="--pseudo-label-mode direct --guarded-compose-rule none"
MODE_ARGS[unfiltered_compose]="--pseudo-label-mode compose --guarded-compose-rule run_length_unfiltered_pair"
MODE_ARGS[guarded_compose]="--pseudo-label-mode compose --guarded-compose-rule run_length_no_boundary_continue"

BASELINES=(direct unfiltered_compose guarded_compose)
MANIFEST="${OUT_ROOT}/manifest.tsv"

mkdir -p "${OUT_ROOT}"
printf "baseline\tjob_id\toutput_dir\tresults_path\n" > "${MANIFEST}"

echo "[INFO] Root dir: ${ROOT_DIR}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Output root: ${OUT_ROOT}"
echo "[INFO] Seed model: ${SEED_MODEL}"
echo "[INFO] Num rounds: ${NUM_EXPAND_ROUNDS}"
echo "[INFO] Expand num bits: ${EXPAND_NUM_BITS}"
echo "[INFO] Expand train per bit: ${EXPAND_TRAIN_PER_BIT}"
echo "[INFO] Round warmup steps: ${ROUND_WARMUP_STEPS}"
echo "[INFO] Trainer seed: ${SEED}"
echo "[INFO] Dry run: ${DRY_RUN}"

for baseline in "${BASELINES[@]}"; do
  out_dir="${OUT_ROOT}/${baseline}"
  results_path="${out_dir}/self_improvement_results.json"
  read -r -a baseline_args <<< "${MODE_ARGS[${baseline}]}"
  run_cmd=(
    "${PYTHON_BIN}" "-m" "self.run_length_self_improvement"
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
    "${baseline_args[@]}"
    "--bucket-train-batches-by-bits"
    "--bf16"
    "--per-device-train-batch-size" "${TRAIN_BATCH_SIZE}"
    "--per-device-eval-batch-size" "${EVAL_BATCH_SIZE}"
    "--seed" "${SEED}"
    "--save-model-policy" "all_rounds"
    "--self-improve-warmup-steps" "${ROUND_WARMUP_STEPS}"
    "--resume"
  )
  printf -v quoted_root "%q" "${ROOT_DIR}"
  printf -v quoted_run_cmd " %q" "${run_cmd[@]}"
  wrapped_cmd="cd ${quoted_root} && PYTHONPATH=.${quoted_run_cmd}"
  log_stem="${ROOT_DIR}/artifacts/logs/rl-a10-baseline-${baseline}-%j"
  sbatch_cmd=(
    sbatch
    --job-name "rl-a10-${baseline}"
    --output "${log_stem}.out"
    --error "${log_stem}.err"
  )
  self_add_sbatch_resources sbatch_cmd
  sbatch_cmd+=(--wrap "${wrapped_cmd}")
  self_print_prefixed_command_stdout "Submit command" "${sbatch_cmd[@]}"
  if self_parse_bool "${DRY_RUN}"; then
    job_id="dryrun-${baseline}"
    echo "[INFO] DRY_RUN=1; sbatch not executed for ${baseline}."
  else
    submit_output="$("${sbatch_cmd[@]}")"
    echo "${submit_output}"
    job_id="$(awk '{print $4}' <<< "${submit_output}")"
  fi
  printf "%s\t%s\t%s\t%s\n" "${baseline}" "${job_id}" "${out_dir}" "${results_path}" >> "${MANIFEST}"
  echo "[INFO] baseline=${baseline} job_id=${job_id} output_dir=${out_dir}"
done

echo "[INFO] Manifest: ${MANIFEST}"

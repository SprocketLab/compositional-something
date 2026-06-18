#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/self_common.sh"

self_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/artifacts/runs/addition_fixedwidth_mixed_${TS}}"
SEED_OUT_ROOT="${RUN_ROOT}/seed"
FULLPACK_OUT_ROOT="${RUN_ROOT}/fullpack"
SEED_MODEL="${ROOT_DIR}/artifacts/models/addition_fixedwidth_mixed_seed_best"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SEED="${SEED:-0}"
TRAIN_PER_DIGIT="${TRAIN_PER_DIGIT:-50000}"
MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-5e-4}"
NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS:-8}"
EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS:-2}"
SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT:-5000}"
EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT:-10000}"
ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE:-balanced_visible_lengths}"
DRY_RUN="${DRY_RUN:-0}"

SEED_LAUNCHER="${ROOT_DIR}/launchers/self/run_addition_fixedwidth_mixed_seed_mig.sbatch"
FULLPACK_LAUNCHER="${ROOT_DIR}/launchers/self/run_addition_fixedwidth_mixed_recipe_fullpack.sh"
BASELINES=(short_only direct with_carry with_carry_filtered)
ORIGINAL_COMPOSITION_BASELINES=(direct with_carry with_carry_filtered)

self_print_context \
  "Run root" "${RUN_ROOT}" \
  "Seed output" "${SEED_OUT_ROOT}" \
  "Fullpack output" "${FULLPACK_OUT_ROOT}" \
  "Stable seed model" "${SEED_MODEL}" \
  "Addition sampling mode" "${ADDITION_SAMPLING_MODE}" \
  "Dry run" "${DRY_RUN}"

mkdir -p "${RUN_ROOT}"

if self_parse_bool "${DRY_RUN}"; then
  echo
  echo "[INFO] Seed dry-run:"
  DRY_RUN=1 \
  OUT_ROOT="${SEED_OUT_ROOT}" \
  TRAIN_PER_DIGIT="${TRAIN_PER_DIGIT}" \
  MAX_STEPS="${MAX_STEPS}" \
  LR="${LR}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE}" \
  SEED="${SEED}" \
    bash "${SEED_LAUNCHER}"

  echo
  echo "[INFO] Fullpack dry-run:"
  DRY_RUN=1 \
  OUT_ROOT="${FULLPACK_OUT_ROOT}" \
  SEED_MODEL="${SEED_MODEL}" \
  BASELINE=with_carry_filtered \
  NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS}" \
  EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS}" \
  SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT}" \
  EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE}" \
  SEED="${SEED}" \
    bash "${FULLPACK_LAUNCHER}"

  echo
  echo "[INFO] Original-composition dry-run:"
  DRY_RUN=1 \
  OUT_ROOT="${RUN_ROOT}/fullpack_original_composition" \
  SEED_MODEL="${SEED_MODEL}" \
  BASELINE=with_carry_filtered \
  NUM_EXPAND_ROUNDS="${NUM_EXPAND_ROUNDS}" \
  EXPAND_NUM_DIGITS="${EXPAND_NUM_DIGITS}" \
  SEED_REPLAY_TRAIN_PER_DIGIT="${SEED_REPLAY_TRAIN_PER_DIGIT}" \
  EXPAND_TRAIN_PER_DIGIT="${EXPAND_TRAIN_PER_DIGIT}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  ADDITION_SAMPLING_MODE="${ADDITION_SAMPLING_MODE}" \
  ADDITION_COMPOSITION_PATH_MODE=random \
  SEED="${SEED}" \
    bash "${FULLPACK_LAUNCHER}"
  exit 0
fi

seed_job="$(
  self_submit_sbatch_script \
    "dryrun-add-fw-seed" \
    "add-fw-seed" \
    "artifacts/logs/add-fw-seed-%j.out" \
    "artifacts/logs/add-fw-seed-%j.err" \
    "ALL,OUT_ROOT=${SEED_OUT_ROOT},TRAIN_PER_DIGIT=${TRAIN_PER_DIGIT},MAX_STEPS=${MAX_STEPS},LR=${LR},TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE},EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE},ADDITION_SAMPLING_MODE=${ADDITION_SAMPLING_MODE},SEED=${SEED},SAVE_MODEL=1,PYTHONUNBUFFERED=1" \
    "${SEED_LAUNCHER}"
)"

echo
echo "[INFO] Submitted seed job ${seed_job}"
echo "[INFO]   output dir: ${SEED_OUT_ROOT}"
echo "[INFO]   logs: artifacts/logs/add-fw-seed-${seed_job}.out / artifacts/logs/add-fw-seed-${seed_job}.err"

for baseline in "${BASELINES[@]}"; do
  wrapped_cmd="$(
    self_wrap_env_command \
      "${FULLPACK_LAUNCHER}" \
      "OUT_ROOT=${FULLPACK_OUT_ROOT}" \
      "SEED_MODEL=${SEED_MODEL}" \
      "BASELINE=${baseline}" \
      "NUM_EXPAND_ROUNDS=${NUM_EXPAND_ROUNDS}" \
      "EXPAND_NUM_DIGITS=${EXPAND_NUM_DIGITS}" \
      "SEED_REPLAY_TRAIN_PER_DIGIT=${SEED_REPLAY_TRAIN_PER_DIGIT}" \
      "EXPAND_TRAIN_PER_DIGIT=${EXPAND_TRAIN_PER_DIGIT}" \
      "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \
      "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}" \
      "ADDITION_SAMPLING_MODE=${ADDITION_SAMPLING_MODE}" \
      "ADDITION_COMPOSITION_PATH_MODE=fixed_binary" \
      "SEED=${SEED}" \
      "PYTHONUNBUFFERED=1"
  )"
  job="$(
    self_submit_wrapped_job \
      "add-fw-${baseline}" \
      "artifacts/logs/add-fw-${baseline}-%j.out" \
      "artifacts/logs/add-fw-${baseline}-%j.err" \
      "mig" \
      "gpu:1g.10gb:1" \
      "1" \
      "64G" \
      "48:00:00" \
      "afterok:${seed_job}" \
      "${wrapped_cmd}"
  )"
  echo
  echo "[INFO] Submitted baseline job ${job}: ${baseline}"
  echo "[INFO]   depends on seed job: ${seed_job}"
  echo "[INFO]   output dir: ${FULLPACK_OUT_ROOT}/${baseline}"
  echo "[INFO]   logs: artifacts/logs/add-fw-${baseline}-${job}.out / artifacts/logs/add-fw-${baseline}-${job}.err"
done

ORIGINAL_FULLPACK_OUT_ROOT="${RUN_ROOT}/fullpack_original_composition"
for baseline in "${ORIGINAL_COMPOSITION_BASELINES[@]}"; do
  wrapped_cmd="$(
    self_wrap_env_command \
      "${FULLPACK_LAUNCHER}" \
      "OUT_ROOT=${ORIGINAL_FULLPACK_OUT_ROOT}" \
      "SEED_MODEL=${SEED_MODEL}" \
      "BASELINE=${baseline}" \
      "NUM_EXPAND_ROUNDS=${NUM_EXPAND_ROUNDS}" \
      "EXPAND_NUM_DIGITS=${EXPAND_NUM_DIGITS}" \
      "SEED_REPLAY_TRAIN_PER_DIGIT=${SEED_REPLAY_TRAIN_PER_DIGIT}" \
      "EXPAND_TRAIN_PER_DIGIT=${EXPAND_TRAIN_PER_DIGIT}" \
      "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \
      "EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}" \
      "ADDITION_SAMPLING_MODE=${ADDITION_SAMPLING_MODE}" \
      "ADDITION_COMPOSITION_PATH_MODE=random" \
      "SEED=${SEED}" \
      "PYTHONUNBUFFERED=1"
  )"
  job="$(
    self_submit_wrapped_job \
      "add-fw-rand-${baseline}" \
      "artifacts/logs/add-fw-rand-${baseline}-%j.out" \
      "artifacts/logs/add-fw-rand-${baseline}-%j.err" \
      "mig" \
      "gpu:1g.10gb:1" \
      "1" \
      "64G" \
      "48:00:00" \
      "afterok:${seed_job}" \
      "${wrapped_cmd}"
  )"
  echo
  echo "[INFO] Submitted original-composition baseline job ${job}: ${baseline}"
  echo "[INFO]   depends on seed job: ${seed_job}"
  echo "[INFO]   output dir: ${ORIGINAL_FULLPACK_OUT_ROOT}/${baseline}"
  echo "[INFO]   logs: artifacts/logs/add-fw-rand-${baseline}-${job}.out / artifacts/logs/add-fw-rand-${baseline}-${job}.err"
done

cat > "${RUN_ROOT}/submission_manifest.txt" <<EOF
run_root=${RUN_ROOT}
seed_job=${seed_job}
seed_output=${SEED_OUT_ROOT}
fullpack_output=${FULLPACK_OUT_ROOT}
original_composition_output=${ORIGINAL_FULLPACK_OUT_ROOT}
seed_model=${SEED_MODEL}
baselines=${BASELINES[*]}
original_composition_baselines=${ORIGINAL_COMPOSITION_BASELINES[*]}
addition_sampling_mode=${ADDITION_SAMPLING_MODE}
num_expand_rounds=${NUM_EXPAND_ROUNDS}
expand_num_digits=${EXPAND_NUM_DIGITS}
seed_replay_train_per_digit=${SEED_REPLAY_TRAIN_PER_DIGIT}
expand_train_per_digit=${EXPAND_TRAIN_PER_DIGIT}
EOF

echo
echo "[INFO] Wrote manifest: ${RUN_ROOT}/submission_manifest.txt"

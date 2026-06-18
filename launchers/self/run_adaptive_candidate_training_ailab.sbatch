#!/usr/bin/env bash
#SBATCH --job-name=adaptive-candidate
#SBATCH --output=artifacts/logs/adaptive-candidate-%j.out
#SBATCH --error=artifacts/logs/adaptive-candidate-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=ailab
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root

: "${TASK:?TASK must be set to addition or run_length}"
: "${OUT_DIR:?OUT_DIR must point to the adaptive candidate output directory}"

mkdir -p "${OUT_DIR}"
adaptive_setup_runtime_env
adaptive_resolve_python
adaptive_source_config_files

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PROPOSAL_MODEL_NAME="${PROPOSAL_MODEL_NAME:-current}"
CONDITION="${CONDITION:-config}"
PROPOSAL_OUTPUT_SCHEMA="${PROPOSAL_OUTPUT_SCHEMA:-action_prediction}"
NUM_ROUNDS="${NUM_ROUNDS:-100}"
MAX_ATTEMPT_ROUNDS="${MAX_ATTEMPT_ROUNDS:-$((NUM_ROUNDS * 10))}"
NO_SELECTION_PATIENCE="${NO_SELECTION_PATIENCE:-${MAX_ATTEMPT_ROUNDS}}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
CONTROLLER_EXECUTION_MODE="${CONTROLLER_EXECUTION_MODE:-local}"
CONTROLLER_WORKER_SBATCH_SCRIPT="${CONTROLLER_WORKER_SBATCH_SCRIPT:-${ROOT_DIR}/launchers/self/run_adaptive_controller_worker_ailab.sbatch}"
CONTROLLER_WORKER_POLL_SECONDS="${CONTROLLER_WORKER_POLL_SECONDS:-30}"
CONTROLLER_WORKER_TIME_LIMIT="${CONTROLLER_WORKER_TIME_LIMIT:-04:00:00}"
if [[ -z "${CONTROLLER_SEED_WORKER_TIME_LIMIT:-}" ]]; then
  case "${TASK}" in
    addition)
      CONTROLLER_SEED_WORKER_TIME_LIMIT="02:00:00"
      ;;
    run_length)
      CONTROLLER_SEED_WORKER_TIME_LIMIT="03:00:00"
      ;;
    *)
      CONTROLLER_SEED_WORKER_TIME_LIMIT="${CONTROLLER_WORKER_TIME_LIMIT}"
      ;;
  esac
fi
CONTROLLER_ROUND_WORKER_TIME_LIMIT="${CONTROLLER_ROUND_WORKER_TIME_LIMIT:-01:00:00}"
CONTROLLER_GRPO_WORKER_TIME_LIMIT="${CONTROLLER_GRPO_WORKER_TIME_LIMIT:-01:00:00}"
CANDIDATE_EXECUTION_MODE="${CANDIDATE_EXECUTION_MODE:-local_parallel}"
CANDIDATE_LOCAL_PARALLELISM="${CANDIDATE_LOCAL_PARALLELISM:-4}"
CANDIDATE_LOCAL_PACK_SIZE="${CANDIDATE_LOCAL_PACK_SIZE:-1}"
CANDIDATE_LOCAL_CACHE_BASE_STATE="${CANDIDATE_LOCAL_CACHE_BASE_STATE:-0}"
CANDIDATE_ARRAY_SBATCH_SCRIPT="${CANDIDATE_ARRAY_SBATCH_SCRIPT:-${ROOT_DIR}/launchers/self/run_adaptive_candidate_worker_ailab.sbatch}"
CANDIDATE_ARRAY_MAX_PARALLEL="${CANDIDATE_ARRAY_MAX_PARALLEL:-4}"
CANDIDATE_ARRAY_POLL_SECONDS="${CANDIDATE_ARRAY_POLL_SECONDS:-30}"
CANDIDATE_ARRAY_TIMEOUT_SECONDS="${CANDIDATE_ARRAY_TIMEOUT_SECONDS:-0}"
if [[ -z "${CANDIDATE_ARRAY_TIME_LIMIT:-}" ]]; then
  case "${TASK}" in
    addition)
      CANDIDATE_ARRAY_TIME_LIMIT="02:00:00"
      ;;
    run_length)
      CANDIDATE_ARRAY_TIME_LIMIT="03:00:00"
      ;;
    *)
      CANDIDATE_ARRAY_TIME_LIMIT="04:00:00"
      ;;
  esac
fi
LAMBDA_FINAL="${LAMBDA_FINAL:-0.1}"
SELECTION_MIN_REWARD="${SELECTION_MIN_REWARD:-0.0}"
PROPOSAL_TRACE_REPLAY_RATIO="${PROPOSAL_TRACE_REPLAY_RATIO:-0.0}"
PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES="${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES:-256}"
POST_TASK_PROPOSAL_REHEARSAL="${POST_TASK_PROPOSAL_REHEARSAL:-1}"
POST_TASK_PROPOSAL_REHEARSAL_REPEAT_COUNT="${POST_TASK_PROPOSAL_REHEARSAL_REPEAT_COUNT:-64}"
POST_TASK_PROPOSAL_REHEARSAL_MAX_EXAMPLES="${POST_TASK_PROPOSAL_REHEARSAL_MAX_EXAMPLES:-256}"
OUTCOME_TRACE_TARGET_MODE="${OUTCOME_TRACE_TARGET_MODE:-numeric}"
OUTCOME_TRACE_REPLAY_RATIO="${OUTCOME_TRACE_REPLAY_RATIO:-0.10}"
OUTCOME_TRACE_REPLAY_MAX_EXAMPLES="${OUTCOME_TRACE_REPLAY_MAX_EXAMPLES:-4096}"
INVALID_OUTCOME_REWARD="${INVALID_OUTCOME_REWARD:--0.1}"
if [[ -z "${PROPOSAL_GRPO_STEPS:-}" ]]; then
  if [[ "${CONDITION}" == "config" ]]; then
    PROPOSAL_GRPO_STEPS="1"
  else
    PROPOSAL_GRPO_STEPS="0"
  fi
fi
PROPOSAL_GRPO_LEARNING_RATE="${PROPOSAL_GRPO_LEARNING_RATE:-1e-6}"
PROPOSAL_GRPO_KL_COEF="${PROPOSAL_GRPO_KL_COEF:-0.01}"
PROPOSAL_GRPO_GRAD_CLIP="${PROPOSAL_GRPO_GRAD_CLIP:-1.0}"
PROPOSAL_GRPO_ZERO_VARIANCE="${PROPOSAL_GRPO_ZERO_VARIANCE:-fixed_baseline}"
PROPOSAL_GRPO_REWARD_MODE="${PROPOSAL_GRPO_REWARD_MODE:-outcome}"
PROPOSAL_GRPO_OUTCOME_SCALE="${PROPOSAL_GRPO_OUTCOME_SCALE:-0.05}"
PROPOSAL_GRPO_FIXED_BASELINE="${PROPOSAL_GRPO_FIXED_BASELINE:-0.5}"
KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS="${KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS:-0}"
INITIAL_EVAL_PER_SIZE="${INITIAL_EVAL_PER_SIZE:-50}"
EVAL_PER_SIZE="${EVAL_PER_SIZE:-100}"
COMPOSED_EVAL_PER_SIZE="${COMPOSED_EVAL_PER_SIZE:-100}"
PLAN_LOG_PATH="${PLAN_LOG_PATH:-${ROOT_DIR}/plan/260603-self-improvement-init.md}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"
REPAIR_ATTEMPTS="${REPAIR_ATTEMPTS:-1}"
PROGRAM_TIMEOUT_SECONDS="${PROGRAM_TIMEOUT_SECONDS:-1.0}"
PROGRAM_BATCH_TIMEOUT_SECONDS="${PROGRAM_BATCH_TIMEOUT_SECONDS:-30.0}"
if [[ -z "${PROPOSAL_MAX_NEW_TOKENS:-}" ]]; then
  if [[ "${CONDITION}" == "config" ]]; then
    PROPOSAL_MAX_NEW_TOKENS="256"
  else
    PROPOSAL_MAX_NEW_TOKENS="512"
  fi
fi

case "${CONDITION}" in
  config|program|policy|meta) ;;
  *)
    echo "[ERROR] Unsupported CONDITION=${CONDITION}" >&2
    exit 2
    ;;
esac

case "${CANDIDATE_EXECUTION_MODE}" in
  local_parallel|slurm_array|serial) ;;
  *)
    echo "[ERROR] Unsupported CANDIDATE_EXECUTION_MODE=${CANDIDATE_EXECUTION_MODE}" >&2
    exit 2
    ;;
esac

case "${CONTROLLER_EXECUTION_MODE}" in
  local|slurm) ;;
  *)
    echo "[ERROR] Unsupported CONTROLLER_EXECUTION_MODE=${CONTROLLER_EXECUTION_MODE}" >&2
    exit 2
    ;;
esac

case "${OUTCOME_TRACE_TARGET_MODE}" in
  none|numeric|textual|numeric_textual) ;;
  *)
    echo "[ERROR] Unsupported OUTCOME_TRACE_TARGET_MODE=${OUTCOME_TRACE_TARGET_MODE}" >&2
    exit 2
    ;;
esac

case "${PROPOSAL_OUTPUT_SCHEMA}" in
  plain|action_prediction) ;;
  *)
    echo "[ERROR] Unsupported PROPOSAL_OUTPUT_SCHEMA=${PROPOSAL_OUTPUT_SCHEMA}" >&2
    exit 2
    ;;
esac

case "${PROPOSAL_GRPO_ZERO_VARIANCE}" in
  fixed_baseline|skip) ;;
  *)
    echo "[ERROR] Unsupported PROPOSAL_GRPO_ZERO_VARIANCE=${PROPOSAL_GRPO_ZERO_VARIANCE}" >&2
    exit 2
    ;;
esac

case "${PROPOSAL_GRPO_REWARD_MODE}" in
  outcome|validity) ;;
  *)
    echo "[ERROR] Unsupported PROPOSAL_GRPO_REWARD_MODE=${PROPOSAL_GRPO_REWARD_MODE}" >&2
    exit 2
    ;;
esac

TASK_ARGS=()
if [[ "${TASK}" == "addition" ]]; then
  TASK_ARGS+=(
    --initial-min-size "${INITIAL_MIN_SIZE:-3}"
    --initial-max-size "${INITIAL_MAX_SIZE:-7}"
    --frontier-max-size "${FRONTIER_MAX_SIZE:-31}"
    --initial-train-per-size "${INITIAL_TRAIN_PER_SIZE:-5000}"
    --candidate-train-per-size "${CANDIDATE_TRAIN_PER_SIZE:-5000}"
    --decode-max-new-tokens "${DECODE_MAX_NEW_TOKENS:-48}"
  )
elif [[ "${TASK}" == "run_length" ]]; then
  TASK_ARGS+=(
    --initial-min-size "${INITIAL_MIN_SIZE:-8}"
    --initial-max-size "${INITIAL_MAX_SIZE:-16}"
    --frontier-max-size "${FRONTIER_MAX_SIZE:-48}"
    --initial-train-per-size "${INITIAL_TRAIN_PER_SIZE:-50000}"
    --candidate-train-per-size "${CANDIDATE_TRAIN_PER_SIZE:-2000}"
    --target-mode "${TARGET_MODE:-run_state}"
    --symbol-alphabet-size "${SYMBOL_ALPHABET_SIZE:-2}"
    --decode-max-new-tokens "${DECODE_MAX_NEW_TOKENS:-32}"
  )
else
  echo "[ERROR] Unsupported TASK=${TASK}" >&2
  exit 2
fi

OPTIONAL_ARGS=()
if [[ -n "${PROPOSAL_FIXTURE_JSONL:-}" ]]; then
  OPTIONAL_ARGS+=(--proposal-fixture-jsonl "${PROPOSAL_FIXTURE_JSONL}")
fi
if [[ "${TREAT_SEED_AS_ROUND_ZERO:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--treat-seed-as-round-zero)
fi
if [[ "${INIT_FROM_SCRATCH:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--init-from-scratch)
fi
if [[ "${TOKENIZER_MODE:-auto}" != "auto" ]]; then
  OPTIONAL_ARGS+=(--tokenizer-mode "${TOKENIZER_MODE}")
fi
if [[ "${KEEP_ALL_CANDIDATE_MODELS:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--keep-all-candidate-models)
fi
if [[ "${KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS}" == "1" ]]; then
  OPTIONAL_ARGS+=(--keep-all-proposal-grpo-checkpoints)
fi
if [[ "${CANDIDATE_LOCAL_CACHE_BASE_STATE}" == "1" ]]; then
  OPTIONAL_ARGS+=(--candidate-local-cache-base-state)
fi
if [[ "${BUCKET_TRAIN_BATCHES_BY_SIZE:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--bucket-train-batches-by-size)
fi
if [[ "${ALLOW_REPEAT_TARGETS:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--allow-repeat-targets)
fi
if [[ "${POST_TASK_PROPOSAL_REHEARSAL}" == "1" ]]; then
  OPTIONAL_ARGS+=(--post-task-proposal-rehearsal)
else
  OPTIONAL_ARGS+=(--no-post-task-proposal-rehearsal)
fi
if [[ -n "${INIT_FINAL_ACCURACY:-}" ]]; then
  OPTIONAL_ARGS+=(--init-final-accuracy "${INIT_FINAL_ACCURACY}")
fi
if [[ "${BF16:-1}" == "1" ]]; then
  OPTIONAL_ARGS+=(--bf16)
elif [[ "${FP16:-0}" == "1" ]]; then
  OPTIONAL_ARGS+=(--fp16)
fi

echo "[INFO] Adaptive candidate loop"
echo "[INFO] Task: ${TASK}"
echo "[INFO] Condition: ${CONDITION}"
echo "[INFO] Proposal output schema: ${PROPOSAL_OUTPUT_SCHEMA}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Proposal model: ${PROPOSAL_MODEL_NAME}"
echo "[INFO] Rounds/candidates: ${NUM_ROUNDS}/${NUM_CANDIDATES}"
echo "[INFO] Max attempts / no-selection patience: ${MAX_ATTEMPT_ROUNDS}/${NO_SELECTION_PATIENCE}"
echo "[INFO] Train epochs/max steps: ${NUM_EPOCHS:-1}/${MAX_STEPS} (MAX_STEPS=0 uses epoch-based training)"
echo "[INFO] Learning rate: ${LEARNING_RATE}"
echo "[INFO] Train/eval batch size: ${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
echo "[INFO] Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "[INFO] Controller execution mode: ${CONTROLLER_EXECUTION_MODE}"
echo "[INFO] Controller worker poll/fallback time limit: ${CONTROLLER_WORKER_POLL_SECONDS}/${CONTROLLER_WORKER_TIME_LIMIT}"
echo "[INFO] Controller seed/round/GRPO time limits: ${CONTROLLER_SEED_WORKER_TIME_LIMIT}/${CONTROLLER_ROUND_WORKER_TIME_LIMIT}/${CONTROLLER_GRPO_WORKER_TIME_LIMIT}"
echo "[INFO] Candidate execution mode: ${CANDIDATE_EXECUTION_MODE}"
echo "[INFO] Candidate local parallelism/pack/cache-base-state: ${CANDIDATE_LOCAL_PARALLELISM}/${CANDIDATE_LOCAL_PACK_SIZE}/${CANDIDATE_LOCAL_CACHE_BASE_STATE}"
echo "[INFO] Candidate array max parallel/poll/time limit: ${CANDIDATE_ARRAY_MAX_PARALLEL}/${CANDIDATE_ARRAY_POLL_SECONDS}/${CANDIDATE_ARRAY_TIME_LIMIT}"
echo "[INFO] Proposal trace replay ratio/max examples: ${PROPOSAL_TRACE_REPLAY_RATIO}/${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES}"
echo "[INFO] Post-task proposal rehearsal/repeats/max examples: ${POST_TASK_PROPOSAL_REHEARSAL}/${POST_TASK_PROPOSAL_REHEARSAL_REPEAT_COUNT}/${POST_TASK_PROPOSAL_REHEARSAL_MAX_EXAMPLES}"
echo "[INFO] Outcome trace mode: ${OUTCOME_TRACE_TARGET_MODE}"
echo "[INFO] Outcome trace replay ratio/max examples: ${OUTCOME_TRACE_REPLAY_RATIO}/${OUTCOME_TRACE_REPLAY_MAX_EXAMPLES}"
echo "[INFO] Invalid outcome reward: ${INVALID_OUTCOME_REWARD}"
echo "[INFO] Proposal GRPO steps/lr/kl: ${PROPOSAL_GRPO_STEPS}/${PROPOSAL_GRPO_LEARNING_RATE}/${PROPOSAL_GRPO_KL_COEF}"
echo "[INFO] Proposal GRPO grad clip/zero variance/baseline: ${PROPOSAL_GRPO_GRAD_CLIP}/${PROPOSAL_GRPO_ZERO_VARIANCE}/${PROPOSAL_GRPO_FIXED_BASELINE}"
echo "[INFO] Proposal GRPO reward mode/outcome scale: ${PROPOSAL_GRPO_REWARD_MODE}/${PROPOSAL_GRPO_OUTCOME_SCALE}"
echo "[INFO] Keep all proposal-GRPO checkpoints: ${KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS}"
echo "[INFO] HF_HOME: ${HF_HOME}"
echo "[INFO] HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "[INFO] PYTORCH_CUDA_ALLOC_CONF: ${PYTORCH_CUDA_ALLOC_CONF}"
echo "[INFO] Python: ${PYTHON_BIN}"
echo "[INFO] Slurm Job ID: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
adaptive_print_torch_probe

if [[ "${RUN_COMPILE_CHECK}" == "1" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    self/core/driver.py \
    self/core/proposals.py \
    self/core/program_sandbox.py \
    self/self_improvement_core.py \
    self/self_improvement_tasks.py
fi

"${PYTHON_BIN}" -m self.core.driver \
  --task "${TASK}" \
  --condition "${CONDITION}" \
  --model-name "${MODEL_NAME}" \
  --proposal-model-name "${PROPOSAL_MODEL_NAME}" \
  --proposal-output-schema "${PROPOSAL_OUTPUT_SCHEMA}" \
  --output-dir "${OUT_DIR}" \
  --num-rounds "${NUM_ROUNDS}" \
  --max-attempt-rounds "${MAX_ATTEMPT_ROUNDS}" \
  --no-selection-patience "${NO_SELECTION_PATIENCE}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --num-epochs "${NUM_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --per-device-train-batch-size "${TRAIN_BATCH_SIZE}" \
  --per-device-eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --controller-execution-mode "${CONTROLLER_EXECUTION_MODE}" \
  --controller-worker-sbatch-script "${CONTROLLER_WORKER_SBATCH_SCRIPT}" \
  --controller-worker-poll-seconds "${CONTROLLER_WORKER_POLL_SECONDS}" \
  --controller-worker-time-limit "${CONTROLLER_WORKER_TIME_LIMIT}" \
  --controller-seed-worker-time-limit "${CONTROLLER_SEED_WORKER_TIME_LIMIT}" \
  --controller-round-worker-time-limit "${CONTROLLER_ROUND_WORKER_TIME_LIMIT}" \
  --controller-grpo-worker-time-limit "${CONTROLLER_GRPO_WORKER_TIME_LIMIT}" \
  --candidate-execution-mode "${CANDIDATE_EXECUTION_MODE}" \
  --candidate-local-parallelism "${CANDIDATE_LOCAL_PARALLELISM}" \
  --candidate-local-pack-size "${CANDIDATE_LOCAL_PACK_SIZE}" \
  --candidate-array-sbatch-script "${CANDIDATE_ARRAY_SBATCH_SCRIPT}" \
  --candidate-array-max-parallel "${CANDIDATE_ARRAY_MAX_PARALLEL}" \
  --candidate-array-poll-seconds "${CANDIDATE_ARRAY_POLL_SECONDS}" \
  --candidate-array-timeout-seconds "${CANDIDATE_ARRAY_TIMEOUT_SECONDS}" \
  --candidate-array-time-limit "${CANDIDATE_ARRAY_TIME_LIMIT}" \
  --initial-eval-per-size "${INITIAL_EVAL_PER_SIZE}" \
  --eval-per-size "${EVAL_PER_SIZE}" \
  --composed-eval-per-size "${COMPOSED_EVAL_PER_SIZE}" \
  --proposal-max-new-tokens "${PROPOSAL_MAX_NEW_TOKENS}" \
  --repair-attempts "${REPAIR_ATTEMPTS}" \
  --proposal-trace-replay-ratio "${PROPOSAL_TRACE_REPLAY_RATIO}" \
  --proposal-trace-replay-max-examples "${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES}" \
  --post-task-proposal-rehearsal-repeat-count "${POST_TASK_PROPOSAL_REHEARSAL_REPEAT_COUNT}" \
  --post-task-proposal-rehearsal-max-examples "${POST_TASK_PROPOSAL_REHEARSAL_MAX_EXAMPLES}" \
  --outcome-trace-target-mode "${OUTCOME_TRACE_TARGET_MODE}" \
  --outcome-trace-replay-ratio "${OUTCOME_TRACE_REPLAY_RATIO}" \
  --outcome-trace-replay-max-examples "${OUTCOME_TRACE_REPLAY_MAX_EXAMPLES}" \
  --invalid-outcome-reward "${INVALID_OUTCOME_REWARD}" \
  --proposal-grpo-steps "${PROPOSAL_GRPO_STEPS}" \
  --proposal-grpo-learning-rate "${PROPOSAL_GRPO_LEARNING_RATE}" \
  --proposal-grpo-kl-coef "${PROPOSAL_GRPO_KL_COEF}" \
  --proposal-grpo-grad-clip "${PROPOSAL_GRPO_GRAD_CLIP}" \
  --proposal-grpo-zero-variance "${PROPOSAL_GRPO_ZERO_VARIANCE}" \
  --proposal-grpo-reward-mode "${PROPOSAL_GRPO_REWARD_MODE}" \
  --proposal-grpo-outcome-scale "${PROPOSAL_GRPO_OUTCOME_SCALE}" \
  --proposal-grpo-fixed-baseline "${PROPOSAL_GRPO_FIXED_BASELINE}" \
  --program-timeout-seconds "${PROGRAM_TIMEOUT_SECONDS}" \
  --program-batch-timeout-seconds "${PROGRAM_BATCH_TIMEOUT_SECONDS}" \
  --lambda-final "${LAMBDA_FINAL}" \
  --selection-min-reward "${SELECTION_MIN_REWARD}" \
  --seed "${SEED:-42}" \
  --plan-log-path "${PLAN_LOG_PATH}" \
  "${TASK_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}"

echo "[INFO] Finished adaptive candidate loop."

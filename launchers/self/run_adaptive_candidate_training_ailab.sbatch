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

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/launchers/self/lib/adaptive_common.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/launchers/self"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
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
TREAT_SEED_AS_ROUND_ZERO="${TREAT_SEED_AS_ROUND_ZERO:-0}"
CONDITION="${CONDITION:-config}"
PROPOSAL_OUTPUT_SCHEMA="${PROPOSAL_OUTPUT_SCHEMA:-action_observation}"
PROPOSAL_PROMPT_ACTION_HISTORY="${PROPOSAL_PROMPT_ACTION_HISTORY:-1}"
PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS="${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS:-5}"
PREPARED_START_RUN_DIR="${PREPARED_START_RUN_DIR:-}"
MAX_ATTEMPT_ROUNDS="${MAX_ATTEMPT_ROUNDS:-100}"
MAX_SELECTED_ROUNDS="${MAX_SELECTED_ROUNDS:-0}"
NO_SELECTION_PATIENCE="${NO_SELECTION_PATIENCE:-10}"
if [[ "${NO_SELECTION_PATIENCE}" =~ ^[0-9]+$ && "${NO_SELECTION_PATIENCE}" -lt 1 ]]; then
  NO_SELECTION_PATIENCE="1"
fi
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
PROPOSAL_TEMPERATURE="${PROPOSAL_TEMPERATURE:-0.9}"
PROPOSAL_TOP_P="${PROPOSAL_TOP_P:-0.95}"
FORCE_UNIQUE_PROPOSALS="${FORCE_UNIQUE_PROPOSALS:-1}"
PROPOSAL_UNIQUE_MAX_DRAWS="${PROPOSAL_UNIQUE_MAX_DRAWS:-0}"
SYNTHETIC_PROPOSAL_SFT="${SYNTHETIC_PROPOSAL_SFT:-0}"
SYNTHETIC_PROPOSAL_SFT_SEED_MIX="${SYNTHETIC_PROPOSAL_SFT_SEED_MIX:-0}"
SYNTHETIC_PROPOSAL_SFT_EXAMPLES="${SYNTHETIC_PROPOSAL_SFT_EXAMPLES:-0}"
SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS="${SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS:-1}"
SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE="${SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE:-1e-6}"
SYNTHETIC_PROPOSAL_SFT_TOP_K="${SYNTHETIC_PROPOSAL_SFT_TOP_K:-4}"
SYNTHETIC_PROPOSAL_SFT_TEMPERATURE="${SYNTHETIC_PROPOSAL_SFT_TEMPERATURE:-0.7}"
MAX_STEPS="${MAX_STEPS:-100}"
SEED_MAX_STEPS="${SEED_MAX_STEPS:-0}"
CANDIDATE_TRAINING_MODE="${CANDIDATE_TRAINING_MODE:-single_stage}"
ROLLOUT_TRAIN_PER_SIZE="${ROLLOUT_TRAIN_PER_SIZE:-512}"
ROLLOUT_EVAL_PER_SIZE="${ROLLOUT_EVAL_PER_SIZE:-20}"
SELECTED_MAX_STEPS="${SELECTED_MAX_STEPS:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
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
CANDIDATE_LOCAL_PARALLELISM="${CANDIDATE_LOCAL_PARALLELISM:-2}"
CANDIDATE_LOCAL_PACK_SIZE="${CANDIDATE_LOCAL_PACK_SIZE:-1}"
CANDIDATE_LOCAL_CACHE_BASE_STATE="${CANDIDATE_LOCAL_CACHE_BASE_STATE:-0}"
LAMBDA_FINAL="${LAMBDA_FINAL:-0.1}"
SELECTION_MIN_REWARD="${SELECTION_MIN_REWARD:-0.0}"
SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD="${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD:-0.80}"
PROPOSAL_TRACE_REPLAY_RATIO="${PROPOSAL_TRACE_REPLAY_RATIO:-0.0}"
PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES="${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES:-256}"
PROPOSAL_OBSERVATION_LOSS_WEIGHT="${PROPOSAL_OBSERVATION_LOSS_WEIGHT:-0.2}"
PROPOSAL_FORMAT_LOSS_WEIGHT="${PROPOSAL_FORMAT_LOSS_WEIGHT:-0.02}"
PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES="${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES:-256}"
PROPOSAL_GRPO_DEDUPLICATE_ACTIONS="${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS:-1}"
PROPOSAL_GRPO_SPAN="${PROPOSAL_GRPO_SPAN:-reasoning_action}"
PROPOSAL_UPDATE_MICROBATCH_SIZE="${PROPOSAL_UPDATE_MICROBATCH_SIZE:-8}"
PROPOSAL_UPDATE_ACCUMULATION_STEPS="${PROPOSAL_UPDATE_ACCUMULATION_STEPS:-1}"
PROPOSAL_SAMPLING_BATCH_SIZE="${PROPOSAL_SAMPLING_BATCH_SIZE:-8}"
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
PROPOSAL_GRPO_ZERO_VARIANCE="${PROPOSAL_GRPO_ZERO_VARIANCE:-skip}"
PROPOSAL_GRPO_REWARD_MODE="${PROPOSAL_GRPO_REWARD_MODE:-outcome}"
PROPOSAL_GRPO_OUTCOME_SCALE="${PROPOSAL_GRPO_OUTCOME_SCALE:-0.05}"
PROPOSAL_GRPO_FIXED_BASELINE="${PROPOSAL_GRPO_FIXED_BASELINE:-0.5}"
PROPOSAL_GRPO_NOVELTY_BONUS_BETA="${PROPOSAL_GRPO_NOVELTY_BONUS_BETA:-0.05}"
KEEP_FINAL_MODEL_CHECKPOINT="${KEEP_FINAL_MODEL_CHECKPOINT:-0}"
KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS="${KEEP_ALL_PROPOSAL_GRPO_CHECKPOINTS:-0}"
INITIAL_EVAL_PER_SIZE="${INITIAL_EVAL_PER_SIZE:-50}"
EVAL_PER_SIZE="${EVAL_PER_SIZE:-100}"
COMPOSED_EVAL_PER_SIZE="${COMPOSED_EVAL_PER_SIZE:-100}"
PLAN_LOG_PATH="${PLAN_LOG_PATH:-${ROOT_DIR}/plan/260603-self-improvement-init.md}"
RUN_COMPILE_CHECK="${RUN_COMPILE_CHECK:-1}"
PROPOSAL_MAX_NEW_TOKENS="${PROPOSAL_MAX_NEW_TOKENS:-160}"

case "${CONDITION}" in
  config) ;;
  *)
    echo "[ERROR] Unsupported CONDITION=${CONDITION}" >&2
    exit 2
    ;;
esac

case "${CANDIDATE_EXECUTION_MODE}" in
  local_parallel|serial) ;;
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
  plain|action_observation) ;;
  *)
    echo "[ERROR] Unsupported PROPOSAL_OUTPUT_SCHEMA=${PROPOSAL_OUTPUT_SCHEMA}" >&2
    exit 2
    ;;
esac

case "${PROPOSAL_PROMPT_ACTION_HISTORY}" in
  0|1) ;;
  *)
    echo "[ERROR] PROPOSAL_PROMPT_ACTION_HISTORY must be 0 or 1, got ${PROPOSAL_PROMPT_ACTION_HISTORY}" >&2
    exit 2
    ;;
esac

case "${FORCE_UNIQUE_PROPOSALS}" in
  0|1) ;;
  *)
    echo "[ERROR] FORCE_UNIQUE_PROPOSALS must be 0 or 1, got ${FORCE_UNIQUE_PROPOSALS}" >&2
    exit 2
    ;;
esac

case "${SYNTHETIC_PROPOSAL_SFT}" in
  0|1) ;;
  *)
    echo "[ERROR] SYNTHETIC_PROPOSAL_SFT must be 0 or 1, got ${SYNTHETIC_PROPOSAL_SFT}" >&2
    exit 2
    ;;
esac

case "${SYNTHETIC_PROPOSAL_SFT_SEED_MIX}" in
  0|1) ;;
  *)
    echo "[ERROR] SYNTHETIC_PROPOSAL_SFT_SEED_MIX must be 0 or 1, got ${SYNTHETIC_PROPOSAL_SFT_SEED_MIX}" >&2
    exit 2
    ;;
esac

if [[ "${SYNTHETIC_PROPOSAL_SFT}" == "1" && "${SYNTHETIC_PROPOSAL_SFT_SEED_MIX}" == "1" ]]; then
  echo "[ERROR] SYNTHETIC_PROPOSAL_SFT and SYNTHETIC_PROPOSAL_SFT_SEED_MIX are mutually exclusive." >&2
  exit 2
fi

case "${PROPOSAL_GRPO_ZERO_VARIANCE}" in
  fixed_baseline|skip) ;;
  *)
    echo "[ERROR] Unsupported PROPOSAL_GRPO_ZERO_VARIANCE=${PROPOSAL_GRPO_ZERO_VARIANCE}" >&2
    exit 2
    ;;
esac

case "${PROPOSAL_GRPO_REWARD_MODE}" in
  outcome|rank|validity) ;;
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
if [[ "${KEEP_FINAL_MODEL_CHECKPOINT}" == "1" ]]; then
  OPTIONAL_ARGS+=(--keep-final-model-checkpoint)
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
if [[ "${FORCE_UNIQUE_PROPOSALS}" == "1" ]]; then
  OPTIONAL_ARGS+=(--force-unique-proposals)
else
  OPTIONAL_ARGS+=(--no-force-unique-proposals)
fi
if [[ -n "${PREPARED_START_RUN_DIR}" ]]; then
  OPTIONAL_ARGS+=(--prepared-start-run-dir "${PREPARED_START_RUN_DIR}")
fi
if [[ "${SYNTHETIC_PROPOSAL_SFT}" == "1" ]]; then
  OPTIONAL_ARGS+=(--synthetic-proposal-sft)
fi
if [[ "${SYNTHETIC_PROPOSAL_SFT_SEED_MIX}" == "1" ]]; then
  OPTIONAL_ARGS+=(--synthetic-proposal-sft-seed-mix)
fi
if [[ "${PROPOSAL_PROMPT_ACTION_HISTORY}" == "1" ]]; then
  OPTIONAL_ARGS+=(--proposal-prompt-action-history)
else
  OPTIONAL_ARGS+=(--no-proposal-prompt-action-history)
fi
if [[ "${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS}" == "1" ]]; then
  OPTIONAL_ARGS+=(--proposal-grpo-deduplicate-actions)
else
  OPTIONAL_ARGS+=(--no-proposal-grpo-deduplicate-actions)
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
echo "[INFO] Proposal prompt action history/max items: ${PROPOSAL_PROMPT_ACTION_HISTORY}/${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS}"
echo "[INFO] Output dir: ${OUT_DIR}"
echo "[INFO] Model: ${MODEL_NAME}"
echo "[INFO] Proposal model: ${PROPOSAL_MODEL_NAME}"
echo "[INFO] Treat seed as round zero: ${TREAT_SEED_AS_ROUND_ZERO}"
echo "[INFO] Prepared start run dir: ${PREPARED_START_RUN_DIR:-none}"
echo "[INFO] Proposal sampling temp/top_p/force-unique/max-draws/batch-size: ${PROPOSAL_TEMPERATURE}/${PROPOSAL_TOP_P}/${FORCE_UNIQUE_PROPOSALS}/${PROPOSAL_UNIQUE_MAX_DRAWS}/${PROPOSAL_SAMPLING_BATCH_SIZE}"
echo "[INFO] Synthetic proposal SFT enabled/examples/epochs/lr/top-k/temp: ${SYNTHETIC_PROPOSAL_SFT}/${SYNTHETIC_PROPOSAL_SFT_EXAMPLES}/${SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS}/${SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE}/${SYNTHETIC_PROPOSAL_SFT_TOP_K}/${SYNTHETIC_PROPOSAL_SFT_TEMPERATURE}"
echo "[INFO] Synthetic proposal seed mix: ${SYNTHETIC_PROPOSAL_SFT_SEED_MIX}"
echo "[INFO] Attempts/candidates: ${MAX_ATTEMPT_ROUNDS}/${NUM_CANDIDATES}"
echo "[INFO] Max selected candidates: ${MAX_SELECTED_ROUNDS}"
echo "[INFO] Max attempts / no-selection patience: ${MAX_ATTEMPT_ROUNDS}/${NO_SELECTION_PATIENCE}"
echo "[INFO] Train epochs/candidate max steps/seed max steps: ${NUM_EPOCHS:-1}/${MAX_STEPS}/${SEED_MAX_STEPS} (0 uses epoch-based training)"
echo "[INFO] Candidate training mode/rollout train/eval/selected max steps: ${CANDIDATE_TRAINING_MODE}/${ROLLOUT_TRAIN_PER_SIZE}/${ROLLOUT_EVAL_PER_SIZE}/${SELECTED_MAX_STEPS}"
echo "[INFO] Learning rate: ${LEARNING_RATE}"
echo "[INFO] Train/eval batch size: ${TRAIN_BATCH_SIZE}/${EVAL_BATCH_SIZE}"
echo "[INFO] Gradient accumulation steps: ${GRAD_ACCUM_STEPS}"
echo "[INFO] Controller execution mode: ${CONTROLLER_EXECUTION_MODE}"
echo "[INFO] Controller worker poll/fallback time limit: ${CONTROLLER_WORKER_POLL_SECONDS}/${CONTROLLER_WORKER_TIME_LIMIT}"
echo "[INFO] Controller seed/round/GRPO time limits: ${CONTROLLER_SEED_WORKER_TIME_LIMIT}/${CONTROLLER_ROUND_WORKER_TIME_LIMIT}/${CONTROLLER_GRPO_WORKER_TIME_LIMIT}"
echo "[INFO] Candidate execution mode: ${CANDIDATE_EXECUTION_MODE}"
echo "[INFO] Candidate local parallelism/pack/cache-base-state: ${CANDIDATE_LOCAL_PARALLELISM}/${CANDIDATE_LOCAL_PACK_SIZE}/${CANDIDATE_LOCAL_CACHE_BASE_STATE}"
echo "[INFO] Candidate eval backend: transformers"
echo "[INFO] Proposal trace replay ratio/max examples: ${PROPOSAL_TRACE_REPLAY_RATIO}/${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES}"
echo "[INFO] Outcome trace mode: ${OUTCOME_TRACE_TARGET_MODE}"
echo "[INFO] Outcome trace replay ratio/max examples: ${OUTCOME_TRACE_REPLAY_RATIO}/${OUTCOME_TRACE_REPLAY_MAX_EXAMPLES}"
echo "[INFO] Invalid outcome reward: ${INVALID_OUTCOME_REWARD}"
echo "[INFO] Proposal GRPO steps/lr/kl: ${PROPOSAL_GRPO_STEPS}/${PROPOSAL_GRPO_LEARNING_RATE}/${PROPOSAL_GRPO_KL_COEF}"
echo "[INFO] Proposal GRPO grad clip/zero variance/baseline: ${PROPOSAL_GRPO_GRAD_CLIP}/${PROPOSAL_GRPO_ZERO_VARIANCE}/${PROPOSAL_GRPO_FIXED_BASELINE}"
echo "[INFO] Proposal GRPO reward mode/outcome scale/dedup/novelty beta: ${PROPOSAL_GRPO_REWARD_MODE}/${PROPOSAL_GRPO_OUTCOME_SCALE}/${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS}/${PROPOSAL_GRPO_NOVELTY_BONUS_BETA}"
echo "[INFO] Source admission target-accuracy threshold: ${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD}"
echo "[INFO] Proposal update loss/observation/format/replay/microbatch/accum: merged_agent/${PROPOSAL_OBSERVATION_LOSS_WEIGHT}/${PROPOSAL_FORMAT_LOSS_WEIGHT}/${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES}/${PROPOSAL_UPDATE_MICROBATCH_SIZE}/${PROPOSAL_UPDATE_ACCUMULATION_STEPS}"
echo "[INFO] Proposal GRPO span: ${PROPOSAL_GRPO_SPAN}"
echo "[INFO] Keep final model checkpoint: ${KEEP_FINAL_MODEL_CHECKPOINT}"
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
    self/adaptive/driver.py \
    self/adaptive/proposal.py \
    self/core/composition.py \
    self/core/models.py
fi

"${PYTHON_BIN}" -m self.adaptive.driver \
  --task "${TASK}" \
  --condition "${CONDITION}" \
  --model-name "${MODEL_NAME}" \
  --proposal-model-name "${PROPOSAL_MODEL_NAME}" \
  --proposal-output-schema "${PROPOSAL_OUTPUT_SCHEMA}" \
  --proposal-prompt-action-history-max-items "${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS}" \
  --output-dir "${OUT_DIR}" \
  --max-selected-rounds "${MAX_SELECTED_ROUNDS}" \
  --max-attempt-rounds "${MAX_ATTEMPT_ROUNDS}" \
  --no-selection-patience "${NO_SELECTION_PATIENCE}" \
  --num-candidates "${NUM_CANDIDATES}" \
  --num-epochs "${NUM_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS}" \
  --seed-max-steps "${SEED_MAX_STEPS}" \
  --candidate-training-mode "${CANDIDATE_TRAINING_MODE}" \
  --rollout-train-per-size "${ROLLOUT_TRAIN_PER_SIZE}" \
  --rollout-eval-per-size "${ROLLOUT_EVAL_PER_SIZE}" \
  --selected-max-steps "${SELECTED_MAX_STEPS}" \
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
  --initial-eval-per-size "${INITIAL_EVAL_PER_SIZE}" \
  --eval-per-size "${EVAL_PER_SIZE}" \
  --composed-eval-per-size "${COMPOSED_EVAL_PER_SIZE}" \
  --proposal-max-new-tokens "${PROPOSAL_MAX_NEW_TOKENS}" \
  --proposal-temperature "${PROPOSAL_TEMPERATURE}" \
  --proposal-top-p "${PROPOSAL_TOP_P}" \
  --proposal-sampling-batch-size "${PROPOSAL_SAMPLING_BATCH_SIZE}" \
  --proposal-unique-max-draws "${PROPOSAL_UNIQUE_MAX_DRAWS}" \
  --synthetic-proposal-sft-examples "${SYNTHETIC_PROPOSAL_SFT_EXAMPLES}" \
  --synthetic-proposal-sft-num-epochs "${SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS}" \
  --synthetic-proposal-sft-learning-rate "${SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE}" \
  --synthetic-proposal-sft-top-k "${SYNTHETIC_PROPOSAL_SFT_TOP_K}" \
  --synthetic-proposal-sft-temperature "${SYNTHETIC_PROPOSAL_SFT_TEMPERATURE}" \
  --proposal-trace-replay-ratio "${PROPOSAL_TRACE_REPLAY_RATIO}" \
  --proposal-trace-replay-max-examples "${PROPOSAL_TRACE_REPLAY_MAX_EXAMPLES}" \
  --outcome-trace-target-mode "${OUTCOME_TRACE_TARGET_MODE}" \
  --outcome-trace-replay-ratio "${OUTCOME_TRACE_REPLAY_RATIO}" \
  --outcome-trace-replay-max-examples "${OUTCOME_TRACE_REPLAY_MAX_EXAMPLES}" \
  --invalid-outcome-reward "${INVALID_OUTCOME_REWARD}" \
  --proposal-grpo-steps "${PROPOSAL_GRPO_STEPS}" \
  --proposal-observation-loss-weight "${PROPOSAL_OBSERVATION_LOSS_WEIGHT}" \
  --proposal-format-loss-weight "${PROPOSAL_FORMAT_LOSS_WEIGHT}" \
  --proposal-format-replay-max-examples "${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES}" \
  --proposal-grpo-span "${PROPOSAL_GRPO_SPAN}" \
  --proposal-update-microbatch-size "${PROPOSAL_UPDATE_MICROBATCH_SIZE}" \
  --proposal-update-accumulation-steps "${PROPOSAL_UPDATE_ACCUMULATION_STEPS}" \
  --proposal-grpo-learning-rate "${PROPOSAL_GRPO_LEARNING_RATE}" \
  --proposal-grpo-kl-coef "${PROPOSAL_GRPO_KL_COEF}" \
  --proposal-grpo-grad-clip "${PROPOSAL_GRPO_GRAD_CLIP}" \
  --proposal-grpo-zero-variance "${PROPOSAL_GRPO_ZERO_VARIANCE}" \
  --proposal-grpo-reward-mode "${PROPOSAL_GRPO_REWARD_MODE}" \
  --proposal-grpo-outcome-scale "${PROPOSAL_GRPO_OUTCOME_SCALE}" \
  --proposal-grpo-fixed-baseline "${PROPOSAL_GRPO_FIXED_BASELINE}" \
  --proposal-grpo-novelty-bonus-beta "${PROPOSAL_GRPO_NOVELTY_BONUS_BETA}" \
  --lambda-final "${LAMBDA_FINAL}" \
  --selection-min-reward "${SELECTION_MIN_REWARD}" \
  --source-admission-target-accuracy-threshold "${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD}" \
  --seed "${SEED:-42}" \
  --plan-log-path "${PLAN_LOG_PATH}" \
  "${TASK_ARGS[@]}" \
  "${OPTIONAL_ARGS[@]}"

echo "[INFO] Finished adaptive candidate loop."

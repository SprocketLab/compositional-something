#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/adaptive_candidate_training_ailab_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT_DIR}/launchers/self/run_adaptive_candidate_training_ailab.sbatch}"
TASKS="${TASKS:-addition run_length}"
CONDITIONS="${CONDITIONS:-config}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PROPOSAL_MODEL_NAME="${PROPOSAL_MODEL_NAME:-current}"
TREAT_SEED_AS_ROUND_ZERO="${TREAT_SEED_AS_ROUND_ZERO:-0}"
OUTCOME_TRACE_TARGET_MODES="${OUTCOME_TRACE_TARGET_MODES:-numeric}"
PROPOSAL_GRPO_REWARD_MODES="${PROPOSAL_GRPO_REWARD_MODES:-outcome}"
PROPOSAL_GRPO_ZERO_VARIANCE_MODES="${PROPOSAL_GRPO_ZERO_VARIANCE_MODES:-skip}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
NUM_CANDIDATES_LIST="${NUM_CANDIDATES_LIST:-${NUM_CANDIDATES}}"
PROPOSAL_TEMPERATURE="${PROPOSAL_TEMPERATURE:-0.9}"
PROPOSAL_TOP_P="${PROPOSAL_TOP_P:-0.95}"
PROPOSAL_PROMPT_ACTION_HISTORY="${PROPOSAL_PROMPT_ACTION_HISTORY:-1}"
PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS="${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS:-5}"
FORCE_UNIQUE_PROPOSALS="${FORCE_UNIQUE_PROPOSALS:-1}"
PROPOSAL_UNIQUE_MAX_DRAWS="${PROPOSAL_UNIQUE_MAX_DRAWS:-0}"
MAX_ATTEMPT_ROUNDS="${MAX_ATTEMPT_ROUNDS:-100}"
MAX_SELECTED_ROUNDS="${MAX_SELECTED_ROUNDS:-0}"
NO_SELECTION_PATIENCE="${NO_SELECTION_PATIENCE:-${MAX_ATTEMPT_ROUNDS}}"
MAX_STEPS="${MAX_STEPS:-100}"
SEED_MAX_STEPS="${SEED_MAX_STEPS:-0}"
SBATCH_MEM="${SBATCH_MEM:-48G}"
SBATCH_TIME="${SBATCH_TIME:-}"
SBATCH_DEPENDENCY="${SBATCH_DEPENDENCY:-}"
ADAPTIVE_CONFIG_EXPORT="${ADAPTIVE_CONFIG_FILES:-${ADAPTIVE_CONFIG_FILE:-${ROOT_DIR}/launchers/self/config/adaptive_candidate_base.env}}"
CANDIDATE_LOCAL_PARALLELISM="${CANDIDATE_LOCAL_PARALLELISM:-2}"
CANDIDATE_LOCAL_PACK_SIZE="${CANDIDATE_LOCAL_PACK_SIZE:-2}"
CANDIDATE_LOCAL_CACHE_BASE_STATE="${CANDIDATE_LOCAL_CACHE_BASE_STATE:-1}"
CANDIDATE_EVAL_BACKEND="${CANDIDATE_EVAL_BACKEND:-transformers}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
VLLM_FLASHINFER_SAMPLER="${VLLM_FLASHINFER_SAMPLER:-off}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-0}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-0}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-0}"
PROPOSAL_UPDATE_LOSS_MODE="${PROPOSAL_UPDATE_LOSS_MODE:-merged_agent}"
PROPOSAL_OBSERVATION_LOSS_WEIGHT="${PROPOSAL_OBSERVATION_LOSS_WEIGHT:-0.2}"
PROPOSAL_FORMAT_LOSS_WEIGHT="${PROPOSAL_FORMAT_LOSS_WEIGHT:-0.02}"
PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES="${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES:-256}"
PROPOSAL_GRPO_DEDUPLICATE_ACTIONS="${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS:-1}"
PROPOSAL_GRPO_SPAN="${PROPOSAL_GRPO_SPAN:-reasoning_action}"
PROPOSAL_GRPO_LEARNING_RATE="${PROPOSAL_GRPO_LEARNING_RATE:-1e-6}"
PROPOSAL_GRPO_LEARNING_RATES="${PROPOSAL_GRPO_LEARNING_RATES:-${PROPOSAL_GRPO_LEARNING_RATE}}"
PROPOSAL_GRPO_KL_COEF="${PROPOSAL_GRPO_KL_COEF:-0.01}"
PROPOSAL_GRPO_NOVELTY_BONUS_BETA="${PROPOSAL_GRPO_NOVELTY_BONUS_BETA:-0.05}"
SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD="${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD:-0.80}"
PROPOSAL_UPDATE_MICROBATCH_SIZE="${PROPOSAL_UPDATE_MICROBATCH_SIZE:-8}"
PROPOSAL_SAMPLING_BATCH_SIZE="${PROPOSAL_SAMPLING_BATCH_SIZE:-8}"
if [[ -z "${POST_TASK_PROPOSAL_REHEARSAL:-}" ]]; then
  if [[ "${PROPOSAL_UPDATE_LOSS_MODE}" == "merged_agent" ]]; then
    POST_TASK_PROPOSAL_REHEARSAL="0"
  else
    POST_TASK_PROPOSAL_REHEARSAL="1"
  fi
fi

adaptive_resolve_python

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

submit_cell() {
  local task="$1"
  local condition="$2"
  local outcome_mode="$3"
  local reward_mode="$4"
  local zero_variance="$5"
  local num_candidates="$6"
  local proposal_lr="$7"
  local task_slug
  task_slug="${task//_/-}"
  local condition_slug
  condition_slug="${condition//_/-}"
  local outcome_slug
  outcome_slug="${outcome_mode//_/-}"
  local reward_slug
  reward_slug="${reward_mode//_/-}"
  local zero_slug
  zero_slug="${zero_variance//_/-}"
  local backend_slug
  backend_slug="${CANDIDATE_EVAL_BACKEND//_/-}"
  local lr_slug
  lr_slug="${proposal_lr//./p}"
  lr_slug="${lr_slug//-/m}"
  local sweep_slug="lr-${lr_slug}"
  local out_dir="${OUT_ROOT}/${task}-${condition}-${outcome_slug}-n${num_candidates}-reward-${reward_slug}-grpo-${zero_slug}-${sweep_slug}-eval-${backend_slug}"
  local -a sbatch_resources
  sbatch_resources=(--mem "${SBATCH_MEM}")
  if [[ -n "${SBATCH_TIME}" ]]; then
    sbatch_resources+=(--time "${SBATCH_TIME}")
  fi
  if [[ -n "${SBATCH_DEPENDENCY}" ]]; then
    sbatch_resources+=(--dependency "${SBATCH_DEPENDENCY}")
  fi

  self_submit_sbatch_script \
    "dryrun-adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${reward_slug}-${zero_slug}-${sweep_slug}" \
    "adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${reward_slug}-${zero_slug}-${sweep_slug}" \
    "${LOG_DIR}/adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${reward_slug}-${zero_slug}-${sweep_slug}-%j.out" \
    "${LOG_DIR}/adaptive-cand-${task_slug}-${condition_slug}-${outcome_slug}-n${num_candidates}-${reward_slug}-${zero_slug}-${sweep_slug}-%j.err" \
    "$(self_sbatch_export_all \
      "TASK=${task}" \
      "CONDITION=${condition}" \
      "MODEL_NAME=${MODEL_NAME}" \
      "PROPOSAL_MODEL_NAME=${PROPOSAL_MODEL_NAME}" \
      "TREAT_SEED_AS_ROUND_ZERO=${TREAT_SEED_AS_ROUND_ZERO}" \
      "MAX_ATTEMPT_ROUNDS=${MAX_ATTEMPT_ROUNDS}" \
      "MAX_SELECTED_ROUNDS=${MAX_SELECTED_ROUNDS}" \
      "NO_SELECTION_PATIENCE=${NO_SELECTION_PATIENCE}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "SEED_MAX_STEPS=${SEED_MAX_STEPS}" \
      "OUTCOME_TRACE_TARGET_MODE=${outcome_mode}" \
      "PROPOSAL_GRPO_REWARD_MODE=${reward_mode}" \
      "PROPOSAL_GRPO_ZERO_VARIANCE=${zero_variance}" \
      "NUM_CANDIDATES=${num_candidates}" \
      "PROPOSAL_TEMPERATURE=${PROPOSAL_TEMPERATURE}" \
      "PROPOSAL_TOP_P=${PROPOSAL_TOP_P}" \
      "PROPOSAL_PROMPT_ACTION_HISTORY=${PROPOSAL_PROMPT_ACTION_HISTORY}" \
      "PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS=${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS}" \
      "FORCE_UNIQUE_PROPOSALS=${FORCE_UNIQUE_PROPOSALS}" \
      "PROPOSAL_UNIQUE_MAX_DRAWS=${PROPOSAL_UNIQUE_MAX_DRAWS}" \
      "OUT_DIR=${out_dir}" \
      "CANDIDATE_LOCAL_PARALLELISM=${CANDIDATE_LOCAL_PARALLELISM}" \
      "CANDIDATE_LOCAL_PACK_SIZE=${CANDIDATE_LOCAL_PACK_SIZE}" \
      "CANDIDATE_LOCAL_CACHE_BASE_STATE=${CANDIDATE_LOCAL_CACHE_BASE_STATE}" \
      "CANDIDATE_EVAL_BACKEND=${CANDIDATE_EVAL_BACKEND}" \
      "VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN}" \
      "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}" \
      "VLLM_DTYPE=${VLLM_DTYPE}" \
      "VLLM_FLASHINFER_SAMPLER=${VLLM_FLASHINFER_SAMPLER}" \
      "VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}" \
      "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}" \
      "VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}" \
      "VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}" \
      "PROPOSAL_UPDATE_LOSS_MODE=${PROPOSAL_UPDATE_LOSS_MODE}" \
      "PROPOSAL_OBSERVATION_LOSS_WEIGHT=${PROPOSAL_OBSERVATION_LOSS_WEIGHT}" \
      "PROPOSAL_FORMAT_LOSS_WEIGHT=${PROPOSAL_FORMAT_LOSS_WEIGHT}" \
      "PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES=${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES}" \
	      "PROPOSAL_GRPO_DEDUPLICATE_ACTIONS=${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS}" \
	      "PROPOSAL_GRPO_SPAN=${PROPOSAL_GRPO_SPAN}" \
	      "PROPOSAL_GRPO_LEARNING_RATE=${proposal_lr}" \
	      "PROPOSAL_GRPO_KL_COEF=${PROPOSAL_GRPO_KL_COEF}" \
	      "PROPOSAL_GRPO_NOVELTY_BONUS_BETA=${PROPOSAL_GRPO_NOVELTY_BONUS_BETA}" \
      "SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD=${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD}" \
      "PROPOSAL_UPDATE_MICROBATCH_SIZE=${PROPOSAL_UPDATE_MICROBATCH_SIZE}" \
      "PROPOSAL_SAMPLING_BATCH_SIZE=${PROPOSAL_SAMPLING_BATCH_SIZE}" \
      "POST_TASK_PROPOSAL_REHEARSAL=${POST_TASK_PROPOSAL_REHEARSAL}" \
      "ADAPTIVE_CONFIG_FILES=${ADAPTIVE_CONFIG_EXPORT}")" \
    "${sbatch_resources[@]}" \
    "${SBATCH_SCRIPT}"
}

declare -a MANIFEST_ARGS=()
for task in ${TASKS}; do
  for condition in ${CONDITIONS}; do
    for outcome_mode in ${OUTCOME_TRACE_TARGET_MODES}; do
      for reward_mode in ${PROPOSAL_GRPO_REWARD_MODES}; do
        for zero_variance in ${PROPOSAL_GRPO_ZERO_VARIANCE_MODES}; do
          for num_candidates in ${NUM_CANDIDATES_LIST}; do
            for proposal_lr in ${PROPOSAL_GRPO_LEARNING_RATES}; do
                outcome_slug="${outcome_mode//_/-}"
                reward_slug="${reward_mode//_/-}"
                zero_slug="${zero_variance//_/-}"
                lr_slug="${proposal_lr//./p}"
                lr_slug="${lr_slug//-/m}"
                sweep_slug="lr-${lr_slug}"
                job_id="$(submit_cell "${task}" "${condition}" "${outcome_mode}" "${reward_mode}" "${zero_variance}" "${num_candidates}" "${proposal_lr}")"
                MANIFEST_ARGS+=(
                  "${task}"
                  "${condition}"
                  "${outcome_mode}"
                  "${reward_mode}"
                  "${zero_variance}"
                  "${CANDIDATE_EVAL_BACKEND}"
                  "${num_candidates}"
                  "${proposal_lr}"
                  "${PROPOSAL_GRPO_KL_COEF}"
                  "${job_id}"
                  "${OUT_ROOT}/${task}-${condition}-${outcome_slug}-n${num_candidates}-reward-${reward_slug}-grpo-${zero_slug}-${sweep_slug}-eval-${CANDIDATE_EVAL_BACKEND//_/-}"
                )
            done
          done
        done
      done
    done
  done
done

MANIFEST="${OUT_ROOT}/submission_manifest.json"
"${PYTHON_BIN}" -m self.launcher_manifests adaptive-candidate \
  --manifest "${MANIFEST}" \
  --out-root "${OUT_ROOT}" \
  --tasks "${TASKS}" \
  --conditions "${CONDITIONS}" \
  --outcome-trace-target-modes "${OUTCOME_TRACE_TARGET_MODES}" \
  --proposal-grpo-reward-modes "${PROPOSAL_GRPO_REWARD_MODES}" \
  --proposal-grpo-zero-variance-modes "${PROPOSAL_GRPO_ZERO_VARIANCE_MODES}" \
  --candidate-eval-backends "${CANDIDATE_EVAL_BACKEND}" \
  --num-candidates-list "${NUM_CANDIDATES_LIST}" \
  --proposal-grpo-learning-rates "${PROPOSAL_GRPO_LEARNING_RATES}" \
  --proposal-grpo-kl-coef "${PROPOSAL_GRPO_KL_COEF}" \
  --adaptive-config-files "${ADAPTIVE_CONFIG_EXPORT}" \
  --model-name "${MODEL_NAME}" \
  --proposal-model-name "${PROPOSAL_MODEL_NAME}" \
  --job-fields "${MANIFEST_ARGS[@]}"

echo "[INFO] Submitted adaptive candidate-training jobs."
echo "[INFO] Manifest: ${MANIFEST}"

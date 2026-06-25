#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/adaptive_common.sh"
adaptive_cd_repo_root

TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/artifacts/runs/adaptive_prepared_start_ailab_${TS}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/artifacts/logs}"
DRY_RUN="${DRY_RUN:-0}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${ROOT_DIR}/launchers/self/run_adaptive_candidate_training_ailab.sbatch}"
PREPARED_START_RUN_DIRS_LIST="${PREPARED_START_RUN_DIRS_LIST:-}"
CONDITION="${CONDITION:-config}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-1.7B}"
PROPOSAL_MODEL_NAME="${PROPOSAL_MODEL_NAME:-current}"
OUTCOME_TRACE_TARGET_MODE="${OUTCOME_TRACE_TARGET_MODE:-numeric}"
PROPOSAL_GRPO_REWARD_MODE="${PROPOSAL_GRPO_REWARD_MODE:-outcome}"
PROPOSAL_GRPO_ZERO_VARIANCE="${PROPOSAL_GRPO_ZERO_VARIANCE:-skip}"
NUM_CANDIDATES="${NUM_CANDIDATES:-8}"
MAX_ATTEMPT_ROUNDS="${MAX_ATTEMPT_ROUNDS:-25}"
MAX_SELECTED_ROUNDS="${MAX_SELECTED_ROUNDS:-0}"
NO_SELECTION_PATIENCE="${NO_SELECTION_PATIENCE:-${MAX_ATTEMPT_ROUNDS}}"
MAX_STEPS="${MAX_STEPS:-100}"
SEED_MAX_STEPS="${SEED_MAX_STEPS:-0}"
SBATCH_MEM="${SBATCH_MEM:-48G}"
SBATCH_TIME="${SBATCH_TIME:-02:59:00}"
ADAPTIVE_CONFIG_EXPORT="${ADAPTIVE_CONFIG_FILES:-${ADAPTIVE_CONFIG_FILE:-${ROOT_DIR}/launchers/self/config/adaptive_candidate_base.env}}"
CANDIDATE_LOCAL_PARALLELISM="${CANDIDATE_LOCAL_PARALLELISM:-2}"
CANDIDATE_LOCAL_PACK_SIZE="${CANDIDATE_LOCAL_PACK_SIZE:-2}"
CANDIDATE_LOCAL_CACHE_BASE_STATE="${CANDIDATE_LOCAL_CACHE_BASE_STATE:-1}"
KEEP_FINAL_MODEL_CHECKPOINT="${KEEP_FINAL_MODEL_CHECKPOINT:-0}"
PROPOSAL_PROMPT_ACTION_HISTORY="${PROPOSAL_PROMPT_ACTION_HISTORY:-1}"
PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS="${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS:-5}"
FORCE_UNIQUE_PROPOSALS="${FORCE_UNIQUE_PROPOSALS:-1}"
PROPOSAL_UNIQUE_MAX_DRAWS="${PROPOSAL_UNIQUE_MAX_DRAWS:-0}"
PROPOSAL_TEMPERATURE="${PROPOSAL_TEMPERATURE:-0.9}"
PROPOSAL_TOP_P="${PROPOSAL_TOP_P:-0.95}"
PROPOSAL_SAMPLING_BATCH_SIZE="${PROPOSAL_SAMPLING_BATCH_SIZE:-8}"
PROPOSAL_OBSERVATION_LOSS_WEIGHT="${PROPOSAL_OBSERVATION_LOSS_WEIGHT:-0.2}"
PROPOSAL_FORMAT_LOSS_WEIGHT="${PROPOSAL_FORMAT_LOSS_WEIGHT:-0.02}"
PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES="${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES:-256}"
PROPOSAL_GRPO_DEDUPLICATE_ACTIONS="${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS:-1}"
PROPOSAL_GRPO_SPAN="${PROPOSAL_GRPO_SPAN:-reasoning_action}"
PROPOSAL_GRPO_LEARNING_RATE="${PROPOSAL_GRPO_LEARNING_RATE:-1e-6}"
PROPOSAL_GRPO_KL_COEF="${PROPOSAL_GRPO_KL_COEF:-0.01}"
PROPOSAL_GRPO_NOVELTY_BONUS_BETA="${PROPOSAL_GRPO_NOVELTY_BONUS_BETA:-0.05}"
SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD="${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD:-0.80}"
PROPOSAL_UPDATE_MICROBATCH_SIZE="${PROPOSAL_UPDATE_MICROBATCH_SIZE:-8}"
SYNTHETIC_PROPOSAL_SFT_EXAMPLES="${SYNTHETIC_PROPOSAL_SFT_EXAMPLES:-0}"
SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS="${SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS:-1}"
SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE="${SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE:-1e-6}"
SYNTHETIC_PROPOSAL_SFT_TOP_K="${SYNTHETIC_PROPOSAL_SFT_TOP_K:-4}"
SYNTHETIC_PROPOSAL_SFT_TEMPERATURE="${SYNTHETIC_PROPOSAL_SFT_TEMPERATURE:-0.7}"
adaptive_resolve_python

if [[ -z "${PREPARED_START_RUN_DIRS_LIST}" ]]; then
  echo "[ERROR] PREPARED_START_RUN_DIRS_LIST must contain one or more prior run directories." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

infer_task() {
  local run_name="$1"
  case "${run_name}" in
    addition-*) printf 'addition' ;;
    run_length-*) printf 'run_length' ;;
    *)
      echo "[ERROR] Cannot infer task from prepared start run name: ${run_name}" >&2
      exit 2
      ;;
  esac
}

infer_synthetic_examples() {
  local run_name="$1"
  if [[ "${run_name}" != *syn* ]]; then
    echo "[ERROR] Cannot infer synthetic amount from prepared start run name: ${run_name}" >&2
    exit 2
  fi
  printf '%s' "${run_name##*syn}"
}

submit_prepared_run() {
  local prepared_dir="$1"
  local run_name
  run_name="$(basename "${prepared_dir}")"
  local task
  task="$(infer_task "${run_name}")"
  local synthetic_examples
  synthetic_examples="$(infer_synthetic_examples "${run_name}")"
  local task_slug="${task//_/-}"
  local out_dir="${OUT_ROOT}/${task}-${CONDITION}-${OUTCOME_TRACE_TARGET_MODE}-n${NUM_CANDIDATES}-postsynthetic-syn${synthetic_examples}-25a"
  local job_slug="prepared-${task_slug}-${CONDITION}-${OUTCOME_TRACE_TARGET_MODE}-n${NUM_CANDIDATES}-postsynthetic-syn${synthetic_examples}-25a"
  local -a sbatch_resources
  sbatch_resources=(--mem "${SBATCH_MEM}" --time "${SBATCH_TIME}")

  self_submit_sbatch_script \
    "dryrun-adaptive-${job_slug}" \
    "adaptive-${job_slug}" \
    "${LOG_DIR}/adaptive-${job_slug}-%j.out" \
    "${LOG_DIR}/adaptive-${job_slug}-%j.err" \
    "$(self_sbatch_export_all \
      "TASK=${task}" \
      "CONDITION=${CONDITION}" \
      "MODEL_NAME=${MODEL_NAME}" \
      "PROPOSAL_MODEL_NAME=${PROPOSAL_MODEL_NAME}" \
      "PREPARED_START_RUN_DIR=${prepared_dir}" \
      "MAX_ATTEMPT_ROUNDS=${MAX_ATTEMPT_ROUNDS}" \
      "MAX_SELECTED_ROUNDS=${MAX_SELECTED_ROUNDS}" \
      "NO_SELECTION_PATIENCE=${NO_SELECTION_PATIENCE}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "SEED_MAX_STEPS=${SEED_MAX_STEPS}" \
      "OUTCOME_TRACE_TARGET_MODE=${OUTCOME_TRACE_TARGET_MODE}" \
      "PROPOSAL_GRPO_REWARD_MODE=${PROPOSAL_GRPO_REWARD_MODE}" \
      "PROPOSAL_GRPO_ZERO_VARIANCE=${PROPOSAL_GRPO_ZERO_VARIANCE}" \
      "NUM_CANDIDATES=${NUM_CANDIDATES}" \
      "PROPOSAL_TEMPERATURE=${PROPOSAL_TEMPERATURE}" \
      "PROPOSAL_TOP_P=${PROPOSAL_TOP_P}" \
      "PROPOSAL_PROMPT_ACTION_HISTORY=${PROPOSAL_PROMPT_ACTION_HISTORY}" \
      "PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS=${PROPOSAL_PROMPT_ACTION_HISTORY_MAX_ITEMS}" \
      "FORCE_UNIQUE_PROPOSALS=${FORCE_UNIQUE_PROPOSALS}" \
      "PROPOSAL_UNIQUE_MAX_DRAWS=${PROPOSAL_UNIQUE_MAX_DRAWS}" \
      "SYNTHETIC_PROPOSAL_SFT=0" \
      "SYNTHETIC_PROPOSAL_SFT_SEED_MIX=0" \
      "SYNTHETIC_PROPOSAL_SFT_EXAMPLES=${SYNTHETIC_PROPOSAL_SFT_EXAMPLES}" \
      "SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS=${SYNTHETIC_PROPOSAL_SFT_NUM_EPOCHS}" \
      "SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE=${SYNTHETIC_PROPOSAL_SFT_LEARNING_RATE}" \
      "SYNTHETIC_PROPOSAL_SFT_TOP_K=${SYNTHETIC_PROPOSAL_SFT_TOP_K}" \
      "SYNTHETIC_PROPOSAL_SFT_TEMPERATURE=${SYNTHETIC_PROPOSAL_SFT_TEMPERATURE}" \
      "OUT_DIR=${out_dir}" \
      "CANDIDATE_LOCAL_PARALLELISM=${CANDIDATE_LOCAL_PARALLELISM}" \
      "CANDIDATE_LOCAL_PACK_SIZE=${CANDIDATE_LOCAL_PACK_SIZE}" \
      "CANDIDATE_LOCAL_CACHE_BASE_STATE=${CANDIDATE_LOCAL_CACHE_BASE_STATE}" \
      "PROPOSAL_OBSERVATION_LOSS_WEIGHT=${PROPOSAL_OBSERVATION_LOSS_WEIGHT}" \
      "PROPOSAL_FORMAT_LOSS_WEIGHT=${PROPOSAL_FORMAT_LOSS_WEIGHT}" \
      "PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES=${PROPOSAL_FORMAT_REPLAY_MAX_EXAMPLES}" \
      "PROPOSAL_GRPO_DEDUPLICATE_ACTIONS=${PROPOSAL_GRPO_DEDUPLICATE_ACTIONS}" \
      "PROPOSAL_GRPO_SPAN=${PROPOSAL_GRPO_SPAN}" \
      "PROPOSAL_GRPO_LEARNING_RATE=${PROPOSAL_GRPO_LEARNING_RATE}" \
      "PROPOSAL_GRPO_KL_COEF=${PROPOSAL_GRPO_KL_COEF}" \
      "PROPOSAL_GRPO_NOVELTY_BONUS_BETA=${PROPOSAL_GRPO_NOVELTY_BONUS_BETA}" \
      "SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD=${SOURCE_ADMISSION_TARGET_ACCURACY_THRESHOLD}" \
      "PROPOSAL_UPDATE_MICROBATCH_SIZE=${PROPOSAL_UPDATE_MICROBATCH_SIZE}" \
      "PROPOSAL_SAMPLING_BATCH_SIZE=${PROPOSAL_SAMPLING_BATCH_SIZE}" \
      "KEEP_FINAL_MODEL_CHECKPOINT=${KEEP_FINAL_MODEL_CHECKPOINT}" \
      "ADAPTIVE_CONFIG_FILES=${ADAPTIVE_CONFIG_EXPORT}")" \
    "${sbatch_resources[@]}" \
    "${SBATCH_SCRIPT}"
}

declare -a MANIFEST_ARGS=()
for prepared_dir in ${PREPARED_START_RUN_DIRS_LIST}; do
  if [[ ! -d "${prepared_dir}" ]]; then
    echo "[ERROR] Prepared start run directory does not exist: ${prepared_dir}" >&2
    exit 2
  fi
  run_name="$(basename "${prepared_dir}")"
  task="$(infer_task "${run_name}")"
  synthetic_examples="$(infer_synthetic_examples "${run_name}")"
  out_dir="${OUT_ROOT}/${task}-${CONDITION}-${OUTCOME_TRACE_TARGET_MODE}-n${NUM_CANDIDATES}-postsynthetic-syn${synthetic_examples}-25a"
  job_id="$(submit_prepared_run "${prepared_dir}")"
  MANIFEST_ARGS+=("${task}" "${synthetic_examples}" "${prepared_dir}" "${job_id}" "${out_dir}")
done

MANIFEST="${OUT_ROOT}/submission_manifest.json"
"${PYTHON_BIN}" - "${MANIFEST}" "${OUT_ROOT}" "${MAX_ATTEMPT_ROUNDS}" "${NO_SELECTION_PATIENCE}" "${SBATCH_TIME}" "${MANIFEST_ARGS[@]}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
out_root = sys.argv[2]
max_attempts = int(sys.argv[3])
patience = int(sys.argv[4])
walltime = sys.argv[5]
fields = sys.argv[6:]
if len(fields) % 5:
    raise SystemExit("prepared-start manifest fields must be groups of 5")
jobs = {}
for index in range(0, len(fields), 5):
    task, synthetic_examples, prepared_dir, job_id, out_dir = fields[index : index + 5]
    key = f"{task}-postsynthetic-syn{synthetic_examples}-25a"
    summary_path = Path(prepared_dir) / "summary.json"
    checkpoint = None
    if summary_path.exists():
        checkpoint = json.loads(summary_path.read_text()).get("current_checkpoint")
    jobs[key] = {
        "task": task,
        "synthetic_proposal_sft_examples": int(synthetic_examples),
        "prepared_start_run_dir": prepared_dir,
        "prepared_start_checkpoint": checkpoint,
        "job_id": job_id,
        "output_dir": out_dir,
        "status": "submitted",
    }
payload = {
    "out_root": out_root,
    "max_attempt_rounds": max_attempts,
    "no_selection_patience": patience,
    "walltime": walltime,
    "jobs": jobs,
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
PY

echo "[INFO] Submitted adaptive prepared-start jobs."
echo "[INFO] Manifest: ${MANIFEST}"

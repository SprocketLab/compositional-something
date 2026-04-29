#!/usr/bin/env bash
# Curriculum self-improvement on multiplication (schoolbook CoT) with an
# addition LM driving the diamond operator.
#
# Differences from run_self_improvement.multiplication_cot.curriculum.sh:
#   - Entry point: self.self_improvement_multiplication_cot_pseudo_addition.
#   - The diamond operator (build_composed_pseudo_map) folds component
#     predictions pred_i * 10^(k-1-i) into a single y_tilde via iterative
#     pairwise addition through the addition LM at ADDITION_MODEL_PATH instead
#     of native integer summation.
#
# Initial: train on all E_{i,j} with i, j in [1, curriculum-initial-n].
# Stages (round_idx -> stage):
#   for target_n = initial_n+1 .. max_n:
#     for m = 1 .. target_n: stage introduces (target_n, m) and (m, target_n).
# Decomposition: schoolbook along the shorter dimension (n>=m -> split b into m
# digits -> m components of E_{n,1}; n<m -> split a into n digits -> n
# components of E_{1,m}). CoT is only trained for shapes with both digits >= 2.
#
# Each round trains on D_0 (initial) + D_1 + ... + D_{r-1} with D_{r-1}
# up-sampled to 50% via --curriculum-recent-fraction.

set -euo pipefail

INITIAL_N=${INITIAL_N:-2}
MAX_N=${MAX_N:-6}
INITIAL_PER_SHAPE=${INITIAL_PER_SHAPE:-400}
EXPAND_PER_SHAPE=${EXPAND_PER_SHAPE:-4000}
EVAL_PER_SHAPE=${EVAL_PER_SHAPE:-50}
COMPOSED_EVAL_PER_SHAPE=${COMPOSED_EVAL_PER_SHAPE:-50}
RECENT_FRACTION=${RECENT_FRACTION:-0.5}
BATCH=${BATCH:-16}
EPOCHS=${EPOCHS:-3}
SEED=${SEED:-42}
ADDITION_MODEL_PATH=${ADDITION_MODEL_PATH:-models/addition_self_improvement_round8}
ADDITION_BATCH=${ADDITION_BATCH:-128}
ADDITION_MAX_NEW_TOKENS=${ADDITION_MAX_NEW_TOKENS:-64}
# Pick up the visible-GPU count automatically if NUM_GPUS isn't set.
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS=$(nvidia-smi -L | wc -l)
    else
        NUM_GPUS=1
    fi
fi

TAG="curriculum_n${INITIAL_N}to${MAX_N}_init${INITIAL_PER_SHAPE}_stage${EXPAND_PER_SHAPE}_recent${RECENT_FRACTION}_addmodel"
OUT_DIR="artifacts/runs/self_improvement_multiplication_cot_${TAG}"
LOG_DIR="./.log"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/self_improvement_multiplication_cot.${TAG}.log"

# DDP via torchrun when more than one GPU is visible — eval/component-prediction
# generation shards across ranks (see core/multiplication_pipeline_pseudo_addition.py),
# so throughput scales with NUM_GPUS instead of stalling on cuda:0.
if [[ "${NUM_GPUS}" -gt 1 ]]; then
    LAUNCHER=(torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}")
else
    LAUNCHER=(python -u)
fi

"${LAUNCHER[@]}" -m self.self_improvement_multiplication_cot_pseudo_addition \
    --bf16 \
    --per-device-train-batch-size "${BATCH}" \
    --per-device-eval-batch-size "${BATCH}" \
    --num-epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --curriculum-mode \
    --curriculum-initial-n "${INITIAL_N}" \
    --curriculum-max-n "${MAX_N}" \
    --curriculum-recent-fraction "${RECENT_FRACTION}" \
    --initial-train-per-digit "${INITIAL_PER_SHAPE}" \
    --initial-eval-per-digit "${EVAL_PER_SHAPE}" \
    --expand-train-per-digit "${EXPAND_PER_SHAPE}" \
    --eval-per-digit "${EVAL_PER_SHAPE}" \
    --composed-eval-per-digit "${COMPOSED_EVAL_PER_SHAPE}" \
    --addition-model-path "${ADDITION_MODEL_PATH}" \
    --addition-model-batch-size "${ADDITION_BATCH}" \
    --addition-model-max-new-tokens "${ADDITION_MAX_NEW_TOKENS}" \
    --output-dir "${OUT_DIR}" \
    2>&1 | tee "${LOG_FILE}"

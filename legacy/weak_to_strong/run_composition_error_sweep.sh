#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

PROPORTIONS=(0 10 20 30 40 50 60 70 80 90 100)

# Customize additional arguments for each experiment as needed.
W2S_ARGS=()
SELF_IMPROVEMENT_ARGS=()

run_w2s_queue() {
    for percent in "${PROPORTIONS[@]}"; do
        echo "[INFO] Launching weak_to_strong (GPU0) with composition error percent=${percent}"
        w2s_output_dir="${ROOT_DIR}/artifacts/runs/legacy_w2s/w2s_err_${percent}"
        if [ "${#W2S_ARGS[@]}" -gt 0 ]; then
            CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/weak_to_strong_composition_error_experiment.py" \
                --composition-error-percent "${percent}" \
                -- --output-dir "${w2s_output_dir}" "${W2S_ARGS[@]}" &
        else
            CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/weak_to_strong_composition_error_experiment.py" \
                --composition-error-percent "${percent}" \
                -- --output-dir "${w2s_output_dir}" &
        fi
        wait "$!"
    done
}

run_self_queue() {
    for percent in "${PROPORTIONS[@]}"; do
        echo "[INFO] Launching self_improvement (GPU7) with composition error percent=${percent}"
        self_output_dir="${ROOT_DIR}/artifacts/runs/self_improvement/error_${percent}"
        if [ "${#SELF_IMPROVEMENT_ARGS[@]}" -gt 0 ]; then
            CUDA_VISIBLE_DEVICES=7 python -m self.experiments.composition_error_sweep \
                --composition-error-percent "${percent}" \
                -- --output-dir "${self_output_dir}" "${SELF_IMPROVEMENT_ARGS[@]}" &
        else
            CUDA_VISIBLE_DEVICES=7 python -m self.experiments.composition_error_sweep \
                --composition-error-percent "${percent}" \
                -- --output-dir "${self_output_dir}" &
        fi
        wait "$!"
    done
}

run_w2s_queue &
w2s_queue_pid=$!

run_self_queue &
self_queue_pid=$!

wait "${w2s_queue_pid}"
wait "${self_queue_pid}"

echo "[INFO] Completed composition error sweep."

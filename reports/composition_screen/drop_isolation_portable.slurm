#!/bin/bash
#SBATCH --job-name=drop-isolation
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=8:00:00
# Week 0 gate, first half: the four-arm screen on the BASE model (plan §4).
# Expect this to be uninformative -- MuSiQue's base screen was +.023 with a CI
# spanning zero -- and read the verdict from drop_seed instead.  Submit with:
#   source reports/composition_screen/cluster_env.sh
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/drop_isolation_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)

# Pass ADAPTER_DIR to screen a trained checkpoint instead of the base model.
ADAPTER_ARG=""
if [[ -n "${ADAPTER_DIR:-}" ]]; then
  ADAPTER_ARG="--adapter $ADAPTER_DIR"
fi

"$PY" "$R"/drop_isolation.py \
  --model "$MODEL" \
  $ADAPTER_ARG \
  --dev-data "$R"/data/drop_qdmr_dev.jsonl \
  --dev-gold "$R"/data/drop_gold_dev.jsonl \
  --atom-pool "$R"/data/drop_atom_pool_dev.jsonl \
  --out-dir "$R"/drop_isolation \
  --cells 2 3 4 --per-k "${PER_K:-100}" --batch-size "${BATCH_SIZE:-16}" --seed 7

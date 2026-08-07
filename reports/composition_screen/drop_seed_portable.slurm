#!/bin/bash
#SBATCH --job-name=drop-seed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=24:00:00
# Week 0 gate, second half: train the extraction seed and rerun the screen
# (plan §4, §7.1).  This job carries the verdict -- the base screen is expected
# to be uninformative, as MuSiQue's was.  Submit with:
#   source reports/composition_screen/cluster_env.sh
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/drop_seed_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)

# GRAD_CKPT comes from cluster_env.sh; it is required on 48 GB cards.
CKPT_ARG=""
if [[ "${GRAD_CKPT:-0}" == "1" ]]; then
  CKPT_ARG="--gradient-checkpointing"
fi

"$PY" "$R"/drop_seed.py \
  --model "$MODEL" \
  --seed-pool "$R"/data/drop_seed_pool_train.jsonl \
  --train-data "$R"/data/drop_qdmr_train.jsonl \
  --dev-data "$R"/data/drop_qdmr_dev.jsonl \
  --dev-gold "$R"/data/drop_gold_dev.jsonl \
  --atom-pool "$R"/data/drop_atom_pool_dev.jsonl \
  --out-dir "$R"/drop_seed \
  --train-size "${TRAIN_SIZE:-4000}" --max-steps "${MAX_STEPS:-300}" \
  --max-length 2048 --micro-batch-size "${MICRO_BATCH:-1}" \
  --cells 2 3 4 --per-k "${PER_K:-100}" --batch-size "${BATCH_SIZE:-16}" \
  --seed 7 $CKPT_ARG

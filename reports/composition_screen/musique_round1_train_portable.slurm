#!/bin/bash
#SBATCH --job-name=musique-r1-train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
# Round 1 stage 2: continue from the r1seed adapter on pseudo-labels.
# LABEL_KEY selects the arm (composed | direct | gold). Submit with:
#   LABEL_KEY=composed sbatch --dependency=afterok:<label-job> \
#          $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/musique_round1_train_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
ARM=${LABEL_KEY:-composed}
GC_FLAG=""
[ "${GRAD_CKPT:-0}" = "1" ] && GC_FLAG="--gradient-checkpointing"

"$PY" "$R"/musique_round1_train.py \
  --model "$MODEL" \
  --adapter "$R"/musique_seed_r1seed/adapter \
  --labels "$R"/musique_round1/round1_labels.jsonl \
  --label-key "$ARM" \
  --train-data "$R"/data/musique_train.jsonl \
  --dev-data "$R"/data/musique_dev.jsonl \
  --before-per-instance "$R"/musique_seed_r1seed/per_instance.jsonl \
  --out-dir "$R"/musique_round1_"$ARM" \
  --max-steps 300 --learning-rate 1e-4 \
  --max-length 4096 --micro-batch-size 1 --per-k 100 --batch-size 8 --seed 7 \
  $GC_FLAG

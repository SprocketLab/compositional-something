#!/bin/bash
#SBATCH --job-name=musique-seed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
# Site values (REPO, PY, HF_HOME, GRAD_CKPT) come from cluster_env.sh, exported
# through sbatch. Submit with:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/musique_seed_portable.slurm
#
# Writes to musique_seed_$SITE_TAG (default: hostname's cluster name) so the
# Della-trained adapter in musique_seed/ is never overwritten.
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
GC_FLAG=""
[ "${GRAD_CKPT:-0}" = "1" ] && GC_FLAG="--gradient-checkpointing"
# Overridable for variant seeds, e.g. Round 1's partition seed:
#   TRAIN_DATA=.../musique_train_seed30.jsonl OUT_TAG=r1seed sbatch ...
TRAIN=${TRAIN_DATA:-"$R"/data/musique_train.jsonl}
OUT="$R"/musique_seed_${OUT_TAG:-${SITE_TAG:-neuronic}}

"$PY" "$R"/musique_seed.py \
  --model "$MODEL" \
  --train-data "$TRAIN" \
  --dev-data "$R"/data/musique_dev.jsonl \
  --out-dir "$OUT" \
  --train-size 4000 --max-steps 300 --learning-rate 2e-4 \
  --max-length 4096 --micro-batch-size 1 --per-k 100 --batch-size 8 \
  $GC_FLAG

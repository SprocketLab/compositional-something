#!/bin/bash
#SBATCH --job-name=musique-r1-label
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=24:00:00
# Round 1 stage 1: pseudo-label 2-hop composition-source questions with the
# partition seed's chain. Depends on the r1seed adapter existing. Submit with:
#   sbatch --dependency=afterok:<r1seed-job> \
#          $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/musique_round1_label_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
# Later rounds override via env, e.g. Round 2:
#   ROUND=2 HOPS=3 ADAPTER_DIR=$R/musique_round1_composed_rehearsal/adapter sbatch ...
ROUND=${ROUND:-1}
HOPS=${HOPS:-2}
ADAPTER_DIR=${ADAPTER_DIR:-"$R"/musique_seed_r1seed/adapter}

"$PY" "$R"/musique_round1_label.py \
  --model "$MODEL" \
  --adapter "$ADAPTER_DIR" \
  --train-data "$R"/data/musique_train.jsonl \
  --partition "$R"/data/musique_round1_partition.json \
  --out-dir "$R"/musique_round"$ROUND" \
  --hops "$HOPS" \
  --pool-size 5000 --batch-size 16 --seed 7

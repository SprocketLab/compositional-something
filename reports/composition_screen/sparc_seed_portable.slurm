#!/bin/bash
#SBATCH --job-name=sparc-seed
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
# SParC seed: turn-1 + Spider-easy on seed-source dbs (plan §12). Submit with:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/sparc_seed_portable.slurm
# Env overrides: OUT_TAG (default sparc_seed), LIMIT=8 for a smoke run.
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
GC_FLAG=""
[ "${GRAD_CKPT:-0}" = "1" ] && GC_FLAG="--gradient-checkpointing"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

"$PY" "$R"/sparc_seed.py \
  --model "$MODEL" \
  --sequences "$R"/data/sparc_sequences.jsonl \
  --gold "$R"/data/sparc_gold.jsonl \
  --spider-singles "$R"/data/sparc_spider_singles.jsonl \
  --spider-gold "$R"/data/sparc_gold_spider.jsonl \
  --schemas "$R"/data/sparc_schemas.json \
  --partition "$R"/data/sparc_partition.json \
  --db-root "$R"/data/sparc_raw/spider/database \
  --testsuite-root "$R"/data/sparc_raw/testsuite \
  --out-dir "$R"/"${OUT_TAG:-sparc_seed}" \
  --train-size 3000 --max-steps 300 --learning-rate 2e-4 \
  --max-length 4096 --micro-batch-size 1 --per-cell 100 --batch-size 8 \
  --seed 7 $GC_FLAG $LIMIT_FLAG

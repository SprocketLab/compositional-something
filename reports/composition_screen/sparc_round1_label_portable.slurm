#!/bin/bash
#SBATCH --job-name=sparc-r1-label
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
# Round 1 stage 1: pseudo-label with the seed's guarded chain. Submit with:
#   sbatch --dependency=afterok:<seed-job> \
#          $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/sparc_round1_label_portable.slurm
# Env overrides: TURNS (default 2), ACCEPT_LEVEL (default L4), ADAPTER_DIR,
#                ROUND (out-dir suffix), LIMIT=8 for a smoke run.
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
TURNS=${TURNS:-2}
ROUND=${ROUND:-1}
ADAPTER_DIR=${ADAPTER_DIR:-"$R"/sparc_seed/adapter}
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

"$PY" "$R"/sparc_round1_label.py \
  --model "$MODEL" \
  --adapter "$ADAPTER_DIR" \
  --sequences "$R"/data/sparc_sequences.jsonl \
  --gold "$R"/data/sparc_gold.jsonl \
  --schemas "$R"/data/sparc_schemas.json \
  --tables "$R"/data/sparc_raw/spider/tables.json \
  --partition "$R"/data/sparc_partition.json \
  --db-root "$R"/data/sparc_raw/spider/database \
  --testsuite-root "$R"/data/sparc_raw/testsuite \
  --out-dir "$R"/sparc_round"$ROUND" \
  --pool-size 3000 --turns "$TURNS" --accept-level "${ACCEPT_LEVEL:-L4}" \
  --batch-size 16 --chunk 512 --seed 7 $LIMIT_FLAG

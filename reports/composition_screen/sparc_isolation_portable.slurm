#!/bin/bash
#SBATCH --job-name=sparc-isolation
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:00:00
# The pre-registered gate (plan §4). Submit with:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/sparc_isolation_portable.slurm
# Env overrides:
#   ADAPTER=<dir>          seeded re-screen (same script, plan §4)
#   MODEL_OVERRIDE=<path>  the 1.7B record run (plan §14)
#   LIMIT=8                smoke run
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=${MODEL_OVERRIDE:-$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)}
TAG=${OUT_TAG:-sparc_isolation}
ADAPTER_FLAG=""
[ -n "${ADAPTER:-}" ] && ADAPTER_FLAG="--adapter $ADAPTER"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

"$PY" "$R"/sparc_isolation.py \
  --model "$MODEL" \
  --sequences "$R"/data/sparc_sequences.jsonl \
  --gold "$R"/data/sparc_gold.jsonl \
  --schemas "$R"/data/sparc_schemas.json \
  --db-root "$R"/data/sparc_raw/spider/database \
  --testsuite-root "$R"/data/sparc_raw/testsuite \
  --out "$R"/"$TAG".json \
  --per-cell 100 --batch-size 16 --max-new-tokens 128 --seed 7 \
  $ADAPTER_FLAG $LIMIT_FLAG

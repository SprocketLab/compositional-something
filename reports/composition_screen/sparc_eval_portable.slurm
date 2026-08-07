#!/bin/bash
#SBATCH --job-name=sparc-eval
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=04:00:00
# Frozen-model baselines B0/B2/B6/B8 (plan §15). Submit with:
#   MODE=chain sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/sparc_eval_portable.slurm
# Env overrides: MODE (single|seq|chain|chain-goldprefix), SLICE (audit|dev),
#                ADAPTER (default seed adapter), LIMIT=8 for a smoke run.
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
MODE=${MODE:-chain}
SLICE=${SLICE:-audit}
ADAPTER=${ADAPTER:-"$R"/sparc_seed/adapter}
ADAPTER_FLAG=""
[ "$ADAPTER" != "none" ] && ADAPTER_FLAG="--adapter $ADAPTER"
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

"$PY" "$R"/sparc_eval.py \
  --model "$MODEL" \
  --mode "$MODE" --slice "$SLICE" \
  --sequences "$R"/data/sparc_sequences.jsonl \
  --gold "$R"/data/sparc_gold.jsonl \
  --schemas "$R"/data/sparc_schemas.json \
  --partition "$R"/data/sparc_partition.json \
  --db-root "$R"/data/sparc_raw/spider/database \
  --testsuite-root "$R"/data/sparc_raw/testsuite \
  --out-dir "$R"/sparc_baselines \
  --per-cell 100 --batch-size 16 --seed 7 \
  $ADAPTER_FLAG $LIMIT_FLAG

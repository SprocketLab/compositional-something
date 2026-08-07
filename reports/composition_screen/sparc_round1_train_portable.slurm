#!/bin/bash
#SBATCH --job-name=sparc-r1-train
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G
#SBATCH --time=08:00:00
# Round 1 stage 2: continue from the seed adapter on pseudo-labels.
# LABEL_KEY selects the arm (composed | direct | gold). Submit with:
#   LABEL_KEY=composed sbatch --dependency=afterok:<label-job> \
#          $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/sparc_round1_train_portable.slurm
# REHEARSAL_FRAC=0.25 mixes turn-mode rehearsal and suffixes the out-dir.
# Later rounds override ROUND / ADAPTER_DIR / BEFORE_PI / LABELS / REPLAY_*.
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
RH=${REHEARSAL_FRAC:-0}
ROUND=${ROUND:-1}
ADAPTER_DIR=${ADAPTER_DIR:-"$R"/sparc_seed/adapter}
BEFORE_PI=${BEFORE_PI:-"$R"/sparc_seed/per_instance.jsonl}
OUT="$R"/sparc_round"$ROUND"_"$ARM"
RH_FLAG=""
if [ "$RH" != 0 ]; then RH_FLAG="--rehearsal-frac $RH"; OUT="${OUT}_rehearsal"; fi
REPLAY_FLAG=""
if [ -n "${REPLAY_FRAC:-}" ]; then
  REPLAY_FLAG="--replay-data ${REPLAY_DATA:-"$R"/sparc_seed/seed_examples.jsonl} --replay-frac $REPLAY_FRAC"
fi
if [ -n "${REPLAY_LABELS:-}" ]; then
  REPLAY_FLAG="$REPLAY_FLAG --replay-labels $REPLAY_LABELS"
fi
LIMIT_FLAG=""
[ -n "${LIMIT:-}" ] && LIMIT_FLAG="--limit $LIMIT"

"$PY" "$R"/sparc_round1_train.py \
  --model "$MODEL" \
  --adapter "$ADAPTER_DIR" \
  --labels "${LABELS:-"$R"/sparc_round1/round1_labels.jsonl}" \
  --label-key "$ARM" \
  --sequences "$R"/data/sparc_sequences.jsonl \
  --gold "$R"/data/sparc_gold.jsonl \
  --schemas "$R"/data/sparc_schemas.json \
  --partition "$R"/data/sparc_partition.json \
  --db-root "$R"/data/sparc_raw/spider/database \
  --testsuite-root "$R"/data/sparc_raw/testsuite \
  --before-per-instance "$BEFORE_PI" \
  --out-dir "$OUT" \
  --max-steps 300 --learning-rate 1e-4 \
  --max-length 4096 --micro-batch-size 1 --per-cell 100 --batch-size 8 \
  --seed 7 $GC_FLAG $RH_FLAG $REPLAY_FLAG $LIMIT_FLAG

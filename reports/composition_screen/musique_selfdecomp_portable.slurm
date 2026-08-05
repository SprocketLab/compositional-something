#!/bin/bash
#SBATCH --job-name=musique-selfdecomp
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00
# Self-proposed decomposition arm (plan §4.2 / handoff §5a). Needs the seed
# run's adapter and per_instance.jsonl from THIS cluster -- run
# musique_seed_portable.slurm first. Submit with:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#          reports/composition_screen/musique_selfdecomp_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
SEED_DIR="$R"/musique_seed_${SITE_TAG:-neuronic}
# PROMPT_STYLE=v2 sbatch ... runs the tightened prompt into its own out-dir.
STYLE=${PROMPT_STYLE:-v1}
OUT="$R"/musique_selfdecomp
[ "$STYLE" != v1 ] && OUT="${OUT}_${STYLE}"

"$PY" "$R"/musique_selfdecomp.py \
  --model "$MODEL" \
  --adapter "$SEED_DIR"/adapter \
  --dev-data "$R"/data/musique_dev.jsonl \
  --seed-per-instance "$SEED_DIR"/per_instance.jsonl \
  --out-dir "$OUT" \
  --decompose-with base --prompt-style "$STYLE" \
  --per-k 100 --batch-size 8 --seed 7

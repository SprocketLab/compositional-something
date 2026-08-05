#!/bin/bash
#SBATCH --job-name=musique-seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
# partition / gres / cpus / mem come from cluster_env.sh via sbatch flags:
#   sbatch $(bash reports/composition_screen/sbatch_flags.sh) musique_seed_portable.slurm

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/cluster_env.sh"
cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$HERE/logs"

MODEL="$(resolve_model)" || exit 1
echo "repo=$REPO"; echo "python=$PY"; echo "model=$MODEL"; echo "grad_ckpt=$GRAD_CKPT"

CKPT_FLAG=""
[ "$GRAD_CKPT" = "1" ] && CKPT_FLAG="--gradient-checkpointing"

"$PY" "$HERE/musique_seed.py" \
  --model "$MODEL" \
  --train-data "$HERE/data/musique_train.jsonl" \
  --dev-data "$HERE/data/musique_dev.jsonl" \
  --out-dir "$HERE/musique_seed" \
  --train-size 4000 --max-steps 300 --learning-rate 2e-4 \
  --max-length 4096 --micro-batch-size 1 --per-k 100 --batch-size 8 \
  $CKPT_FLAG

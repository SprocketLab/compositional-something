#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G
#SBATCH --time=00:45:00
# Site values (REPO, PY, HF_HOME) come from cluster_env.sh, exported through
# sbatch. Submit with:  sbatch $(bash reports/composition_screen/sbatch_flags.sh) \
#                              reports/composition_screen/memprobe_portable.slurm
set -euo pipefail
cd "$REPO"
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=$PWD/reports/composition_screen
MODEL=$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.5-4B/snapshots/*/ | head -1)
"$PY" "$R"/memprobe.py --model "$MODEL" --label Qwen3.5-4B

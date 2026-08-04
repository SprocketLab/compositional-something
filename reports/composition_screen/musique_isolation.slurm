#!/bin/bash
#SBATCH --job-name=musique-iso
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/musiso-%j.out
#SBATCH --error=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/musiso-%j.err

set -euo pipefail
cd /scratch/gpfs/BRENDEN/changho/compositional-something
export PYTHONPATH=$PWD
export HF_HOME=/scratch/gpfs/BRENDEN/changho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=$PWD/reports/composition_screen

/home/cs1095/.conda/envs/torch-env/bin/python $R/musique_isolation.py \
  --model /scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --data $R/data/musique_dev.jsonl \
  --out $R/musique_isolation.json \
  --per-k 100 --batch-size 16

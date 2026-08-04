#!/bin/bash
#SBATCH --job-name=musique-seed
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/musseed-%j.out
#SBATCH --error=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/musseed-%j.err

set -euo pipefail
cd /scratch/gpfs/BRENDEN/changho/compositional-something
export PYTHONPATH=$PWD
export HF_HOME=/scratch/gpfs/BRENDEN/changho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
R=$PWD/reports/composition_screen

/home/cs1095/.conda/envs/torch-env/bin/python $R/musique_seed.py \
  --model /scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --train-data $R/data/musique_train.jsonl \
  --dev-data $R/data/musique_dev.jsonl \
  --out-dir $R/musique_seed \
  --train-size 4000 --max-steps 300 --learning-rate 2e-4 \
  --max-length 4096 --micro-batch-size 1 --per-k 100 --batch-size 8

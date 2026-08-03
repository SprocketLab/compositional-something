#!/bin/bash
#SBATCH --job-name=clutrr-gateb
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/gateb-%j.out
#SBATCH --error=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/gateb-%j.err

set -euo pipefail
cd /scratch/gpfs/BRENDEN/changho/compositional-something
export PYTHONPATH=$PWD
export HF_HOME=/scratch/gpfs/BRENDEN/changho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
R=$PWD/reports/composition_screen

/home/cs1095/.conda/envs/torch-env/bin/python $R/gate_b_extraction.py \
  --model /scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --adapter $R/clutrr_seed/adapter \
  --pool-csv $R/data/unlabeled_pool.csv \
  --out $R/gate_b_extraction.json \
  --per-k 60 --max-hop 4 --batch-size 24

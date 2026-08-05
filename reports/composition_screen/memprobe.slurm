#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --partition=ailab
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/memprobe-%j.out
#SBATCH --error=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/memprobe-%j.err
set -euo pipefail
cd /scratch/gpfs/BRENDEN/changho/compositional-something
export PYTHONPATH=$PWD HF_HOME=/scratch/gpfs/BRENDEN/changho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/cs1095/.conda/envs/torch-env/bin/python /scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/memprobe.py

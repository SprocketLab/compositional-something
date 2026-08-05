#!/bin/bash
#SBATCH --job-name=memprobe
#SBATCH --partition=ailab
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=128G --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/memprobe-%j.out
#SBATCH --error=/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/memprobe-%j.err
set -euo pipefail
cd /scratch/gpfs/BRENDEN/changho/compositional-something
export PYTHONPATH=$PWD HF_HOME=/scratch/gpfs/BRENDEN/changho/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/cs1095/.conda/envs/torch-env/bin/python
R=$PWD/reports/composition_screen
$P $R/memprobe.py --model /scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --label Qwen3.5-4B
echo "================================================================"
$P $R/memprobe.py --model /scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --label Qwen3-8B

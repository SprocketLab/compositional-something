# Site-specific values -- the ONLY file that changes between clusters.
# Source this before submitting: its exports reach the job via sbatch's
# default --export=ALL, so the *_portable.slurm scripts stay site-agnostic.
#
# Current values: neuronic (Princeton CS), L40 48 GB, 2026-08-04.
export REPO=/n/fs/cogai/cs1095/compositional-something
export PY=/n/fs/cogai/cs1095/venvs/compositional/bin/python  # torch 2.13.0+cu130, transformers 5.14.1, peft 0.20.0 (uma-transformer-l40's transformers 4.57.6 predates qwen3_5)
export HF_HOME=/n/fs/cogai/cs1095/hf_cache
export PARTITION=all
export GRES=gpu:1            # every node is gpu:l40:8
export GRAD_CKPT=1           # required on 48 GB: seq-4096 training is 86.6 GiB without it

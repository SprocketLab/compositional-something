# GPU memory profile — 4B LoRA, measured 2026-08-04

Measured for a possible move to 48 GB cards. Qwen3.5-4B, LoRA r16/a32 all-linear,
bf16, `micro_batch_size=1`, AdamW. Reproduce with
`sbatch reports/composition_screen/memprobe.slurm` (~2 min).

Figures are `torch.cuda.max_memory_allocated`, so they **exclude** allocator
fragmentation and CUDA context (~0.5–1 GiB). Score against ~44 GiB, not 48.

| configuration | peak (GiB) | fits 48 GB |
|---|---:|---|
| weights + LoRA resident | 8.0 | yes |
| train seq 1024, no checkpointing | 28.0 | yes |
| train seq 1024, gradient checkpointing | 11.4 | yes |
| train seq 2048, no checkpointing | 47.5 | **no** (zero margin) |
| train seq 2048, gradient checkpointing | 14.4 | yes |
| train seq 4096, no checkpointing | **86.6** | **no** |
| train seq 4096, gradient checkpointing | **20.4** | yes |
| generate bs 8 @ 3000 tok | 13.9 | yes |
| generate bs 16 @ 3000 tok | 19.7 | yes |

Activations dominate: ~20 GiB per 1024 tokens unchecked, ~3 GiB per 1024
checkpointed. At 4096 tokens checkpointing is a 4.2x reduction.

## Consequence

`train_lora` does not set `gradient_checkpointing` (`self/coding/training.py:181`),
so it defaults off. On 48 GB cards that caps training at seq 1024. Enabling it
makes every current configuration fit with room for `micro_batch_size=2`.
Suggested change: add it as an explicit `train_lora` parameter defaulting to
off, so existing runs stay bit-comparable and launchers opt in.

Throughput cost is real but unmeasured here — the probe's 2-step runs are too
short to time. Budget 20–40%.

## Not measured

* **Qwen3-8B.** Weights alone are ~16 GiB. Checkpointed at 4096 it is plausibly
  high-30s GiB, but this is a guess. Used in only 2 files.
* 10 GB MIG slices (`gpu:1g.10gb`) remain viable only for 0.6B/1.7B work; 4B
  needs 11.4 GiB even checkpointed at seq 1024.

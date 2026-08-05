# GPU memory profile — measured 2026-08-04

Measured for a move to 48 GB cards (2 available per job). Qwen3.5-4B and
Qwen3-8B, LoRA r16/a32 all-linear, bf16, AdamW. Reproduce with
`sbatch reports/composition_screen/memprobe.slurm` (~3 min for both models).

Figures are `torch.cuda.max_memory_allocated`, so they **exclude** allocator
fragmentation and CUDA context (~0.5–1 GiB). Scored against 44 GiB, not 48.

## Qwen3.5-4B (4.24B params, 32.5M trainable LoRA, 8.0 GiB resident)

| configuration | peak (GiB) | fits |
|---|---:|---|
| train seq 1024, mb 1, no checkpointing | 28.0 | yes |
| train seq 1024, mb 1, **checkpointing** | 11.4 | yes |
| train seq 1024, mb 2, checkpointing | 14.3 | yes |
| train seq 2048, mb 1, no checkpointing | 47.5 | **no** |
| train seq 2048, mb 1, **checkpointing** | 14.4 | yes |
| train seq 2048, mb 2, checkpointing | 20.4 | yes |
| train seq 4096, mb 1, no checkpointing | **86.6** | **no** |
| train seq 4096, mb 1, **checkpointing** | 20.4 | yes |
| train seq 4096, mb 2, checkpointing | 32.4 | yes |
| generate bs 8 @ 3000 tok | 13.9 | yes |
| generate bs 16 @ 3000 tok | 19.7 | yes |

## Qwen3-8B (8.23B params, 43.6M trainable LoRA, 15.4 GiB resident)

| configuration | peak (GiB) | fits |
|---|---:|---|
| train seq 1024, mb 1, no checkpointing | 25.3 | yes |
| train seq 1024, mb 1, **checkpointing** | 18.0 | yes |
| train seq 2048, mb 1, no checkpointing | 34.7 | yes |
| train seq 2048, mb 1, **checkpointing** | 20.0 | yes |
| train seq 4096, mb 1, no checkpointing | 53.5 | **no** |
| train seq 4096, mb 1, **checkpointing** | 24.1 | yes |
| train seq 4096, mb 2, checkpointing | 32.2 | yes |
| generate bs 8 @ 3000 tok | 23.1 | yes |
| generate bs 16 @ 3000 tok | 30.8 | yes |

## Verdict

**Everything fits on one 48 GB card with gradient checkpointing, including 8B.**
Without it, training caps at seq 1024 (4B) or seq 2048 (8B).

`train_lora` now takes `gradient_checkpointing: bool = False` — default off so
existing runs stay bit-comparable, launchers opt in. It uses the non-reentrant
path; the reentrant one silently fails to propagate gradients to LoRA adapters
when every base weight is frozen.

## The 4B is heavier than the 8B unchecked, and that is fixable

28.0 vs 25.3 GiB at seq 1024. Three causes, from `config.json`:

* **vocab 248,320** against the 8B's 151,936. The fp32 logits tensor at seq 4096
  is ~4.1 GB versus ~2.5 GB, and at long context it dominates.
* **hybrid linear attention** — `layer_types` runs three `linear_attention`
  layers per `full_attention` one (`full_attention_interval: 4`). Neither `fla`
  nor `causal_conv1d` is installed, so it takes the torch fallback, which is
  what emits the "fast path is not available" warning in every log. The fallback
  is memory-hungry and slow.
* **multimodal** — the config carries a `vision_config` and image/video token
  ids. A vision tower is loaded that no experiment here uses.

**Installing `flash-linear-attention` and `causal-conv1d` on the new cluster
should cut both memory and step time for every Qwen3.5 run**, and may offset
checkpointing's 20–40% slowdown. Untested here; worth measuring once installed.

## Two 48 GB GPUs

Only one of the two ways to use them helps with memory:

* **DDP** — each card holds a full model copy, so per-GPU memory is unchanged.
  Buys ~2x throughput; this is the right default, since memory already fits.
  `Trainer` does this automatically under `torchrun`, but no launcher here uses
  torchrun, and DDP + LoRA + checkpointing needs
  `ddp_find_unused_parameters=False` or it can hang. Real integration work.
* **Sharding** (FSDP or `device_map="auto"`) — halves the weight footprint.
  Not needed: 8B at seq 4096 checkpointed is 24.1 GiB on a single card.

Raising `micro_batch_size` is the cheaper throughput win to try first: mb 2 at
seq 4096 costs 32.4 GiB (4B) / 32.2 GiB (8B) and still fits.

## Caveat on an earlier version of this file

The probe reuses one model object across cases, and
`TrainingArguments(gradient_checkpointing=False)` does not *disable*
checkpointing on a model that already has it on. An earlier revision therefore
reported every "no checkpointing" case after the first as the checkpointed
figure (seq4096-OFF as 20.4 instead of 86.6). `memprobe.py` now forces the model
state explicitly before each case.

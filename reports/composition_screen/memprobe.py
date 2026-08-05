"""Peak GPU memory for the configurations this project actually runs.

Answers whether the work fits on a 48 GB card, and what has to change.  Each
case is measured from a fresh allocator state via reset_peak_memory_stats.

Gradient checkpointing is driven through `train_lora`'s own parameter rather
than by toggling the model directly, so this doubles as a regression test that
the parameter is wired up and that LoRA gradients still flow under it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/scratch/gpfs/BRENDEN/changho/compositional-something")

from self.coding.atomic_data import AtomicExample
from self.coding.training import (
    generate_predictions,
    load_qwen_lora_model,
    load_qwen_tokenizer,
    train_lora,
)

GB = 2 ** 30
OUT = Path("/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/probe")
FILLER = "The quick brown fox jumps over the lazy dog. " * 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--budget", type=float, default=44.0,
                    help="usable GiB on the target card (48 minus context/fragmentation)")
    args = ap.parse_args()

    tok = load_qwen_tokenizer(args.model)
    model = load_qwen_lora_model(args.model)
    results = {"weights_only": round(torch.cuda.memory_allocated() / GB, 1)}
    print(f"[{args.label}] weights+LoRA resident: {results['weights_only']:.1f} GiB", flush=True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{args.label}] params {total/1e9:.2f}B, trainable (LoRA) {trainable/1e6:.1f}M", flush=True)
    results["params_B"] = round(total / 1e9, 2)

    def make(n_tokens: int, count: int):
        text = tok.decode(tok(FILLER * 12)["input_ids"][: n_tokens - 40])
        return [AtomicExample(task="probe", source_id=str(i), source_group_id=str(i),
                              split="x", messages=({"role": "user", "content": text},),
                              target="answer", evaluator={}, metadata={})
                for i in range(count)]

    def probe(name, fn, want_checkpointing):
        # TrainingArguments(gradient_checkpointing=False) does not TURN OFF
        # checkpointing on a model that already has it enabled, and this probe
        # reuses one model across cases.  Without an explicit disable, every
        # "OFF" case after the first "ON" one silently reports the checkpointed
        # figure -- which is how an earlier version of this script reported
        # seq4096-OFF as 20.4 GiB instead of 86.6 GiB.
        if want_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            fn()
            peak = torch.cuda.max_memory_allocated() / GB
            results[name] = round(peak, 1)
            print(f"[{args.label}] {name}: peak {peak:.1f} GiB", flush=True)
        except torch.OutOfMemoryError:
            results[name] = "OOM"
            print(f"[{args.label}] {name}: OOM", flush=True)
        except Exception as exc:                      # a wiring bug must not read as OOM
            results[name] = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"[{args.label}] {name}: ERROR {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()

    for seq in (1024, 2048, 4096):
        for gc in (False, True):
            for mb in ((1, 2) if gc else (1,)):       # unchecked mb=2 is hopeless at 4096
                probe(f"train_seq{seq}_mb{mb}_gradckpt{'ON' if gc else 'OFF'}",
                      lambda s=seq, g=gc, m=mb: train_lora(
                          model=model, tokenizer=tok, examples=make(s, 4 * m),
                          output_dir=OUT, max_length=s, max_steps=2, learning_rate=1e-5,
                          micro_batch_size=m, effective_batch_size=2 * m, seed=0,
                          gradient_checkpointing=g),
                      want_checkpointing=gc)

    for bs in (8, 16):
        probe(f"generate_bs{bs}_seq3000", lambda b=bs: generate_predictions(
            model=model, tokenizer=tok, examples=make(3000, b),
            batch_size=b, max_new_tokens=32), want_checkpointing=False)

    print(json.dumps(results, indent=2))
    print(f"\n--- [{args.label}] fits in {args.budget:.0f} GiB usable? ---")
    for k, v in results.items():
        if k == "params_B":
            continue
        if isinstance(v, (int, float)):
            print(f"  {k:44s} {v:6.1f} GiB  {'YES' if v < args.budget else 'NO'}")
        else:
            print(f"  {k:44s} {v}")


if __name__ == "__main__":
    main()

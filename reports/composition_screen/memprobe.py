"""Peak GPU memory for the configurations this project actually runs.

Answers whether the work fits on a 48 GB card, and what has to change.
Each case is measured in a fresh allocator state via reset_peak_memory_stats.
"""
import json, sys, torch
sys.path.insert(0, "/scratch/gpfs/BRENDEN/changho/compositional-something")
sys.path.insert(0, "/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen")

from self.coding.atomic_data import AtomicExample
from self.coding.training import load_qwen_lora_model, load_qwen_tokenizer, train_lora, generate_predictions
from pathlib import Path

MODEL = "/scratch/gpfs/BRENDEN/changho/hf_cache/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
OUT = Path("/scratch/gpfs/BRENDEN/changho/compositional-something/reports/composition_screen/logs/probe")
GB = 2 ** 30
results = {}

tok = load_qwen_tokenizer(MODEL)
model = load_qwen_lora_model(MODEL)
results["weights_only"] = torch.cuda.memory_allocated() / GB
print(f"weights+LoRA resident: {results['weights_only']:.1f} GiB", flush=True)

filler = "The quick brown fox jumps over the lazy dog. " * 400


def make(n_tokens, count):
    text = tok.decode(tok(filler * 12)["input_ids"][: n_tokens - 40])
    return [AtomicExample(task="probe", source_id=str(i), source_group_id=str(i), split="x",
                          messages=({"role": "user", "content": text},), target="answer",
                          evaluator={}, metadata={}) for i in range(count)]


def probe(name, fn):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        fn()
        peak = torch.cuda.max_memory_allocated() / GB
        results[name] = round(peak, 1)
        print(f"{name}: peak {peak:.1f} GiB", flush=True)
    except torch.OutOfMemoryError:
        results[name] = "OOM"
        print(f"{name}: OOM", flush=True)
    torch.cuda.empty_cache()


for seq in (1024, 2048, 4096):
    for gc in (False, True):
        tag = f"train_seq{seq}_mb1_gradckpt{'ON' if gc else 'OFF'}"
        if gc:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        else:
            model.gradient_checkpointing_disable()
        probe(tag, lambda s=seq: train_lora(
            model=model, tokenizer=tok, examples=make(s, 8), output_dir=OUT,
            max_length=s, max_steps=2, learning_rate=1e-5,
            micro_batch_size=1, effective_batch_size=2, seed=0))

model.gradient_checkpointing_disable()
for bs in (8, 16):
    probe(f"generate_bs{bs}_seq3000", lambda b=bs: generate_predictions(
        model=model, tokenizer=tok, examples=make(3000, b), batch_size=b, max_new_tokens=32))

print(json.dumps(results, indent=2))
print("\n--- fits on 48 GiB (leaving ~4 GiB headroom for fragmentation)? ---")
for k, v in results.items():
    if isinstance(v, float) or isinstance(v, int):
        print(f"  {k:42s} {v:6.1f} GiB  {'YES' if v < 44 else 'NO'}")

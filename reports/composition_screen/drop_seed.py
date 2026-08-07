#!/usr/bin/env python3
"""Train the DROP extraction seed, then rerun the screen (plan §4, §7.1).

Week 0 gate, second half.  The MuSiQue screen returned +.023 with a CI spanning
zero on the base model and reached +.080 (p=.0009) only after a one-hop seed
was trained, so the seed run is part of the gate rather than a response to a
null base screen.

Seed supervision
----------------
The seed cannot be built from QDMR steps.  BREAK releases no gold intermediate
values and holds 23 single-step DROP decompositions in train, so no step
carries a label.  Supervision comes from DROP's own span-answer questions on
the 1,941 train passages BREAK does not cover:

  * single-span answers give the length-one atom;
  * multi-span answers give the list atom that AGGREGATE nodes consume, which
    is what the count and sum family needs.

Targets are rendered in the `A | B | C` list format the extraction interface
uses at every model-owned node, so the seed teaches the output format as well
as extraction.  Passages outside BREAK's coverage are disjoint from the
composition pool by construction, which is the anti-memorization rule of §7.

Evaluation reuses `drop_isolation.screen` unchanged, so before and after are
measured by the same code path and per-instance outcomes are paired.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from drop_executor import render_list
from drop_isolation import (
    BatchedRunner,
    draw_cells,
    load_atom_proxy,
    print_pooled,
    prompt,
    screen,
    verdict,
)
from musique_seed import boot_ci, mcnemar

from self.coding.atomic_data import AtomicExample
from self.coding.training import load_qwen_lora_model, load_qwen_tokenizer, train_lora


def seed_example(row: dict) -> AtomicExample:
    return AtomicExample(
        task="drop_qdmr", source_id=row["example_id"],
        source_group_id=row["passage_id"], split="train",
        messages=({"role": "user",
                   "content": prompt(row["passage"], row["question"])},),
        target=render_list(row["answer"]), evaluator={},
        metadata={"answer_type": row["answer_type"]},
    )


def normalize_question(text: str) -> str:
    return " ".join(text.lower().split())


def leakage_gate(seed_rows: list[dict], dev_rows: list[dict],
                 comp_rows: list[dict]) -> dict:
    """Hard-exit on any passage, question, or id shared across pools.

    Passage-level disjointness is the binding constraint here: DROP asks many
    questions per passage, so a shared passage leaks extractable content even
    when no question repeats (plan §7).
    """
    seed_p = {r["passage_id"] for r in seed_rows}
    dev_p = {r["passage_id"] for r in dev_rows}
    comp_p = {r["passage_id"] for r in comp_rows}
    leak = {
        "seed_vs_dev_passages": len(seed_p & dev_p),
        "seed_vs_composition_passages": len(seed_p & comp_p),
        "seed_vs_dev_query_ids": len({r["query_id"] for r in seed_rows}
                                     & {r["query_id"] for r in dev_rows}),
        "seed_vs_dev_questions": len(
            {normalize_question(r["question"]) for r in seed_rows}
            & {normalize_question(r["original_question"]) for r in dev_rows}),
    }
    print("leakage check:", leak, flush=True)
    if any(leak.values()):
        raise SystemExit(f"train/dev/composition leakage detected: {leak}")
    return leak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed-pool", type=Path, required=True)
    ap.add_argument("--train-data", type=Path, required=True)
    ap.add_argument("--dev-data", type=Path, required=True)
    ap.add_argument("--dev-gold", type=Path, required=True)
    ap.add_argument("--atom-pool", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--cells", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--count-policy", default="reject",
                    help="see drop_isolation.py --count-policy (plan Risk 6)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="required to fit training on a 48 GB card")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = [json.loads(l) for l in open(args.seed_pool)]
    dev_rows = [json.loads(l) for l in open(args.dev_data)]
    comp_rows = [json.loads(l) for l in open(args.train_data)]
    gold = {r["example_id"]: r for r in map(json.loads, open(args.dev_gold))}
    leak = leakage_gate(seed_rows, dev_rows, comp_rows)

    rng = random.Random(args.seed)
    sample_by_k = draw_cells(dev_rows, args.cells, args.per_k, rng)
    passage_ids = {r["passage_id"] for s in sample_by_k.values() for r in s}
    atoms = load_atom_proxy(args.atom_pool, passage_ids,
                            args.per_k * len(sample_by_k), rng)

    tok = load_qwen_tokenizer(args.model)

    # `train_lora` REJECTS over-length examples rather than truncating, so the
    # filter runs on true tokenized length, not a character estimate.
    rng.shuffle(seed_rows)
    kept, dropped = [], 0
    for row in seed_rows:
        if len(kept) >= args.train_size:
            break
        item = seed_example(row)
        n_tok = len(tok(item.messages[0]["content"] + item.target)["input_ids"])
        if n_tok + 32 > args.max_length:
            dropped += 1
            continue
        kept.append(item)
    print(f"seed training examples: {len(kept)} (dropped {dropped} over "
          f"{args.max_length} tokens, pool {len(seed_rows)})", flush=True)
    if len(kept) < args.train_size:
        raise SystemExit(f"only {len(kept)} examples fit; need {args.train_size}")

    model = load_qwen_lora_model(args.model)
    torch.manual_seed(args.seed)
    runner = BatchedRunner(model, tok, args.batch_size)

    records: list[dict] = []
    report: dict = {"config": {k: str(v) for k, v in vars(args).items()},
                    "leakage": leak,
                    "seed_examples": len(kept), "seed_dropped": dropped}

    print("== BEFORE (base model) ==", flush=True)
    report["before"] = screen(runner, sample_by_k, gold, atoms, "before",
                              records, args.seed,
                              count_policy=args.count_policy)

    print("== training the extraction seed ==", flush=True)
    # The before-phase generation leaves a large allocator pool behind and
    # training then OOMs on the logits tensor even though the live model is
    # small; this is the same failure musique_seed.py hit.
    torch.cuda.empty_cache()
    print(f"  cuda reserved after empty_cache: "
          f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    tr = train_lora(model=model, tokenizer=tok, examples=kept,
                    output_dir=args.out_dir, max_length=args.max_length,
                    max_steps=args.max_steps, learning_rate=args.learning_rate,
                    micro_batch_size=args.micro_batch_size,
                    effective_batch_size=16, seed=args.seed,
                    gradient_checkpointing=args.gradient_checkpointing)
    report["training"] = {k: v for k, v in tr.items() if k != "log_history"}

    print("== AFTER ==", flush=True)
    report["after"] = screen(runner, sample_by_k, gold, atoms, "after",
                             records, args.seed,
                             count_policy=args.count_policy)

    # --- did seeding move the contrast the gate turns on? ---
    before, after = report["before"]["pooled"], report["after"]["pooled"]
    report["seed_effect"] = {
        "headroom_before": before["headroom"],
        "headroom_after": after["headroom"],
        "delta": after["headroom"] - before["headroom"],
        "atom_before": report["before"]["atom_proxy"]["accuracy"],
        "atom_after": report["after"]["atom_proxy"]["accuracy"],
    }

    # paired before/after on direct accuracy, the B0 baseline of plan §15
    by_id = {}
    for r in records:
        if r["arm_set"] == "screen":
            by_id.setdefault(r["id"], {})[r["phase"]] = r
    direct_pairs = [(v["after"]["direct_ok"], v["before"]["direct_ok"])
                    for v in by_id.values() if len(v) == 2]
    report["direct_before_after"] = {
        "n": len(direct_pairs),
        "delta": (sum(a for a, _ in direct_pairs)
                  - sum(b for _, b in direct_pairs)) / max(len(direct_pairs), 1),
        "mcnemar": mcnemar(direct_pairs),
        "ci95": list(boot_ci(direct_pairs, random.Random(args.seed))),
    }

    (args.out_dir / "per_instance.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    (args.out_dir / "drop_seed.json").write_text(
        json.dumps(report, indent=2, default=str))

    for phase in ("before", "after"):
        print_pooled(phase, report[phase]["pooled"])
    print(f"\nSEED EFFECT ON HEADROOM: {report['seed_effect']['delta']:+.3f}",
          flush=True)
    print(f"ATOM PROXY: {report['seed_effect']['atom_before']:.3f} -> "
          f"{report['seed_effect']['atom_after']:.3f}", flush=True)
    print(f"DIRECT (B0 baseline moves too): "
          f"{report['direct_before_after']['delta']:+.3f}", flush=True)
    print("VERDICT:", "composition wins after seeding -- proceed to Week 1"
          if verdict(after) else
          "no significant headroom after seeding -- gate not met (plan §4)",
          flush=True)


if __name__ == "__main__":
    main()

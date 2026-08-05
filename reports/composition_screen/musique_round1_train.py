#!/usr/bin/env python3
"""Round 1, stage 2: retrain on pseudo-labels, test whether DIRECT accuracy moves.

Continues LoRA training from the partition-seed adapter on accepted labels from
`musique_round1_label.py`, then evaluates on the standard 300-instance dev
sample.  `--label-key` selects the arm; every arm trains on the SAME accepted
question set so differences are attributable to the labels alone:

  composed  the experiment: labels from the seed's own composition chain
  direct    self-distillation control: the seed's direct answers as labels.
            If this matches the composed arm, self-training -- not composition
            -- explains the gain.
  gold      ceiling: gold answers on the same questions, bounding what label
            noise costs.

The "before" state is the r1seed adapter itself; its after-phase per-instance
records are reused as the paired baseline (`--before-per-instance`) because
evaluation on this cluster is deterministic -- verified when the r1seed job's
before-block matched the reproduction bit-for-bit.  Pass --eval-before to
re-measure instead.

Training poses each question over the full 20-paragraph context (the eval
condition), answer-only target (plan §12.1), lr 1e-4 as in clutrr_round1.py --
half the seed's, since this is a continuation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from musique_isolation import full_context
from musique_seed import boot_ci, evaluate, ex, mcnemar
from musique_selfdecomp import sample_like_seed_screen

from self.coding.training import (
    load_adapter_for_training,
    load_qwen_tokenizer,
    train_lora,
)


def paired_stats(after, before, rng):
    pairs = [(a, b) for a, b in zip(after, before)]
    return {"n": len(pairs),
            "after": sum(a for a, _ in pairs) / len(pairs),
            "before": sum(b for _, b in pairs) / len(pairs),
            "delta": (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / len(pairs),
            "mcnemar": mcnemar(pairs),
            "bootstrap_ci95": boot_ci(pairs, rng)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--label-key", choices=("composed", "direct", "gold"),
                    required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--dev-data", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--before-per-instance", type=Path, default=None)
    ap.add_argument("--eval-before", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if bool(args.before_per_instance) == bool(args.eval_before):
        raise SystemExit("pass exactly one of --before-per-instance / --eval-before")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = [json.loads(l) for l in open(args.labels) if json.loads(l)["accept"]]
    by_id = {json.loads(l)["id"]: json.loads(l) for l in open(args.train_data)}
    print(f"accepted labels: {len(labels)}", flush=True)

    dev_rows = [json.loads(l) for l in open(args.dev_data)]
    sample_by_k = sample_like_seed_screen(dev_rows, args.per_k, args.seed)

    # anti-leakage: the accepted set must not touch dev multihop questions
    dev_qs = {r["question"].strip().lower() for s in sample_by_k.values() for r in s}
    overlap = sum(1 for l in labels if l["question"].strip().lower() in dev_qs)
    if overlap:
        raise SystemExit(f"{overlap} accepted labels overlap dev questions")

    tok = load_qwen_tokenizer(args.model)
    items, dropped = [], 0
    rng = random.Random(args.seed)
    rng.shuffle(labels)
    for l in labels:
        r = by_id[l["id"]]
        target = r["answer"] if args.label_key == "gold" else l[args.label_key]
        item = ex(full_context(r), r["question"], str(target), l["id"], 2)
        n_tok = len(tok(item.messages[0]["content"] + str(item.target))["input_ids"])
        if n_tok + 32 > args.max_length:
            dropped += 1
            continue
        items.append(item)
    print(f"training examples: {len(items)} (dropped {dropped} over length)",
          flush=True)

    model, tok = load_adapter_for_training(args.model, args.adapter)
    records: list[dict] = []
    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "n_train": len(items)}

    if args.eval_before:
        print("== BEFORE (r1seed adapter) ==", flush=True)
        model.eval()
        report["before"] = evaluate(model, tok, sample_by_k, args.batch_size,
                                    "before", records)
        model.train()
    else:
        before = {}
        for rec in map(json.loads, open(args.before_per_instance)):
            if rec["phase"] == "after":
                before[rec["id"]] = rec
        want = {r["id"] for s in sample_by_k.values() for r in s}
        if want - before.keys():
            raise SystemExit("before-per-instance is missing sampled dev ids")
        for rid, rec in before.items():
            if rid in want:
                records.append({"phase": "before", "k": rec["k"], "id": rid,
                                "direct_ok": rec["direct_ok"],
                                "composed_ok": rec["composed_ok"]})
        report["before"] = {"source": str(args.before_per_instance)}

    print(f"== training on {args.label_key} labels ==", flush=True)
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    tr = train_lora(model=model, tokenizer=tok, examples=items,
                    output_dir=args.out_dir, max_length=args.max_length,
                    max_steps=args.max_steps, learning_rate=args.learning_rate,
                    micro_batch_size=args.micro_batch_size,
                    effective_batch_size=16, seed=args.seed,
                    gradient_checkpointing=args.gradient_checkpointing)
    report["training"] = {k: v for k, v in tr.items() if k != "log_history"}

    print("== AFTER ==", flush=True)
    model.eval()
    report["after"] = evaluate(model, tok, sample_by_k, args.batch_size,
                               "after", records)

    # paired before/after on identical instances; direct is the claim
    stat_rng = random.Random(args.seed)
    bmap = {(r["k"], r["id"]): r for r in records if r["phase"] == "before"}
    stats = {}
    for arm in ("direct_ok", "composed_ok"):
        pooled_a, pooled_b, by_hop = [], [], {}
        for k in sorted(sample_by_k):
            a = [r[arm] for r in records if r["phase"] == "after" and r["k"] == k]
            b = [bmap[(k, r["id"])][arm] for r in records
                 if r["phase"] == "after" and r["k"] == k]
            by_hop[str(k)] = paired_stats(a, b, stat_rng)
            pooled_a += a
            pooled_b += b
        stats[arm] = {"pooled": paired_stats(pooled_a, pooled_b, stat_rng),
                      "by_hop": by_hop}
    report["stats"] = stats

    with open(args.out_dir / "per_instance.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(args.out_dir / f"round1_{args.label_key}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    d = stats["direct_ok"]["pooled"]
    print(f"DIRECT before={d['before']:.3f} after={d['after']:.3f} "
          f"delta={d['delta']:+.3f} CI{d['bootstrap_ci95']} "
          f"p={d['mcnemar']['p_value']:.4f}", flush=True)
    for k, v in stats["direct_ok"]["by_hop"].items():
        print(f"  k={k}: {v['before']:.3f} -> {v['after']:.3f} "
              f"({v['delta']:+.3f})", flush=True)


if __name__ == "__main__":
    main()

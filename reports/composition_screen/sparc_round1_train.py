#!/usr/bin/env python3
"""Round 1, stage 2: retrain on pseudo-labels, test whether DIRECT moves.

Continues LoRA training from the seed adapter on accepted labels from
sparc_round1_label.py, then evaluates on the standard audit-db sample.
`--label-key` selects the arm; every arm trains on the SAME accepted question
set so differences are attributable to the labels alone (plan §15):

  composed  B5, the experiment: the seed's own guarded chain SQL
  direct    self-distillation control: the seed's direct SQL as labels
  gold      B7 ceiling: gold final SQL on the same questions (read from the
            audit gold file -- the label file does not store gold, plan §8)

Primary training items are single-shot (plan §13.1: question -> final SQL).
`--rehearsal-frac F` mixes in turn-mode items rebuilt from the SAME label
file's accepted rows -- (schema, previous accepted state, turn question) ->
turn SQL for non-sink turns -- targeting the part-accuracy erosion MuSiQue
round 1 showed.  `--replay-frac` mixes seed examples (--replay-data) or a
previous round's accepted composed labels (--replay-labels), the plan §12
mixture; both length-filter with the same fits()/take() idiom because
train_lora rejects over-length examples.

Anti-leakage (plan §7, stronger than MuSiQue's question-level check because
databases share schema and values): accepted labels must come from
composition-source dbs, disjoint from seed, audit, and dev dbs -- asserted at
run time.

The "before" state is the seed adapter; its after-phase per-instance records
are reused as the paired baseline (--before-per-instance) since evaluation on
this cluster is deterministic (verified for MuSiQue, musique_round1_train.py
docstring).  Pass --eval-before to re-measure instead.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from musique_seed import boot_ci, mcnemar
from sparc_seed import evaluate, load_inputs, sample_audit_cells

from self.coding.atomic_data import read_examples
from self.coding.sparc_composition import single_prompt, sparc_ex, turn_prompt
from self.coding.training import (
    load_adapter_for_training,
    load_qwen_tokenizer,
    train_lora,
)

REHEARSAL_KEYS = ("produced", "parses", "schema_valid", "literals_typed",
                  "monotone", "exec_ok", "shape_ok", "nonempty_ok",
                  "twoview_agree")


def paired_stats(after, before, rng):
    pairs = list(zip(after, before))
    n = max(len(pairs), 1)
    return {"n": len(pairs),
            "after": sum(a for a, _ in pairs) / n,
            "before": sum(b for _, b in pairs) / n,
            "delta": (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / n,
            "mcnemar": mcnemar(pairs),
            "bootstrap_ci95": boot_ci(pairs, rng)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--label-key", choices=("composed", "direct", "gold"),
                    required=True)
    for a in ("--sequences", "--gold", "--schemas", "--partition"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--testsuite-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--before-per-instance", type=Path, default=None)
    ap.add_argument("--eval-before", action="store_true")
    ap.add_argument("--rehearsal-frac", type=float, default=0.0)
    ap.add_argument("--replay-data", type=Path, default=None,
                    help="seed_examples.jsonl (AtomicExamples) to replay")
    ap.add_argument("--replay-labels", type=Path, default=None,
                    help="a previous round's label file; its accepted "
                         "composed labels replay as single-shot items")
    ap.add_argument("--replay-frac", type=float, default=0.0)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke flag: cap labels and per-cell at N")
    args = ap.parse_args()
    if bool(args.before_per_instance) == bool(args.eval_before):
        raise SystemExit("pass exactly one of --before-per-instance / --eval-before")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        args.per_cell = min(args.per_cell, args.limit)
        args.max_steps = min(args.max_steps, 4)

    data = load_inputs(args)
    part = data["partition"]
    comp_dbs = set(part["composition_dbs"])
    banned = (set(part["seed_dbs"]) | set(part["audit_dbs"])
              | set(part["dev_dbs"]))

    labels = [json.loads(l) for l in open(args.labels)]
    labels = [l for l in labels if l["accept"]]
    if args.limit:
        labels = labels[:args.limit]
    print(f"accepted labels: {len(labels)}", flush=True)
    if not labels:
        raise SystemExit("no accepted labels")

    # anti-leakage: db-level (plan §7 -- subsumes question-level)
    bad = [l["id"] for l in labels
           if l["db_id"] not in comp_dbs or l["db_id"] in banned]
    if bad:
        raise SystemExit(f"{len(bad)} accepted labels outside composition dbs: "
                         f"{bad[:5]}")

    sample_by_cell = sample_audit_cells(data, args.per_cell, args.seed)
    print("audit sample:", {c: len(v) for c, v in sample_by_cell.items()},
          flush=True)

    tok = load_qwen_tokenizer(args.model)
    schemas = data["schemas"]

    def fits(item):
        n_tok = len(tok(item.messages[0]["content"] + str(item.target))["input_ids"])
        return n_tok + 160 <= args.max_length

    def take(pool, want):
        kept, dropped_ = [], 0
        for item in pool:
            if len(kept) >= want:
                break
            if fits(item):
                kept.append(item)
            else:
                dropped_ += 1
        return kept, dropped_

    def step_pairs(rows):
        """Non-sink turn-mode pairs from accepted rows (plan §13.1)."""
        for l in rows:
            state = None
            for i, (q, a, c) in enumerate(zip(l["step_questions"][:-1],
                                              l["step_answers"][:-1],
                                              l["step_checks"][:-1]), 1):
                if q and a and all(c.get(k) for k in REHEARSAL_KEYS):
                    yield sparc_ex(
                        turn_prompt(schemas[l["db_id"]]["text"], state, q),
                        a, f"{l['id']}|reh{i}", l["db_id"], 1)
                state = a

    # --- primary items: single-shot question -> arm label ---
    gold_final = {g["sequence_id"]: g["final_sql"] for g in
                  map(json.loads, open(args.gold))}
    rng = random.Random(args.seed)
    rng.shuffle(labels)
    items, dropped = [], 0
    for l in labels:
        target = (gold_final[l["id"]] if args.label_key == "gold"
                  else l[args.label_key])
        if not target:
            dropped += 1
            continue
        item = sparc_ex(single_prompt(schemas[l["db_id"]]["text"],
                                      l["question"]),
                        str(target), l["id"], l["db_id"], l.get("k", 2))
        if fits(item):
            items.append(item)
        else:
            dropped += 1
    print(f"training examples: {len(items)} (dropped {dropped})", flush=True)

    new_frac = 1.0 - args.rehearsal_frac - args.replay_frac
    if new_frac <= 0:
        raise SystemExit("rehearsal + replay must leave room for new labels")
    if args.replay_frac > 0 and not (args.replay_data or args.replay_labels):
        raise SystemExit("--replay-frac needs --replay-data or --replay-labels")
    n_new = len(items)

    if args.replay_frac > 0:
        rep_pool = []
        if args.replay_data:
            rep_pool += read_examples(args.replay_data)
        if args.replay_labels:
            for l in map(json.loads, open(args.replay_labels)):
                if l["accept"] and l["composed"]:
                    rep_pool.append(sparc_ex(
                        single_prompt(schemas[l["db_id"]]["text"],
                                      l["question"]),
                        str(l["composed"]), l["id"], l["db_id"],
                        l.get("k", 2)))
        rng.shuffle(rep_pool)
        rep, rep_dropped = take(rep_pool, round(args.replay_frac / new_frac * n_new))
        items += rep
        print(f"replay mixed in: {len(rep)} (dropped {rep_dropped})", flush=True)

    if args.rehearsal_frac > 0:
        pool = list(step_pairs(labels))
        rng.shuffle(pool)
        reh, r_dropped = take(pool, round(args.rehearsal_frac / new_frac * n_new))
        items += reh
        print(f"rehearsal turn-pairs mixed in: {len(reh)} (dropped {r_dropped})",
              flush=True)

    rng.shuffle(items)
    print(f"total mix: {len(items)}", flush=True)

    model, tok = load_adapter_for_training(args.model, args.adapter)
    records: list = []
    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "n_train": len(items)}
    suite_cache: dict = {}

    if args.eval_before:
        print("== BEFORE (seed adapter) ==", flush=True)
        model.eval()
        report["before"] = evaluate(model, tok, data, sample_by_cell, args,
                                    "before", records, suite_cache)
        model.train()
    else:
        rows = [json.loads(l) for l in open(args.before_per_instance)]
        before = [r for r in rows if r["phase"] == "after"]
        want = {r["sequence_id"] for s in sample_by_cell.values() for r in s}
        have = {r["id"] for r in before if r["slice"] == "trio"}
        if want - have:
            raise SystemExit("before-per-instance is missing sampled audit ids")
        for rec in before:
            records.append({**rec, "phase": "before"})
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
    report["after"] = evaluate(model, tok, data, sample_by_cell, args, "after",
                               records, suite_cache)

    # paired before/after on identical instances; DIRECT is the claim (plan §20)
    stat_rng = random.Random(args.seed)
    stats = {}
    b_trio = {r["id"]: r for r in records
              if r["phase"] == "before" and r["slice"] == "trio"}
    a_trio = [r for r in records
              if r["phase"] == "after" and r["slice"] == "trio"]
    for arm in ("direct_ok", "composed_ok"):
        pooled_a, pooled_b, by_cell = [], [], {}
        for cell in sorted(sample_by_cell):
            a = [r[arm] for r in a_trio if r["cell"] == cell]
            b = [b_trio[r["id"]][arm] for r in a_trio if r["cell"] == cell]
            by_cell[cell] = paired_stats(a, b, stat_rng)
            pooled_a += a
            pooled_b += b
        stats[arm] = {"pooled": paired_stats(pooled_a, pooled_b, stat_rng),
                      "by_cell": by_cell}
    b_t1 = {r["id"]: r for r in records
            if r["phase"] == "before" and r["slice"] == "turn1"}
    a_t1 = [r for r in records
            if r["phase"] == "after" and r["slice"] == "turn1"]
    stats["turn1"] = paired_stats([r["turn1_ok"] for r in a_t1],
                                  [b_t1[r["id"]]["turn1_ok"] for r in a_t1],
                                  stat_rng)
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
    for cell, v in stats["direct_ok"]["by_cell"].items():
        print(f"  k={cell}: {v['before']:.3f} -> {v['after']:.3f} "
              f"({v['delta']:+.3f})", flush=True)
    t1 = stats["turn1"]
    print(f"TURN-1 regression: {t1['before']:.3f} -> {t1['after']:.3f} "
          f"({t1['delta']:+.3f}; plan §20 bar: >= -0.02)", flush=True)


if __name__ == "__main__":
    main()

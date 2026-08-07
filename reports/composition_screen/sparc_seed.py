#!/usr/bin/env python3
"""Train the SParC seed: turn-1 edits + Spider-easy singles, gold supervision.

Plan §12 seed row.  Training data comes ONLY from seed-source databases
(database-level partition, plan §7); evaluation runs on internal-audit
databases so Spider/SParC dev stays frozen until pipeline freeze (plan §18).

Two example formats, interleaved (plan §9):
  turn mode    (schema, empty state, turn-1 question) -> turn-1 gold SQL
  single mode  (schema, Spider-easy question)         -> gold SQL

Before/after evaluation, all paired per instance:
  turn1     turn-mode accuracy on audit-db turn-1 items (§20 bar: >= .70)
  trio      direct / part / composed at each turn-count cell on monotone
            audit sequences -- the same contrast the screen measures, so the
            seeded re-screen story is continuous.

The after-phase per-instance records are the paired "before" baseline for
sparc_round1_train.py (--before-per-instance), the same deterministic-eval
shortcut musique_round1_train.py documents.  seed_examples.jsonl preserves
the exact training items for round-1 replay (plan §12 mixture).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from musique_seed import boot_ci, mcnemar

from self.coding.atomic_data import write_examples
from self.coding.sparc_composition import (
    extract_sql,
    single_prompt,
    sparc_ex,
    suite_correct,
    suite_paths,
    turn_prompt,
)
from self.coding.training import (
    generate_predictions,
    load_qwen_lora_model,
    load_qwen_tokenizer,
    train_lora,
)

TARGET_MARGIN = 160     # tokens reserved for the SQL target + chat wrapping


def k_bucket(k: int) -> str:
    return "4+" if k >= 4 else str(k)


def load_inputs(args):
    """Shared loader for every SParC GPU script."""
    data = {
        "sequences": [json.loads(l) for l in open(args.sequences)],
        "gold": {g["sequence_id"]: g
                 for g in map(json.loads, open(args.gold))},
        "schemas": json.load(open(args.schemas)),
        "partition": json.load(open(args.partition)),
    }
    if getattr(args, "spider_singles", None):
        data["singles"] = [json.loads(l) for l in open(args.spider_singles)]
        data["singles_gold"] = {g["id"]: g["query"] for g in
                                map(json.loads, open(args.spider_gold))}
    return data


def sample_audit_cells(data, per_cell: int, seed: int):
    """The fixed audit-db eval sample every script pairs on (rng seed 7)."""
    audit = set(data["partition"]["audit_dbs"])
    pool = [r for r in data["sequences"]
            if r["split"] == "train" and r["db_id"] in audit
            and r["usable"] and r["monotone"]
            and not data["schemas"][r["db_id"]]["excluded"]]
    by_cell = defaultdict(list)
    for r in pool:
        by_cell[k_bucket(r["turn_count"])].append(r)
    rng = random.Random(seed)
    return {c: (v if len(v) <= per_cell else rng.sample(v, per_cell))
            for c, v in sorted(by_cell.items())}


def evaluate(model, tok, data, sample_by_cell, args, tag, records,
             suite_cache=None):
    """direct / part(gold state) / composed(self-fed) per cell, plus turn-1.

    Appends per-instance rows to `records`; the trio rows carry direct_ok /
    composed_ok for pairing, turn-1 rows carry turn1_ok.
    """
    schemas, gold = data["schemas"], data["gold"]
    suite_cache = {} if suite_cache is None else suite_cache

    def paths_for(db_id):
        if db_id not in suite_cache:
            suite_cache[db_id] = suite_paths(args.testsuite_root, args.db_root,
                                             db_id)
        return suite_cache[db_id]

    def gen(items):
        if not items:
            return []
        return generate_predictions(model=model, tokenizer=tok, examples=items,
                                    batch_size=args.batch_size,
                                    max_new_tokens=args.max_new_tokens)

    out = {}
    # --- turn-1 slice (the seed's own format, §20 regression bar) ---
    t1_pool = [r for s in sample_by_cell.values() for r in s]
    t1_pred = gen([sparc_ex(
        turn_prompt(schemas[r["db_id"]]["text"], None,
                    r["turns"][0]["question"]),
        "", f"{r['sequence_id']}|t1", r["db_id"], 1) for r in t1_pool])
    t1_ok = [suite_correct(paths_for(r["db_id"]), extract_sql(p),
                           gold[r["sequence_id"]]["turn_sqls"][0],
                           timeout_s=args.timeout_s)
             for p, r in zip(t1_pred, t1_pool)]
    for r, ok in zip(t1_pool, t1_ok):
        records.append({"phase": tag, "slice": "turn1", "k": r["turn_count"],
                        "id": r["sequence_id"], "turn1_ok": bool(ok)})
    out["turn1"] = {"n": len(t1_ok), "acc": sum(t1_ok) / max(len(t1_ok), 1)}
    print(f"  [{tag}] turn1: {out['turn1']['acc']:.3f} (n={out['turn1']['n']})",
          flush=True)

    for cell, sample in sample_by_cell.items():
        n = len(sample)
        finals = [gold[r["sequence_id"]]["final_sql"] for r in sample]

        d_pred = gen([sparc_ex(single_prompt(schemas[r["db_id"]]["text"],
                                             r["final_intent_question"]),
                               "", r["sequence_id"], r["db_id"],
                               r["turn_count"]) for r in sample])
        d_ok = [suite_correct(paths_for(r["db_id"]), extract_sql(p), g,
                              timeout_s=args.timeout_s)
                for p, r, g in zip(d_pred, sample, finals)]

        p_items, p_gold = [], []
        for r in sample:
            g = gold[r["sequence_id"]]
            for i, t in enumerate(r["turns"], 1):
                prev = g["turn_sqls"][i - 2] if i > 1 else None
                p_items.append(sparc_ex(
                    turn_prompt(schemas[r["db_id"]]["text"], prev,
                                t["question"]),
                    "", f"{r['sequence_id']}|p{i}", r["db_id"],
                    r["turn_count"]))
                p_gold.append((r["db_id"], g["turn_sqls"][i - 1]))
        p_ok = [suite_correct(paths_for(db), extract_sql(p), g,
                              timeout_s=args.timeout_s)
                for p, (db, g) in zip(gen(p_items), p_gold)]

        state = [None] * n
        for i in range(1, max(r["turn_count"] for r in sample) + 1):
            idx, items = [], []
            for j, r in enumerate(sample):
                if i > r["turn_count"]:
                    continue
                idx.append(j)
                items.append(sparc_ex(
                    turn_prompt(schemas[r["db_id"]]["text"], state[j],
                                r["turns"][i - 1]["question"]),
                    "", f"{r['sequence_id']}|c{i}", r["db_id"],
                    r["turn_count"]))
            for j, p in zip(idx, gen(items)):
                sql = extract_sql(p)
                if sql:
                    state[j] = sql
        c_ok = [s is not None and suite_correct(paths_for(r["db_id"]), s, g,
                                                timeout_s=args.timeout_s)
                for s, r, g in zip(state, sample, finals)]

        for r, do, co in zip(sample, d_ok, c_ok):
            records.append({"phase": tag, "slice": "trio", "k": r["turn_count"],
                            "cell": cell, "id": r["sequence_id"],
                            "direct_ok": bool(do), "composed_ok": bool(co)})
        out[cell] = {"n": n, "direct": sum(d_ok) / n,
                     "part": sum(p_ok) / len(p_ok),
                     "composed": sum(c_ok) / n,
                     "headroom": (sum(c_ok) - sum(d_ok)) / n}
        v = out[cell]
        print(f"  [{tag}] k={cell}: direct={v['direct']:.3f} "
              f"part={v['part']:.3f} composed={v['composed']:.3f} "
              f"({v['headroom']:+.3f})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    for a in ("--sequences", "--gold", "--spider-singles", "--spider-gold",
              "--schemas", "--partition"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--testsuite-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke flag: cap train size and per-cell at N")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        args.train_size = min(args.train_size, args.limit)
        args.per_cell = min(args.per_cell, args.limit)
        args.max_steps = min(args.max_steps, 4)

    data = load_inputs(args)
    part = data["partition"]
    seed_dbs = set(part["seed_dbs"])
    audit_dbs = set(part["audit_dbs"])
    dev_dbs = set(part["dev_dbs"])
    # plan §7: database-level disjointness is the anti-memorization rule
    assert not seed_dbs & (audit_dbs | dev_dbs), "partition leakage"
    assert not set(part["composition_dbs"]) & (seed_dbs | audit_dbs | dev_dbs)

    # --- seed training pool (gold targets, seed-source dbs only) ---
    turn1 = []
    for r in data["sequences"]:
        if (r["split"] == "train" and r["db_id"] in seed_dbs and r["usable"]
                and not data["schemas"][r["db_id"]]["excluded"]):
            g = data["gold"][r["sequence_id"]]
            item = sparc_ex(
                turn_prompt(data["schemas"][r["db_id"]]["text"], None,
                            r["turns"][0]["question"]),
                g["turn_sqls"][0], f"seed-t1|{r['sequence_id']}",
                r["db_id"], 1)
            turn1.append(item)
    easy = []
    for s in data["singles"]:
        if (s["split"] == "train" and s["db_id"] in seed_dbs
                and s["hardness"] == "easy"
                and not data["schemas"][s["db_id"]]["excluded"]):
            easy.append(sparc_ex(
                single_prompt(data["schemas"][s["db_id"]]["text"],
                              s["question"]),
                data["singles_gold"][s["id"]], f"seed-sp|{s['id']}",
                s["db_id"], 1))
    rng = random.Random(args.seed)
    rng.shuffle(turn1)
    rng.shuffle(easy)
    pool = [x for pair in zip(turn1, easy) for x in pair]     # 50/50 interleave
    longer = turn1 if len(turn1) > len(easy) else easy
    pool += longer[min(len(turn1), len(easy)):]

    tok = load_qwen_tokenizer(args.model)

    def fits(item):
        n_tok = len(tok(item.messages[0]["content"] + str(item.target))["input_ids"])
        return n_tok + TARGET_MARGIN <= args.max_length

    kept, dropped = [], 0
    for item in pool:
        if len(kept) >= args.train_size:
            break
        if fits(item):
            kept.append(item)
        else:
            dropped += 1
    print(f"seed pool: turn1={len(turn1)} easy={len(easy)} -> "
          f"training on {len(kept)} (dropped {dropped} over length)", flush=True)
    if len(kept) < min(args.train_size, 500):
        raise SystemExit(f"only {len(kept)} seed examples; too few to train")
    assert all(i.source_group_id in seed_dbs for i in kept)
    write_examples(args.out_dir / "seed_examples.jsonl", kept)

    sample_by_cell = sample_audit_cells(data, args.per_cell, args.seed)
    print("audit sample:", {c: len(v) for c, v in sample_by_cell.items()},
          flush=True)

    model = load_qwen_lora_model(args.model)
    torch.manual_seed(args.seed)
    records: list = []
    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "n_train": len(kept)}

    print("== BEFORE (base) ==", flush=True)
    suite_cache: dict = {}
    report["before"] = evaluate(model, tok, data, sample_by_cell, args,
                                "before", records, suite_cache)

    print("== training seed ==", flush=True)
    torch.cuda.empty_cache()
    tr = train_lora(model=model, tokenizer=tok, examples=kept,
                    output_dir=args.out_dir, max_length=args.max_length,
                    max_steps=args.max_steps, learning_rate=args.learning_rate,
                    micro_batch_size=args.micro_batch_size,
                    effective_batch_size=16, seed=args.seed,
                    gradient_checkpointing=args.gradient_checkpointing)
    report["training"] = {k: v for k, v in tr.items() if k != "log_history"}

    print("== AFTER ==", flush=True)
    model.eval()
    report["after"] = evaluate(model, tok, data, sample_by_cell, args,
                               "after", records, suite_cache)

    # paired stats: turn-1 lift and the trio contrast, before vs after
    stats = {}
    for phase in ("before", "after"):
        trio = [(r["composed_ok"], r["direct_ok"]) for r in records
                if r["phase"] == phase and r["slice"] == "trio"]
        lo, hi = boot_ci(trio, random.Random(args.seed))
        stats[phase] = {
            "n": len(trio),
            "composed": sum(c for c, _ in trio) / max(len(trio), 1),
            "direct": sum(d for _, d in trio) / max(len(trio), 1),
            "headroom": (sum(c for c, _ in trio) - sum(d for _, d in trio))
                        / max(len(trio), 1),
            "ci95": [lo, hi], "mcnemar": mcnemar(trio),
            "turn1": report[phase]["turn1"]["acc"],
        }
    report["stats"] = stats

    (args.out_dir / "per_instance.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    (args.out_dir / "sparc_seed.json").write_text(
        json.dumps(report, indent=2, default=str))

    for phase in ("before", "after"):
        s = stats[phase]
        print(f"\n{phase.upper()}: turn1={s['turn1']:.3f} "
              f"composed={s['composed']:.3f} direct={s['direct']:.3f} "
              f"headroom={s['headroom']:+.3f} "
              f"CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] "
              f"p={s['mcnemar']['p_value']:.4f}", flush=True)
    print(f"\nSEED BAR (plan §20, >= .70 turn-1): "
          f"{'MET' if stats['after']['turn1'] >= 0.70 else 'NOT MET'}",
          flush=True)


if __name__ == "__main__":
    main()

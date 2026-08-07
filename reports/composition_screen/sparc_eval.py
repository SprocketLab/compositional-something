#!/usr/bin/env python3
"""Frozen-model baseline harness: one script for B0 / B2 / B6 / B8 (plan §15).

  --mode single           direct single-shot generation           (B0)
  --mode seq              turns visible, one call, sink SQL only  (B8)
  --mode chain            self-fed turn chain, no retraining      (B2)
  --mode chain-goldprefix every turn conditioned on GOLD previous
                          SQL -- the exact cascade diagnostic     (B6, §16.2)

  --slice audit           internal-audit dbs (threshold work)
  --slice dev             SParC dev (only after pipeline freeze, plan §18)

Per-instance records use the isolation script's schema so the §16 analyses
pool across files.  Scoring is test-suite execution accuracy of the final
query; chain modes also record per-turn correctness for the error-propagation
analysis (§16.2).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sparc_isolation import load_model
from sparc_seed import k_bucket, load_inputs, sample_audit_cells

from self.coding.sparc_composition import (
    extract_sql,
    seq_prompt,
    single_prompt,
    sparc_ex,
    suite_correct,
    suite_paths,
    turn_prompt,
)
from self.coding.training import generate_predictions

MODES = ("single", "seq", "chain", "chain-goldprefix")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--slice", choices=("audit", "dev"), default="audit")
    for a in ("--sequences", "--gold", "--schemas", "--partition"):
        ap.add_argument(a, type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--testsuite-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.limit:
        args.per_cell = min(args.per_cell, args.limit)

    data = load_inputs(args)
    schemas, gold = data["schemas"], data["gold"]
    if args.slice == "audit":
        sample_by_cell = sample_audit_cells(data, args.per_cell, args.seed)
    else:
        pool = [r for r in data["sequences"]
                if r["split"] == "dev" and r["usable"] and r["monotone"]
                and not schemas[r["db_id"]]["excluded"]]
        by_cell = defaultdict(list)
        for r in pool:
            by_cell[k_bucket(r["turn_count"])].append(r)
        rng = random.Random(args.seed)
        sample_by_cell = {c: (v if len(v) <= args.per_cell
                              else rng.sample(v, args.per_cell))
                          for c, v in sorted(by_cell.items())}
    print("sample:", {c: len(v) for c, v in sample_by_cell.items()}, flush=True)

    model, tok = load_model(args.model, args.adapter)
    suite_cache = {}

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

    records = []
    report = {"config": {k: str(v) for k, v in vars(args).items()},
              "by_cell": {}}
    for cell, sample in sample_by_cell.items():
        n = len(sample)
        finals = [gold[r["sequence_id"]]["final_sql"] for r in sample]
        turn_ok = [None] * n

        if args.mode == "single":
            preds = gen([sparc_ex(single_prompt(schemas[r["db_id"]]["text"],
                                                r["final_intent_question"]),
                                  "", r["sequence_id"], r["db_id"],
                                  r["turn_count"]) for r in sample])
            final_sql = [extract_sql(p) for p in preds]
        elif args.mode == "seq":
            preds = gen([sparc_ex(
                seq_prompt(schemas[r["db_id"]]["text"],
                           [t["question"] for t in r["turns"]]),
                "", r["sequence_id"], r["db_id"], r["turn_count"])
                for r in sample])
            final_sql = [extract_sql(p) for p in preds]
        else:
            goldprefix = args.mode == "chain-goldprefix"
            state = [None] * n
            turn_ok = [[] for _ in sample]
            for i in range(1, max(r["turn_count"] for r in sample) + 1):
                idx, items = [], []
                for j, r in enumerate(sample):
                    if i > r["turn_count"]:
                        continue
                    prev = (gold[r["sequence_id"]]["turn_sqls"][i - 2]
                            if goldprefix and i > 1 else state[j])
                    idx.append(j)
                    items.append(sparc_ex(
                        turn_prompt(schemas[r["db_id"]]["text"], prev,
                                    r["turns"][i - 1]["question"]),
                        "", f"{r['sequence_id']}|{i}", r["db_id"],
                        r["turn_count"]))
                for j, p in zip(idx, gen(items)):
                    sql = extract_sql(p)
                    if sql:
                        state[j] = sql
                    r = sample[j]
                    turn_ok[j].append(bool(sql) and suite_correct(
                        paths_for(r["db_id"]), sql,
                        gold[r["sequence_id"]]["turn_sqls"][i - 1],
                        timeout_s=args.timeout_s))
                print(f"  [{cell}] turn {i} done", flush=True)
            final_sql = state

        ok = [s is not None and s != "" and suite_correct(
                  paths_for(r["db_id"]), s, g, timeout_s=args.timeout_s)
              for s, r, g in zip(final_sql, sample, finals)]
        for j, (r, o) in enumerate(zip(sample, ok)):
            records.append({
                "arm": args.mode, "slice": args.slice, "cell": cell,
                "k": r["turn_count"], "sequence_id": r["sequence_id"],
                "db_id": r["db_id"], "ok": bool(o),
                "turn_ok": turn_ok[j] if isinstance(turn_ok[j], list) else None,
                "pred_sql": final_sql[j], "hardness": r["hardness"]})
        v = {"n": n, "acc": sum(ok) / n}
        if args.mode in ("chain", "chain-goldprefix"):
            flat = [x for t in turn_ok for x in t]
            v["per_turn_acc"] = sum(flat) / max(len(flat), 1)
        report["by_cell"][cell] = v
        print(f"[{args.mode}/{args.slice}] k={cell}: acc={v['acc']:.3f}"
              + (f" per-turn={v['per_turn_acc']:.3f}"
                 if "per_turn_acc" in v else ""), flush=True)

    total = sum(v["n"] for v in report["by_cell"].values())
    report["pooled"] = {
        "n": total,
        "acc": sum(v["acc"] * v["n"] for v in report["by_cell"].values()) / total}
    by_hard = defaultdict(list)
    for r in records:
        by_hard[r["hardness"]].append(r["ok"])
    report["by_hardness"] = {h: {"n": len(v), "acc": sum(v) / len(v)}
                             for h, v in sorted(by_hard.items())}

    tag = f"{args.mode.replace('-', '_')}_{args.slice}"
    with open(args.out_dir / f"per_instance_{tag}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(args.out_dir / f"sparc_eval_{tag}.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"POOLED {args.mode}/{args.slice}: {report['pooled']['acc']:.3f} "
          f"(n={total})", flush=True)


if __name__ == "__main__":
    main()

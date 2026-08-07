#!/usr/bin/env python3
"""SParC isolation screen: the pre-registered gate (plan §4).

Four arms on monotone dev sequences, paired per instance, all scored by
test-suite execution accuracy of the final query against hidden gold:

  direct    final intent question -> whole SQL, one call
  part      each turn alone, GOLD previous SQL as state (per-turn scored)
  composed  self-fed chain: each turn conditioned on the model's own SQL
  corrupt   composed, but the state at one turn is replaced by gold SQL from
            a DIFFERENT sequence on the same database

Verdict rule (pre-registered): proceed only if pooled composed-direct headroom
has 95% CI lower bound > 0 and McNemar p < .05; feasibility bar: part >= .60;
corruption check: corrupt must fall substantially below composed (>= .05 here),
otherwise the final turn alone determines the query and composition is not
load-bearing.  The verdict block in the report is machine-checkable.

Gold turn SQL is a screen-side measurement input (plan §4 oracle accounting:
the turn decomposition is released structure; gold SQL is audit-only for the
PIPELINE, but the part/corrupt arms of the screen consume it by design).

Same script serves the base screen, the seeded re-screen (--adapter), and the
1.7B record run (different --model).
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

from self.coding.sparc_composition import (
    extract_sql,
    single_prompt,
    sparc_ex,
    suite_correct,
    suite_paths,
    turn_prompt,
)
from self.coding.training import generate_predictions, load_adapter_for_evaluation


def load_model(model_name: str, adapter):
    if adapter:
        return load_adapter_for_evaluation(model_name, adapter)
    from transformers import AutoModelForCausalLM

    from self.coding.training import load_qwen_tokenizer
    tok = load_qwen_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16).to("cuda").eval()
    return model, tok


def k_bucket(k: int) -> str:
    return "4+" if k >= 4 else str(k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--sequences", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--schemas", type=Path, required=True)
    ap.add_argument("--db-root", type=Path, required=True)
    ap.add_argument("--testsuite-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke flag: cap every cell at N instances")
    args = ap.parse_args()

    schemas = json.load(open(args.schemas))
    gold = {g["sequence_id"]: g for g in map(json.loads, open(args.gold))}
    rows = [json.loads(l) for l in open(args.sequences)]
    dev = [r for r in rows
           if r["split"] == "dev" and r["usable"] and r["monotone"]
           and not schemas[r["db_id"]]["excluded"]]
    by_cell = defaultdict(list)
    for r in dev:
        by_cell[k_bucket(r["turn_count"])].append(r)
    rng = random.Random(args.seed)
    per_cell = min(args.per_cell, args.limit) if args.limit else args.per_cell
    sample_by_cell = {}
    for cell in sorted(by_cell):
        pool = by_cell[cell]
        sample_by_cell[cell] = (pool if len(pool) <= per_cell
                                else rng.sample(pool, per_cell))

    # donor states for the corrupt arm come from the full monotone dev pool
    donors_by_db = defaultdict(list)
    for r in dev:
        donors_by_db[r["db_id"]].append(r)

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
        preds = generate_predictions(model=model, tokenizer=tok, examples=items,
                                     batch_size=args.batch_size,
                                     max_new_tokens=args.max_new_tokens)
        print(f"      generated {len(items)}", flush=True)
        return preds

    def run_chain(sample, corrupt_turns=None):
        """Self-fed chain; corrupt_turns[j] (if set) swaps the state before
        that turn for the donor's gold prefix SQL."""
        state = [None] * len(sample)
        produced = [True] * len(sample)
        max_k = max(r["turn_count"] for r in sample)
        for i in range(1, max_k + 1):
            idx, items = [], []
            for j, r in enumerate(sample):
                if i > r["turn_count"]:
                    continue
                if corrupt_turns and corrupt_turns[j] and i == corrupt_turns[j][0]:
                    state[j] = corrupt_turns[j][1]
                schema_text = schemas[r["db_id"]]["text"]
                q = r["turns"][i - 1]["question"]
                idx.append(j)
                items.append(sparc_ex(turn_prompt(schema_text, state[j], q),
                                      "", f"{r['sequence_id']}|{i}",
                                      r["db_id"], r["turn_count"]))
            for j, p in zip(idx, gen(items)):
                sql = extract_sql(p)
                if sql:
                    state[j] = sql
                else:
                    produced[j] = False       # carry the old state forward
        return state, produced

    records, report = [], {"config": {k: str(v) for k, v in vars(args).items()},
                           "by_cell": {}}
    for cell, sample in sample_by_cell.items():
        n = len(sample)
        print(f"== cell k={cell}: {n} instances ==", flush=True)
        finals = [gold[r["sequence_id"]]["final_sql"] for r in sample]

        # --- direct ---
        d_pred = gen([sparc_ex(single_prompt(schemas[r["db_id"]]["text"],
                                             r["final_intent_question"]),
                               "", r["sequence_id"], r["db_id"], r["turn_count"])
                      for r in sample])
        d_ok = [suite_correct(paths_for(r["db_id"]), extract_sql(p), g,
                              timeout_s=args.timeout_s)
                for p, r, g in zip(d_pred, sample, finals)]

        # --- part: every turn alone, gold previous SQL ---
        p_items, p_gold, p_meta = [], [], []
        for r in sample:
            g = gold[r["sequence_id"]]
            schema_text = schemas[r["db_id"]]["text"]
            for i, t in enumerate(r["turns"], 1):
                prev = g["turn_sqls"][i - 2] if i > 1 else None
                p_items.append(sparc_ex(
                    turn_prompt(schema_text, prev, t["question"]),
                    "", f"{r['sequence_id']}|part{i}", r["db_id"],
                    r["turn_count"]))
                p_gold.append(g["turn_sqls"][i - 1])
                p_meta.append((r, i))
        p_pred = gen(p_items)
        p_ok = [suite_correct(paths_for(r["db_id"]), extract_sql(p), g,
                              timeout_s=args.timeout_s)
                for p, g, (r, _i) in zip(p_pred, p_gold, p_meta)]
        turn_ok = defaultdict(list)
        for ok, (r, i) in zip(p_ok, p_meta):
            turn_ok[r["sequence_id"]].append(bool(ok))

        # --- composed (self-fed) ---
        c_state, c_prod = run_chain(sample)
        c_ok = [s is not None and suite_correct(paths_for(r["db_id"]), s, g,
                                                timeout_s=args.timeout_s)
                for s, r, g in zip(c_state, sample, finals)]

        # --- corrupt ---
        corrupt_turns = []
        for r in sample:
            g = gold[r["sequence_id"]]
            donors = [d for d in donors_by_db[r["db_id"]]
                      if d["sequence_id"] != r["sequence_id"]]
            if not donors or r["turn_count"] < 2:
                corrupt_turns.append(None)
                continue
            t = rng.randrange(2, r["turn_count"] + 1)
            donor = rng.choice(donors)
            donor_prefix = min(t - 1, donor["turn_count"])
            donor_sql = gold[donor["sequence_id"]]["turn_sqls"][donor_prefix - 1]
            corrupt_turns.append((t, donor_sql))
        x_state, _ = run_chain(sample, corrupt_turns)
        x_ok = [None if ct is None else
                (s is not None and suite_correct(paths_for(r["db_id"]), s, g,
                                                 timeout_s=args.timeout_s))
                for s, r, g, ct in zip(x_state, sample, finals, corrupt_turns)]

        for r, do, co, xo, dp, cs in zip(sample, d_ok, c_ok, x_ok, d_pred,
                                         c_state):
            records.append({
                "arm": "all", "k": r["turn_count"], "cell": cell,
                "sequence_id": r["sequence_id"], "db_id": r["db_id"],
                "direct_ok": bool(do), "composed_ok": bool(co),
                "corrupt_ok": xo if xo is None else bool(xo),
                "turn_ok": turn_ok[r["sequence_id"]],
                "direct_sql": extract_sql(dp), "composed_sql": cs})

        x_scored = [x for x in x_ok if x is not None]
        v = {"n": n, "n_parts": len(p_ok), "n_corrupt": len(x_scored),
             "direct": sum(d_ok) / n, "part": sum(p_ok) / len(p_ok),
             "composed": sum(c_ok) / n,
             "corrupt": (sum(x_scored) / len(x_scored)) if x_scored else None,
             "chain_completed": sum(c_prod) / n}
        v["headroom"] = v["composed"] - v["direct"]
        report["by_cell"][cell] = v
        print(f"  direct={v['direct']:.3f} part={v['part']:.3f} "
              f"composed={v['composed']:.3f} ({v['headroom']:+.3f}) "
              f"corrupt={v['corrupt'] if v['corrupt'] is None else round(v['corrupt'], 3)}",
              flush=True)

    # --- pooled paired stats and the machine-checkable verdict ---
    pairs = [(r["composed_ok"], r["direct_ok"]) for r in records]
    lo, hi = boot_ci(pairs, random.Random(args.seed))
    mc = mcnemar(pairs)
    n_all = len(pairs)
    pooled = {
        "n": n_all,
        "direct": sum(d for _, d in pairs) / n_all,
        "composed": sum(c for c, _ in pairs) / n_all,
        "headroom": (sum(c for c, _ in pairs) - sum(d for _, d in pairs)) / n_all,
        "ci95": [lo, hi], "mcnemar": mc,
        "part": (sum(v["part"] * v["n_parts"] for v in report["by_cell"].values())
                 / sum(v["n_parts"] for v in report["by_cell"].values())),
    }
    x_pairs = [(r["composed_ok"], r["corrupt_ok"]) for r in records
               if r["corrupt_ok"] is not None]
    pooled["corrupt"] = (sum(x for _, x in x_pairs) / len(x_pairs)
                         if x_pairs else None)
    pooled["corrupt_drop"] = (pooled["composed"] - pooled["corrupt"]
                              if x_pairs else None)
    report["pooled"] = pooled
    report["verdict"] = {
        "headroom_ci_low_gt0": lo > 0,
        "mcnemar_p": mc["p_value"],
        "mcnemar_sig": mc["p_value"] < 0.05,
        "part_ge_060": pooled["part"] >= 0.60,
        "corrupt_below_composed": (pooled["corrupt_drop"] is not None
                                   and pooled["corrupt_drop"] >= 0.05),
    }
    report["verdict"]["pass"] = all(
        report["verdict"][k] for k in
        ("headroom_ci_low_gt0", "mcnemar_sig", "part_ge_060",
         "corrupt_below_composed"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_path = args.out.with_name(args.out.stem + "_per_instance.jsonl")
    with open(per_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    args.out.write_text(json.dumps(report, indent=2))

    print(f"\nPOOLED: direct={pooled['direct']:.3f} part={pooled['part']:.3f} "
          f"composed={pooled['composed']:.3f} "
          f"headroom={pooled['headroom']:+.3f} CI[{lo:+.3f},{hi:+.3f}] "
          f"p={mc['p_value']:.4f} corrupt_drop={pooled['corrupt_drop']}",
          flush=True)
    print(f"VERDICT: {report['verdict']}", flush=True)


if __name__ == "__main__":
    main()

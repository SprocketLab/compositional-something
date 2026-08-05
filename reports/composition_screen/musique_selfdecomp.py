#!/usr/bin/env python3
"""Self-proposed decomposition: the arm that decides publishability.

The seed screen's composed arm reads `question_decomposition` from the release,
so its claim is "composition helps when someone hands you the decomposition"
(plan §26, standing rule from §25: any arm consuming released gold structure
needs a matched no-oracle arm).  Here the model proposes its own sub-questions
and the chain runs exactly as before.  If the +.080 pooled advantage survives,
the claim strengthens to "composition helps, full stop."

Design choices that matter:

* Same 300 paired dev instances as `musique_seed.py`: identical seed, draw
  order, and usable-filter.  When `--seed-per-instance` is given, the sample is
  asserted against the seed run's logged ids -- a silent sampling divergence
  would invalidate every cross-arm comparison.
* The decomposition is proposed from the QUESTION ALONE -- no paragraphs.  The
  gold decompositions were authored from composition structure, not retrieved
  text, so this is the matched condition; it also keeps the proposer from
  drifting into answering instead of decomposing.
* The proposer does NOT know the gold hop count.  Telling it k would re-import
  a structural oracle; proposed length is logged as a diagnostic instead.
* `--decompose-with base` (default) proposes with the LoRA adapter disabled:
  the seed adapter is a single-hop answer-only specialist and shifts the model
  hard toward short-phrase replies.  Both settings are "the same model" for the
  self-improvement story; `--decompose-with adapter` tests the other reading.
* Chain execution, prompts, scoring, and statistics are imported from the seed
  screen unchanged.  A step whose `#N` reference cannot be filled is skipped,
  and the chain's answer is the last one produced -- same semantics as the gold
  loop, so failure modes are comparable.
* Direct is RE-COMPUTED here rather than copied from the seed run, giving a
  same-process paired contrast; agreement with the seed run's after-records is
  reported as a drift check.  Malformed decompositions score composed_ok=False
  -- proposal failure is a failure of the composition arm, not missing data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from musique_isolation import fill, full_context, score_em
from musique_seed import boot_ci, ex, mcnemar, usable

from self.coding.atomic_data import AtomicExample
from self.coding.training import generate_predictions, load_adapter_for_evaluation

DECOMPOSE_INSTRUCTION = (
    "Break the following question into a numbered list of simpler single-hop "
    "sub-questions that answer it step by step. Later sub-questions may refer "
    "to the answer of an earlier one as #1, #2, and so on. Reply with the "
    "numbered list only, one sub-question per line.\n\n"
    "Question: {question}\nSub-questions:"
)

STEP_RE = re.compile(r"^\s*(\d+)[.):]\s*(.+?)\s*$")


def decompose_ex(question: str, sid: str) -> AtomicExample:
    return AtomicExample(
        task="musique-decompose", source_id=sid, source_group_id=sid, split="x",
        messages=({"role": "user",
                   "content": DECOMPOSE_INSTRUCTION.format(question=question)},),
        target="", evaluator={}, metadata={},
    )


def parse_decomposition(text: str, max_steps: int = 6) -> list[str] | None:
    """Numbered lines -> ordered sub-questions; None if unusable.

    Requires consecutive numbering from 1 and at least two steps -- a one-step
    "decomposition" is the direct arm wearing a costume and must not be scored
    as composition.
    """
    steps = []
    for line in text.splitlines():
        m = STEP_RE.match(line)
        if not m:
            continue
        if int(m.group(1)) != len(steps) + 1:
            break
        steps.append(m.group(2))
    if len(steps) < 2 or len(steps) > max_steps:
        return None
    return steps


def sample_like_seed_screen(dev_rows: list[dict], per_k: int, seed: int) -> dict:
    """Exactly the draw in musique_seed.py: same rng, same order, same filter."""
    by_k = defaultdict(list)
    for r in dev_rows:
        if usable(r):
            by_k[len(r["question_decomposition"])].append(r)
    rng = random.Random(seed)
    sample_by_k = {}
    for k in sorted(by_k):
        pool = by_k[k]
        sample_by_k[k] = pool if len(pool) <= per_k else rng.sample(pool, per_k)
    return sample_by_k


def run_chain(model, tok, rows, decomps, bs, records_extra):
    """The seed screen's no-oracle chain over PROPOSED steps.

    rows/decomps are parallel; decomps[j] is None when parsing failed.
    Returns final-answer list (None where no step executed).
    """
    n = len(rows)
    state = [dict() for _ in range(n)]
    last = [None] * n
    max_k = max((len(d) for d in decomps if d), default=0)
    for i in range(1, max_k + 1):
        idx, items = [], []
        for j, (r, d) in enumerate(zip(rows, decomps)):
            if d is None or len(d) < i:
                continue
            q = fill(d[i - 1], state[j])
            if "#" in q:
                records_extra[j]["unfilled_steps"] += 1
                continue
            idx.append(j)
            items.append(ex(full_context(r), q, "", f"{r['id']}|self{i}", i))
        if not items:
            continue
        preds = generate_predictions(model=model, tokenizer=tok, examples=items,
                                     batch_size=bs, max_new_tokens=32)
        for j, p in zip(idx, preds):
            a = p.split("\n")[0].strip()
            state[j][str(i)] = a
            last[j] = a
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--dev-data", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed-per-instance", type=Path, default=None,
                    help="per_instance.jsonl from the seed run; adds the gold-"
                         "decomposition composed arm and asserts sample identity")
    ap.add_argument("--decompose-with", choices=("base", "adapter"), default="base")
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--decompose-batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev_rows = [json.loads(l) for l in open(args.dev_data)]
    sample_by_k = sample_like_seed_screen(dev_rows, args.per_k, args.seed)

    gold_after = {}
    if args.seed_per_instance:
        for rec in map(json.loads, open(args.seed_per_instance)):
            if rec["phase"] == "after":
                gold_after[rec["id"]] = rec
        for k, sample in sample_by_k.items():
            ours, theirs = {r["id"] for r in sample}, \
                           {i for i, r in gold_after.items() if r["k"] == k}
            if ours != theirs:
                raise SystemExit(f"sample mismatch at k={k}: "
                                 f"{len(ours - theirs)} ids not in seed run")
        print("sample identity vs seed run: OK", flush=True)

    model, tok = load_adapter_for_evaluation(args.model, args.adapter)

    report = {"config": {k: str(v) for k, v in vars(args).items()}}
    records, decomp_log = [], []
    all_pairs_direct, all_pairs_gold = [], []

    for k, sample in sorted(sample_by_k.items()):
        n = len(sample)

        # -- direct arm, recomputed with the seeded model --
        d_items = [ex(full_context(r), r["question"], r["answer"], r["id"], k)
                   for r in sample]
        d_pred = generate_predictions(model=model, tokenizer=tok, examples=d_items,
                                      batch_size=args.batch_size, max_new_tokens=32)
        d_ok = [score_em(p, [r["answer"], *r.get("answer_aliases", [])])
                for p, r in zip(d_pred, sample)]

        # -- propose decompositions --
        p_items = [decompose_ex(r["question"], r["id"]) for r in sample]
        if args.decompose_with == "base":
            with model.disable_adapter():
                raw = generate_predictions(model=model, tokenizer=tok,
                                           examples=p_items,
                                           batch_size=args.decompose_batch_size,
                                           max_new_tokens=160)
        else:
            raw = generate_predictions(model=model, tokenizer=tok, examples=p_items,
                                       batch_size=args.decompose_batch_size,
                                       max_new_tokens=160)
        decomps = [parse_decomposition(t) for t in raw]
        for r, t, d in zip(sample, raw, decomps):
            decomp_log.append({"k": k, "id": r["id"], "question": r["question"],
                               "raw": t, "parsed": d,
                               "gold_steps": [s["question"] for s in
                                              r["question_decomposition"]]})

        # -- execute the chain with the seeded model --
        extra = [{"unfilled_steps": 0} for _ in sample]
        last = run_chain(model, tok, sample, decomps, args.batch_size, extra)
        s_ok = [l is not None and score_em(l, [r["answer"], *r.get("answer_aliases", [])])
                for l, r in zip(last, sample)]

        for r, do, so, d, e in zip(sample, d_ok, s_ok, decomps, extra):
            rec = {"k": k, "id": r["id"], "direct_ok": bool(do),
                   "self_composed_ok": bool(so),
                   "proposed_k": len(d) if d else None,
                   "decompose_failed": d is None,
                   "unfilled_steps": e["unfilled_steps"]}
            if r["id"] in gold_after:
                rec["gold_composed_ok"] = gold_after[r["id"]]["composed_ok"]
                rec["seedrun_direct_ok"] = gold_after[r["id"]]["direct_ok"]
            records.append(rec)

        pairs = [(bool(s), bool(d)) for s, d in zip(s_ok, d_ok)]
        all_pairs_direct.extend(pairs)
        out = {"n": n, "direct": sum(d_ok) / n,
               "self_composed": sum(s_ok) / n,
               "headroom": (sum(s_ok) - sum(d_ok)) / n,
               "decompose_failures": sum(d is None for d in decomps),
               "proposed_k_matches_gold": sum(1 for d in decomps if d and len(d) == k),
               "mcnemar_vs_direct": mcnemar(pairs)}
        if gold_after:
            gpairs = [(bool(s), bool(gold_after[r["id"]]["composed_ok"]))
                      for s, r in zip(s_ok, sample)]
            all_pairs_gold.extend(gpairs)
            out["gold_composed"] = sum(g for _, g in gpairs) / n
            out["mcnemar_vs_gold_composed"] = mcnemar(gpairs)
            drift = sum(1 for rec, do in zip(
                (gold_after[r["id"]] for r in sample), d_ok)
                if rec["direct_ok"] != bool(do))
            out["direct_drift_vs_seedrun"] = drift
        report[str(k)] = out
        g = f" gold_composed={out['gold_composed']:.3f}" if gold_after else ""
        print(f"  [selfdecomp] k={k}: direct={out['direct']:.3f} "
              f"self_composed={out['self_composed']:.3f} ({out['headroom']:+.3f})"
              f"{g} parse_fail={out['decompose_failures']}", flush=True)

    rng = random.Random(args.seed)
    pooled = {"n": len(all_pairs_direct),
              "self_composed_minus_direct":
                  (sum(c for c, _ in all_pairs_direct)
                   - sum(d for _, d in all_pairs_direct)) / len(all_pairs_direct),
              "mcnemar": mcnemar(all_pairs_direct),
              "bootstrap_ci95": boot_ci(all_pairs_direct, rng)}
    if all_pairs_gold:
        pooled["self_composed_minus_gold_composed"] = \
            (sum(c for c, _ in all_pairs_gold)
             - sum(g for _, g in all_pairs_gold)) / len(all_pairs_gold)
        pooled["mcnemar_vs_gold"] = mcnemar(all_pairs_gold)
        pooled["bootstrap_ci95_vs_gold"] = boot_ci(all_pairs_gold, rng)
    report["pooled"] = pooled
    print(f"pooled: self-direct {pooled['self_composed_minus_direct']:+.3f} "
          f"CI {pooled['bootstrap_ci95']} p={pooled['mcnemar']['p_value']:.4f}",
          flush=True)

    with open(args.out_dir / "per_instance.jsonl", "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    with open(args.out_dir / "decompositions.jsonl", "w") as f:
        for rec in decomp_log:
            f.write(json.dumps(rec) + "\n")
    with open(args.out_dir / "musique_selfdecomp.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out_dir}/musique_selfdecomp.json", flush=True)


if __name__ == "__main__":
    main()

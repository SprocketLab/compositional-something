#!/usr/bin/env python3
"""Round 1, stage 1: pseudo-label 2-hop questions with the seed's own chain.

Runs the partition seed's no-oracle composition chain (gold decomposition
STRUCTURE, self-fed answers, full 20-paragraph context -- train-time use of the
release DAG is sanctioned by plan §4.2; the student never sees it) over 2-hop
composition-source questions, guards the results, and writes a label file that
`musique_round1_train.py` consumes.  Also generates the seed's DIRECT answer to
every question at the same time: that is the self-distillation control arm's
label, and buying it here costs one generation instead of a second job.

Measurements, following clutrr_round1.py:
  A. mechanism   -- composed vs direct accuracy on the pool (train distribution)
  B. label quality -- precision/recall of each guard level against gold.  Gold
     is used to MEASURE guards, never to accept examples (plan §10.6).

Guard levels (plan §10, the cheap subset; two-view agreement deferred):
  L0 chain produced a final answer at the sink step
  L1 L0 + final answer is a normalized span of the context
  L2 L1 + final short/non-generic + bridge answer passes the same three checks
The label file records per-guard flags; acceptance for training is L2.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from musique_isolation import fill, full_context, normalize, score_em
from musique_seed import ex, usable

from self.coding.training import generate_predictions, load_adapter_for_evaluation

GENERIC = {normalize(g) for g in (
    "the person", "a person", "person", "he", "she", "it", "they", "them",
    "this", "that", "these", "those", "unknown", "none", "n/a", "yes", "no",
    "the man", "the woman", "the city", "the country", "the answer",
)}


def answer_checks(ans: str, context_norm: str) -> dict:
    """Plan §10.1 local checks, applied to one answer string."""
    a = normalize(ans)
    return {
        "nonempty": bool(a),
        "short": bool(a) and len(ans.split()) <= 8 and len(ans) <= 60,
        "not_generic": bool(a) and a not in GENERIC,
        "span": bool(a) and a in context_norm,
    }


def chunked_generate(model, tok, items, bs, chunk, label):
    preds = []
    for start in range(0, len(items), chunk):
        preds.extend(generate_predictions(
            model=model, tokenizer=tok, examples=items[start:start + chunk],
            batch_size=bs, max_new_tokens=32))
        print(f"  [{label}] {min(start + chunk, len(items))}/{len(items)}",
              flush=True)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--partition", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pool-size", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_ids = set(json.load(open(args.partition))["seed_ids"])
    pool = [r for l in open(args.train_data)
            for r in [json.loads(l)]
            if r["id"] not in seed_ids and r["id"].startswith("2hop")
            and usable(r) and len(r["question_decomposition"]) == 2]
    print(f"composition-source 2-hop usable pool: {len(pool)}", flush=True)
    rng = random.Random(args.seed)
    if len(pool) > args.pool_size:
        pool = rng.sample(pool, args.pool_size)
    print(f"labeling {len(pool)} examples", flush=True)

    model, tok = load_adapter_for_evaluation(args.model, args.adapter)

    # step 1 of every chain has no #refs by construction
    s1_items = [ex(full_context(r), r["question_decomposition"][0]["question"],
                   "", f"{r['id']}|1", 2) for r in pool]
    s1 = chunked_generate(model, tok, s1_items, args.batch_size, args.chunk, "step1")
    bridges = [p.split("\n")[0].strip() for p in s1]

    # step 2: substitute the model's own bridge answer
    idx, s2_items = [], []
    for j, (r, b) in enumerate(zip(pool, bridges)):
        q = fill(r["question_decomposition"][1]["question"], {"1": b})
        if "#" in q:
            continue
        idx.append(j)
        s2_items.append(ex(full_context(r), q, "", f"{r['id']}|2", 2))
    s2 = chunked_generate(model, tok, s2_items, args.batch_size, args.chunk, "step2")
    finals = [None] * len(pool)
    for j, p in zip(idx, s2):
        finals[j] = p.split("\n")[0].strip()

    d_items = [ex(full_context(r), r["question"], "", r["id"], 2) for r in pool]
    d_pred = chunked_generate(model, tok, d_items, args.batch_size, args.chunk,
                              "direct")
    directs = [p.split("\n")[0].strip() for p in d_pred]

    rows, counts = [], {"L0": [0, 0], "L1": [0, 0], "L2": [0, 0]}
    n_composed_ok = n_direct_ok = 0
    for r, b, f, d in zip(pool, bridges, finals, directs):
        ctx_norm = normalize(full_context(r))
        golds = [r["answer"], *r.get("answer_aliases", [])]
        fc = answer_checks(f or "", ctx_norm)
        bc = answer_checks(b, ctx_norm)
        levels = {
            "L0": f is not None,
            "L1": f is not None and fc["span"],
            "L2": (f is not None and all(fc.values()) and all(bc.values())),
        }
        composed_ok = f is not None and score_em(f, golds)
        direct_ok = score_em(d, golds)
        n_composed_ok += composed_ok
        n_direct_ok += direct_ok
        for lv, ok in levels.items():
            if ok:
                counts[lv][0] += 1
                counts[lv][1] += composed_ok
        rows.append({"id": r["id"], "question": r["question"],
                     "composed": f, "bridge": b, "direct": d,
                     "final_checks": fc, "bridge_checks": bc,
                     "levels": levels, "accept": levels["L2"],
                     "composed_correct": bool(composed_ok),
                     "direct_correct": bool(direct_ok),
                     "gold": r["answer"]})

    n = len(pool)
    guard_table = {}
    for lv, (acc, correct) in counts.items():
        guard_table[lv] = {
            "accepted": acc,
            "precision": correct / acc if acc else None,
            "recall": correct / n_composed_ok if n_composed_ok else None,
        }
    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "pool": n,
        "mechanism": {"composed_acc": n_composed_ok / n,
                      "direct_acc": n_direct_ok / n,
                      "headroom": (n_composed_ok - n_direct_ok) / n},
        "guards": guard_table,
    }
    with open(args.out_dir / "round1_labels.jsonl", "w") as f_:
        for row in rows:
            f_.write(json.dumps(row) + "\n")
    with open(args.out_dir / "round1_label_report.json", "w") as f_:
        json.dump(report, f_, indent=2)
    print(f"A mechanism: composed={report['mechanism']['composed_acc']:.3f} "
          f"direct={report['mechanism']['direct_acc']:.3f} "
          f"({report['mechanism']['headroom']:+.3f})", flush=True)
    for lv, g in guard_table.items():
        p = f"{g['precision']:.3f}" if g["precision"] is not None else "-"
        rc = f"{g['recall']:.3f}" if g["recall"] is not None else "-"
        print(f"B {lv}: accepted={g['accepted']}/{n} precision={p} recall={rc}",
              flush=True)
    print(f"wrote {args.out_dir}/round1_labels.jsonl", flush=True)


if __name__ == "__main__":
    main()

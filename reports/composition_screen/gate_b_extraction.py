#!/usr/bin/env python3
"""Gate B: does sentence extraction make composition beat direct on REAL stories?

Route 2 (decomposing a real long story) tied direct prediction, .425 vs .468,
because sub-chain questions were asked over the whole story and the model still
had to find the relevant sentences among distractors.  Route 1 (concatenating
short stories) beat direct but built composites twice as hard as real chains.

Extraction is the untried middle: give each sub-chain only its own sentences, so
a part becomes a genuine short story while the composite stays a real instance.

Everything is paired on the same instances:
  direct            -- ask the whole chain over the whole story
  part (full)       -- ask a sub-chain over the whole story        [route 2]
  part (extracted)  -- ask the same sub-chain over its own span    [new]
  composed_*        -- fold the part answers through the kinship rule

PASS requires extracted-part accuracy >= .80 AND composed_extracted exceeding
direct by >= 5 points.  Gold is read only to score; never to build a label.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kinship import compose, fold
from story_extract import extract, is_clean_chain, node_names
from clutrr_decompose import all_splits, make_example, parse_relation, prompt_for

from self.coding.atomic_data import AtomicExample
from self.coding.training import generate_predictions, load_adapter_for_evaluation


def ex(story: str, a: str, b: str, target: str, sid: str) -> AtomicExample:
    return AtomicExample(
        task="clutrr", source_id=sid, source_group_id=sid, split="pool",
        messages=({"role": "user", "content": prompt_for(story, a, b)},),
        target=target, evaluator={}, metadata={},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--model", "--adapter", "--pool-csv"):
        ap.add_argument(a, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-k", type=int, default=60)
    ap.add_argument("--max-hop", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.pool_csv)) if is_clean_chain(r)]
    by_k = defaultdict(list)
    for r in rows:
        k = len(ast.literal_eval(r["edge_types"]))
        if k >= 5:
            by_k[k].append(r)
    rng = random.Random(args.seed)

    model, tok = load_adapter_for_evaluation(args.model, Path(args.adapter))
    report: dict = {"by_k": {}}

    for k in sorted(by_k):
        sample = by_k[k]
        if len(sample) > args.per_k:
            sample = rng.sample(sample, args.per_k)
        cut = all_splits(k, args.max_hop)[0]           # canonical fewest-chunks cut
        names = [node_names(r) for r in sample]

        direct = [ex(r["clean_story"], nm[0], nm[k], r["target_text"], r["id"])
                  for r, nm in zip(sample, names)]
        full_parts, ext_parts, owner = [], [], []
        for idx, (r, nm) in enumerate(zip(sample, names)):
            for (i, j) in cut:
                seg = nm[i : j + 1]
                gold_seg = fold(ast.literal_eval(r["edge_types"])[i:j]) or "?"
                full_parts.append(ex(r["clean_story"], seg[0], seg[-1], gold_seg, f"{r['id']}|f{i}-{j}"))
                ext_parts.append(ex(extract(r["clean_story"], seg), seg[0], seg[-1],
                                    gold_seg, f"{r['id']}|e{i}-{j}"))
                owner.append(idx)

        print(f"k={k}: {len(direct)} instances, {len(cut)} chunks, {len(full_parts)} part queries x2",
              flush=True)
        d_pred = generate_predictions(model=model, tokenizer=tok, examples=direct,
                                      batch_size=args.batch_size, max_new_tokens=12)
        f_pred = generate_predictions(model=model, tokenizer=tok, examples=full_parts,
                                      batch_size=args.batch_size, max_new_tokens=12)
        e_pred = generate_predictions(model=model, tokenizer=tok, examples=ext_parts,
                                      batch_size=args.batch_size, max_new_tokens=12)

        got_full, got_ext = defaultdict(list), defaultdict(list)
        for o, p in zip(owner, f_pred):
            got_full[o].append(parse_relation(p))
        for o, p in zip(owner, e_pred):
            got_ext[o].append(parse_relation(p))

        def fold_chain(chain):
            if any(c is None for c in chain):
                return None
            cur = chain[0]
            for nxt in chain[1:]:
                cur = compose(cur, nxt) if cur else None
            return cur

        n = len(sample)
        d_ok = sum(parse_relation(p) == r["target_text"] for p, r in zip(d_pred, sample))
        cf = sum(fold_chain(got_full[i]) == r["target_text"] for i, r in enumerate(sample))
        ce = sum(fold_chain(got_ext[i]) == r["target_text"] for i, r in enumerate(sample))
        # part-level accuracy, scored against the algebra's fold of that segment
        pf = sum(parse_relation(p) == e.target for p, e in zip(f_pred, full_parts) if e.target != "?")
        pe = sum(parse_relation(p) == e.target for p, e in zip(e_pred, ext_parts) if e.target != "?")
        scored = sum(1 for e in full_parts if e.target != "?")

        report["by_k"][k] = {
            "n": n, "chunks": len(cut), "scored_parts": scored,
            "direct": d_ok / n,
            "part_full_story": pf / scored if scored else None,
            "part_extracted": pe / scored if scored else None,
            "composed_full_story": cf / n,
            "composed_extracted": ce / n,
            "delta_vs_direct": (ce - d_ok) / n,
        }
        v = report["by_k"][k]
        print(f"k={k}: direct={v['direct']:.3f} | part full={v['part_full_story']:.3f} "
              f"ext={v['part_extracted']:.3f} | composed full={v['composed_full_story']:.3f} "
              f"ext={v['composed_extracted']:.3f} | delta={v['delta_vs_direct']:+.3f}", flush=True)

    tot = sum(v["n"] for v in report["by_k"].values())
    agg = lambda key: sum(v[key] * v["n"] for v in report["by_k"].values()) / tot
    report["overall"] = {m: agg(m) for m in
                         ("direct", "composed_full_story", "composed_extracted", "delta_vs_direct")}
    sp = sum(v["part_extracted"] * v["scored_parts"] for v in report["by_k"].values())
    report["overall"]["part_extracted"] = sp / sum(v["scored_parts"] for v in report["by_k"].values())
    report["gate_b_pass"] = bool(
        report["overall"]["part_extracted"] >= 0.80
        and report["overall"]["delta_vs_direct"] >= 0.05
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2), flush=True)
    print("GATE B:", "PASS" if report["gate_b_pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()

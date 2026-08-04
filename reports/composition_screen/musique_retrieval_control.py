#!/usr/bin/env python3
"""Strip the retrieval oracle out of the composed arm.

`musique_isolation.py` gave every sub-question its own supporting paragraph via
`paragraph_support_idx` -- a GOLD annotation.  The direct arm got all 20
paragraphs and had to find the evidence itself.  So the +.133 headroom mixes two
effects: decomposing the question, and being handed the right paragraph.

The shortcut control quantified how much that oracle is worth: with the gold
paragraph but the upstream entity BLANKED, the model still answered .620-.650.
The paragraph alone carries most of a step.

This run removes the oracle.  The self-fed composed arm sees the full
20-paragraph context at every step, exactly like direct, so the only remaining
difference between arms is that one asks a decomposed question and the other
does not.  Direct is recomputed on the same instances in the same run rather
than read from the earlier report, so the comparison is strictly paired.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from musique_isolation import Runner, fill, full_context, own_paragraph, prompt, score_em


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    by_k = defaultdict(list)
    for r in rows:
        if all(s.get("paragraph_support_idx") is not None
               and s["paragraph_support_idx"] < len(r["paragraphs"])
               for s in r["question_decomposition"]):
            by_k[len(r["question_decomposition"])].append(r)

    # same seed and same draw order as musique_isolation.py -> same instances
    rng = random.Random(args.seed)
    runner = Runner(args.model, args.batch_size)
    report: dict = {"by_hops": {}}

    for k in sorted(by_k):
        sample = by_k[k]
        if len(sample) > args.per_k:
            sample = rng.sample(sample, args.per_k)
        n = len(sample)
        print(f"== {k} hops: {n} instances ==", flush=True)

        d_pred = runner.generate([prompt(full_context(r), r["question"]) for r in sample])
        d_ok = sum(score_em(p, [r["answer"], *r.get("answer_aliases", [])])
                   for p, r in zip(d_pred, sample))

        # self-fed chain, two retrieval conditions, otherwise identical
        results = {}
        for tag, ctx in (("full", full_context), ("own", None)):
            state = [dict() for _ in sample]
            last = [None] * n
            for i in range(1, k + 1):
                idx, prompts_i = [], []
                for j, r in enumerate(sample):
                    step = r["question_decomposition"][i - 1]
                    q = fill(step["question"], state[j])
                    if "#" in q:
                        continue
                    context = ctx(r) if ctx else own_paragraph(r, step)
                    idx.append(j)
                    prompts_i.append(prompt(context, q))
                if not prompts_i:
                    continue
                preds = runner.generate(prompts_i)
                for j, p in zip(idx, preds):
                    ans = p.split("\n")[0].strip()
                    state[j][str(i)] = ans
                    last[j] = ans
            results[tag] = sum(
                l is not None and score_em(l, [r["answer"], *r.get("answer_aliases", [])])
                for l, r in zip(last, sample)
            ) / n

        v = {
            "n": n,
            "direct": d_ok / n,
            "composed_gold_retrieval": results["own"],
            "composed_no_oracle": results["full"],
        }
        v["headroom_with_oracle"] = v["composed_gold_retrieval"] - v["direct"]
        v["headroom_no_oracle"] = v["composed_no_oracle"] - v["direct"]
        v["oracle_worth"] = v["composed_gold_retrieval"] - v["composed_no_oracle"]
        report["by_hops"][k] = v
        print(f"  direct={v['direct']:.3f} | composed(oracle)={v['composed_gold_retrieval']:.3f} "
              f"({v['headroom_with_oracle']:+.3f}) | composed(no oracle)={v['composed_no_oracle']:.3f} "
              f"({v['headroom_no_oracle']:+.3f}) | oracle worth {v['oracle_worth']:+.3f}", flush=True)

    tot = sum(v["n"] for v in report["by_hops"].values())
    report["overall"] = {
        m: sum(v[m] * v["n"] for v in report["by_hops"].values()) / tot
        for m in ("direct", "composed_gold_retrieval", "composed_no_oracle",
                  "headroom_with_oracle", "headroom_no_oracle", "oracle_worth")
    }
    args.out.write_text(json.dumps(report, indent=2))
    o = report["overall"]
    print(json.dumps(o, indent=2), flush=True)
    print(f"\nHEADROOM WITHOUT THE RETRIEVAL ORACLE: {o['headroom_no_oracle']:+.3f}  "
          f"(oracle was worth {o['oracle_worth']:+.3f})", flush=True)


if __name__ == "__main__":
    main()

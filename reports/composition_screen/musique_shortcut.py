#!/usr/bin/env python3
"""Is MuSiQue's composed gain real, or does the last paragraph pin the answer?

`composed_self` scores the FINAL step's answer, and it landed at .647 against an
independent-error prediction of ~.34 (ratio ~2.0).  Some of that is genuine
positive dependence between steps.  But it is equally consistent with the last
hop not needing the earlier ones at all -- if the final paragraph determines the
answer regardless of which entity was substituted into `#N`, then the "composed"
arm is a single-hop lookup wearing a multi-hop costume, and the +.133 headroom is
an artifact.  MuSiQue has documented shortcut problems; this is the control.

Four arms on the FINAL step only, paired on the same instances:

  gold      #N <- the gold upstream answer            (ceiling)
  self      #N <- the model's own upstream answer     (= composed_self)
  corrupt   #N <- a gold answer sampled from a DIFFERENT instance
                  (type-plausible, definitely wrong)
  blank     #N <- left as the literal token           (no entity at all)

If `corrupt` and `blank` stay near `gold`, the final paragraph alone determines
the answer and the multi-hop structure is decorative.  If they collapse, the
composition is doing real work.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from musique_isolation import Runner, fill, own_paragraph, prompt, score_em


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

    rng = random.Random(args.seed)
    runner = Runner(args.model, args.batch_size)
    report: dict = {"by_hops": {}}

    for k in sorted(by_k):
        sample = by_k[k]
        if len(sample) > args.per_k:
            sample = rng.sample(sample, args.per_k)
        # decoy pool: final answers of OTHER instances, so substitutions stay
        # type-plausible (a person/place/date) while being certainly wrong
        pool = [r["answer"] for r in by_k[k]]

        arms = {"gold": [], "corrupt": [], "blank": []}
        golds, kept = [], []
        for r in sample:
            step = r["question_decomposition"][-1]
            if "#" not in step["question"]:
                continue                      # final step has no upstream reference
            gold_up = {str(i): s["answer"] for i, s in enumerate(r["question_decomposition"], 1)}
            decoy = rng.choice([a for a in pool if a != r["answer"]])
            para = own_paragraph(r, step)
            arms["gold"].append(prompt(para, fill(step["question"], gold_up)))
            arms["corrupt"].append(prompt(para, fill(step["question"], {j: decoy for j in gold_up})))
            arms["blank"].append(prompt(para, step["question"]))
            golds.append([step["answer"]])
            kept.append(r["id"])

        print(f"== {k} hops: {len(golds)} final steps with an upstream reference ==", flush=True)
        if not golds:
            continue
        acc = {}
        for name, prompts in arms.items():
            preds = runner.generate(prompts)
            acc[name] = sum(score_em(p, g) for p, g in zip(preds, golds)) / len(golds)
            print(f"  {name:8s} = {acc[name]:.3f}", flush=True)

        acc["n"] = len(golds)
        acc["corrupt_drop"] = acc["gold"] - acc["corrupt"]
        acc["blank_drop"] = acc["gold"] - acc["blank"]
        report["by_hops"][k] = acc
        print(f"  -> corrupt costs {acc['corrupt_drop']:+.3f}, blank costs {acc['blank_drop']:+.3f}",
              flush=True)

    n = sum(v["n"] for v in report["by_hops"].values())
    report["overall"] = {m: sum(v[m] * v["n"] for v in report["by_hops"].values()) / n
                         for m in ("gold", "corrupt", "blank")}
    o = report["overall"]
    o["corrupt_drop"] = o["gold"] - o["corrupt"]
    o["blank_drop"] = o["gold"] - o["blank"]
    # A shortcut is indicated when corrupting the upstream entity barely hurts.
    report["shortcut_suspected"] = bool(o["corrupt_drop"] < 0.15)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(o, indent=2), flush=True)
    print("SHORTCUT SUSPECTED:" if report["shortcut_suspected"] else "COMPOSITION IS LOAD-BEARING:",
          f"corrupting the upstream answer costs {o['corrupt_drop']:+.3f}", flush=True)


if __name__ == "__main__":
    main()

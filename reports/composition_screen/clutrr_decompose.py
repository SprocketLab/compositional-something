#!/usr/bin/env python3
"""Does decomposing a long CLUTRR chain beat predicting it directly?

The seed is reliable at k<=4 (.95 holdout) and fails beyond it (~.45 at k>=5).
That gap is the headroom compositional self-improvement exploits, so the
question is whether it can actually be collected: split a k-hop chain into
sub-chains of at most `max_hop` hops, ask the seed each sub-chain, and combine
the answers with the kinship composition rule derived from 2-hop training data.

Both arms see the same story and the same instances, so the comparison is
paired.  The composition rule is exact on all 62 observed relation pairs, so
unlike addition there is no boundary case for a guard to reject -- any loss
comes from the model's sub-chain predictions, not from the rule.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from kinship import compose as kin_compose

from self.coding.atomic_data import AtomicExample
from self.coding.training import generate_predictions, load_adapter_for_evaluation

RELATIONS = [
    "aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law",
    "granddaughter", "grandfather", "grandmother", "grandson", "mother",
    "mother-in-law", "nephew", "niece", "sister", "son", "son-in-law", "uncle",
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    return " ".join(text.split())


def prompt_for(story: str, a: str, b: str) -> str:
    return (
        "Read the story and state the family relation.\n\n"
        f"Story: {story}\n\n"
        f"Question: How is {b} related to {a}? In other words, {b} is {a}'s what?\n"
        f"Answer with exactly one word from this list: {', '.join(RELATIONS)}.\n"
        "Answer:"
    )


def parse_relation(pred: str) -> str | None:
    p = normalize(pred)
    hits = [r for r in RELATIONS if re.search(rf"\b{re.escape(normalize(r))}\b", p)]
    return max(hits, key=len) if hits else None


def composition_table(train_csv: Path) -> dict[tuple[str, str], str]:
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in csv.DictReader(open(train_csv)):
        et = ast.literal_eval(r["edge_types"])
        if len(et) == 2:
            table[(et[0], et[1])][r["target_text"]] += 1
    return {k: v.most_common(1)[0][0] for k, v in table.items()}


def all_splits(k: int, max_hop: int) -> list[list[tuple[int, int]]]:
    """Every contiguous cut of 0..k into chunks of 1..max_hop hops.

    A fixed midpoint cut fails often: 21.7% of sub-chains fold to a relation
    outside the 18-name answer space (e.g. great-grandmother), which the model
    cannot express.  Searching cuts recovers a working one for 99.1% of
    instances, and needs no gold -- an unusable cut is detected because the
    composition rule returns None.
    """

    def rec(start: int):
        if start == k:
            yield []
            return
        for end in range(start + 1, min(start + max_hop, k) + 1):
            for rest in rec(end):
                yield [(start, end)] + rest

    out = [s for s in rec(0) if len(s) >= 2]
    out.sort(key=lambda s: (len(s), max(b - a for a, b in s)))   # fewest chunks first
    return out


def sub_pairs(k: int, max_hop: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i + 1, min(i + max_hop, k) + 1)]


def make_example(row: dict, names: list[str], i: int, j: int, tag: str) -> AtomicExample:
    return AtomicExample(
        task="clutrr", source_id=f"{row['id']}|{tag}|{i}-{j}", source_group_id=str(row["id"]),
        split="test",
        messages=({"role": "user", "content": prompt_for(row["clean_story"], names[i], names[j])},),
        target=str(row["target_text"]), evaluator={},
        metadata={"i": i, "j": j},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hop", type=int, default=4)
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    table = composition_table(Path(args.train_csv))
    print(f"composition rule: {len(table)} relation pairs", flush=True)

    rows = list(csv.DictReader(open(args.test_csv)))
    by_k = defaultdict(list)
    dropped = 0
    for r in rows:
        k = len(ast.literal_eval(r["edge_types"]))
        names = [g.split(":")[0] for g in r["genders"].split(",")]
        # Only 635/1048 rows are a clean 0->k path with a name per node; the
        # rest reuse entities or query a different pair, so node indices are
        # not a usable handle for sub-chains.  Both arms use the same filtered
        # instances, so the paired comparison is unaffected.
        is_path = ast.literal_eval(r["story_edges"]) == [(i, i + 1) for i in range(k)]
        if is_path and ast.literal_eval(r["query_edge"]) == (0, k) and len(names) >= k + 1:
            by_k[k].append(r)
        else:
            dropped += 1
    print(f"usable clean-path rows: {sum(len(v) for v in by_k.values())} (dropped {dropped})", flush=True)
    rng = random.Random(args.seed)

    model, tokenizer = load_adapter_for_evaluation(args.model, Path(args.adapter))
    report: dict = {"max_hop": args.max_hop, "by_k": {}}

    for k in sorted(by_k):
        if k <= args.max_hop:
            continue
        sample = by_k[k]
        if len(sample) > args.per_k:
            sample = rng.sample(sample, args.per_k)
        names = [[g.split(":")[0] for g in r["genders"].split(",")] for r in sample]
        pairs = sub_pairs(k, args.max_hop)
        cuts = all_splits(k, args.max_hop)

        direct = [make_example(r, nm, 0, k, "direct") for r, nm in zip(sample, names)]
        parts, owner = [], []
        for idx, (r, nm) in enumerate(zip(sample, names)):
            for (i, j) in pairs:
                parts.append(make_example(r, nm, i, j, "part"))
                owner.append(idx)

        print(f"k={k}: {len(direct)} direct + {len(parts)} sub-chain queries "
              f"({len(pairs)}/instance), {len(cuts)} candidate cuts", flush=True)
        d_pred = generate_predictions(model=model, tokenizer=tokenizer, examples=direct,
                                      batch_size=args.batch_size, max_new_tokens=12)
        p_pred = generate_predictions(model=model, tokenizer=tokenizer, examples=parts,
                                      batch_size=args.batch_size, max_new_tokens=12)

        answer = defaultdict(dict)
        for o, ex, pr in zip(owner, parts, p_pred):
            answer[o][(ex.metadata["i"], ex.metadata["j"])] = parse_relation(pr)

        d_ok = comp_ok = unresolved = 0
        detail = []
        for idx, (r, dp) in enumerate(zip(sample, d_pred)):
            gold = r["target_text"]
            direct_ok = parse_relation(dp) == gold
            composed, used = None, None
            for cut in cuts:                      # first cut whose composition resolves
                subs = [answer[idx].get(seg) for seg in cut]
                if any(s is None for s in subs):
                    continue
                cur = subs[0]
                for nxt in subs[1:]:
                    cur = kin_compose(cur, nxt) if cur else None
                    if cur is None:
                        break
                if cur is not None:
                    composed, used = cur, cut
                    break
            if composed is None:
                unresolved += 1
            d_ok += direct_ok
            comp_ok += composed == gold
            detail.append({"id": r["id"], "k": k, "gold": gold, "direct": parse_relation(dp),
                           "composed": composed, "cut": used,
                           "direct_ok": direct_ok, "composed_ok": composed == gold})

        n = len(sample)
        report["by_k"][k] = {
            "n": n, "n_candidate_cuts": len(cuts), "queries_per_instance": len(pairs),
            "direct_accuracy": d_ok / n, "composed_accuracy": comp_ok / n,
            "delta": (comp_ok - d_ok) / n, "unresolved": unresolved / n, "detail": detail,
        }
        print(f"k={k}: direct={d_ok/n:.3f}  composed={comp_ok/n:.3f}  "
              f"delta={(comp_ok-d_ok)/n:+.3f}  unresolved={unresolved/n:.3f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    summary = {k: {m: v[m] for m in ("n", "direct_accuracy", "composed_accuracy", "delta", "unresolved")}
               for k, v in report["by_k"].items()}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

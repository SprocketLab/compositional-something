#!/usr/bin/env python3
"""Round 1 on REAL CLUTRR long stories, with extraction and pseudo-replay.

Differs from `clutrr_round1.py` in the two ways that invalidated it:

  * Inputs are real k>=5 stories from a disjoint CLUTRR config, not composites
    built by concatenating two short stories.  Constructed k=8 turned out to be
    twice as hard as real k=8 (.260 vs .507), so training on it transferred
    poorly.  Sub-chains are made easy by *extracting their sentences* instead.

  * Training mixes in pseudo-replay: fresh k<=4 chains the seed never trained
    on, labelled by the seed's OWN predictions.  Training on frontier labels
    alone cost .055-.085 on k<=4.  Addition replays the same way -- pseudo, not
    gold -- so this introduces no supervision the method does not already use.

Gold is read from the pool only to audit pseudo-label precision after the fact,
never on the labelling path.  The evaluation split is untouched by both.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kinship import compose
from story_extract import extract, is_clean_chain, node_names
from clutrr_decompose import all_splits, parse_relation, prompt_for

from self.coding.atomic_data import AtomicExample
from self.coding.training import (
    generate_predictions,
    load_adapter_for_evaluation,
    load_adapter_for_training,
    train_lora,
)


def ex(story: str, a: str, b: str, target: str, sid: str, k: int) -> AtomicExample:
    return AtomicExample(
        task="clutrr", source_id=sid, source_group_id=sid, split="x",
        messages=({"role": "user", "content": prompt_for(story, a, b)},),
        target=target, evaluator={}, metadata={"k": k},
    )


def gen(model, tok, items, bs):
    return generate_predictions(model=model, tokenizer=tok, examples=items,
                                batch_size=bs, max_new_tokens=12)


def story_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode()).hexdigest()


def eval_test(model, tok, test_csv, per_k, bs, rng, tag):
    """Accuracy per k on the held-out split, reported k<=4 and k>=5 separately.

    The blended number hid an .085 forgetting effect behind a flat frontier in
    the previous round.
    """
    by_k = defaultdict(list)
    for r in csv.DictReader(open(test_csv)):
        k = len(ast.literal_eval(r["edge_types"]))
        names = [g.split(":")[0] for g in r["genders"].split(",")]
        if len(names) < k + 1 or ast.literal_eval(r["query_edge"]) != (0, k):
            continue
        by_k[k].append((r, names))
    out, hit, tot = {}, Counter(), Counter()
    for k in sorted(by_k):
        pool = by_k[k]
        pool = pool if len(pool) <= per_k else rng.sample(pool, per_k)
        items = [ex(r["clean_story"], nm[0], nm[k], r["target_text"], r["id"], k) for r, nm in pool]
        preds = gen(model, tok, items, bs)
        ok = sum(parse_relation(p) == i.target for p, i in zip(preds, items))
        out[k] = {"n": len(items), "accuracy": ok / len(items)}
        band = "k<=4" if k <= 4 else "k>=5"
        hit[band] += ok
        tot[band] += len(items)
        print(f"  [{tag}] k={k}: n={len(items)} acc={ok/len(items):.3f}", flush=True)
    out["bands"] = {b: {"n": tot[b], "accuracy": hit[b] / tot[b]} for b in tot}
    for b, v in out["bands"].items():
        print(f"  [{tag}] {b}: n={v['n']} acc={v['accuracy']:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--model", "--adapter", "--pool-csv", "--train-csv", "--test-csv"):
        ap.add_argument(a, required=True)
    ap.add_argument("--unseen-ids", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--replay-ratio", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--max-hop", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    report: dict = {"config": vars(args) | {"out_dir": str(args.out_dir)}}

    # --- pool must not overlap the evaluation split; verify, do not assume ---
    test_rows = list(csv.DictReader(open(args.test_csv)))
    test_hashes = {story_hash(r["clean_story"]) for r in test_rows}
    pool = [r for r in csv.DictReader(open(args.pool_csv)) if is_clean_chain(r)]
    pool = [r for r in pool if len(ast.literal_eval(r["edge_types"])) >= 5]
    overlap = sum(story_hash(r["clean_story"]) in test_hashes for r in pool)
    print(f"pool k>=5 clean-chain rows: {len(pool)}  overlap with eval split: {overlap}", flush=True)
    if overlap:
        raise SystemExit(f"pool overlaps the evaluation split on {overlap} stories")
    report["pool"] = {"n": len(pool), "overlap_with_test": overlap}

    model, tok = load_adapter_for_evaluation(args.model, Path(args.adapter))

    print("== before: seed on the REAL test split ==", flush=True)
    report["test_before"] = eval_test(model, tok, args.test_csv, args.per_k, args.batch_size, rng, "before")

    # --- pseudo-label the pool: extraction + composition, no gold on this path ---
    print("== pseudo-labelling the unlabeled pool ==", flush=True)
    part_items, owner, segs = [], [], []
    for idx, r in enumerate(pool):
        k = len(ast.literal_eval(r["edge_types"]))
        nm = node_names(r)
        cut = all_splits(k, args.max_hop)[0]
        for (i, j) in cut:
            seg = nm[i : j + 1]
            part_items.append(ex(extract(r["clean_story"], seg), seg[0], seg[-1], "", f"{r['id']}|{i}-{j}", j - i))
            owner.append(idx)
            segs.append((i, j))
    print(f"  {len(part_items)} extracted sub-chain queries over {len(pool)} stories", flush=True)
    p_pred = gen(model, tok, part_items, args.batch_size)

    parts_by_owner = defaultdict(list)
    for o, p in zip(owner, p_pred):
        parts_by_owner[o].append(parse_relation(p))

    labelled, unresolved = [], 0
    for idx, r in enumerate(pool):
        chain = parts_by_owner[idx]
        cur = chain[0] if chain and chain[0] else None
        for nxt in chain[1:]:
            cur = compose(cur, nxt) if (cur and nxt) else None
        if cur is None:
            unresolved += 1
            continue
        k = len(ast.literal_eval(r["edge_types"]))
        nm = node_names(r)
        labelled.append((r, ex(r["clean_story"], nm[0], nm[k], cur, r["id"], k)))
    print(f"  accepted {len(labelled)} / {len(pool)}  (unresolved {unresolved})", flush=True)

    # audit only, after labelling: how good were those labels?
    prec = Counter()
    for r, item in labelled:
        prec["n"] += 1
        prec["correct"] += item.target == r["target_text"]
    report["pseudo_labels"] = {
        "n_pool": len(pool), "n_accepted": len(labelled), "unresolved": unresolved,
        "precision_vs_gold": prec["correct"] / max(prec["n"], 1),
    }
    print(f"  pseudo-label precision (audit only): {report['pseudo_labels']['precision_vs_gold']:.3f}",
          flush=True)

    # --- pseudo-replay: seed's own predictions on fresh k<=4 chains ---
    keep = set(json.loads(Path(args.unseen_ids).read_text()))
    fresh = [r for r in csv.DictReader(open(args.train_csv))
             if r["id"] in keep and is_clean_chain(r)
             and len(ast.literal_eval(r["edge_types"])) <= args.max_hop]
    n_replay = int(len(labelled) * args.replay_ratio)
    fresh = rng.sample(fresh, min(n_replay, len(fresh)))
    r_items = [ex(r["clean_story"], node_names(r)[0], node_names(r)[len(ast.literal_eval(r["edge_types"]))],
                  "", f"replay|{r['id']}", len(ast.literal_eval(r["edge_types"]))) for r in fresh]
    print(f"== pseudo-replay: {len(r_items)} fresh k<={args.max_hop} chains ==", flush=True)
    r_pred = gen(model, tok, r_items, args.batch_size)

    replay_rows, agree = [], Counter()
    for r, item, p in zip(fresh, r_items, r_pred):
        lab = parse_relation(p)
        if lab is None:
            continue
        agree["n"] += 1
        agree["matches_gold"] += lab == r["target_text"]
        replay_rows.append(replace(item, target=lab))      # same prompt, seed's own answer
    frac = agree["matches_gold"] / max(agree["n"], 1)
    report["replay"] = {"n": agree["n"], "target_matches_gold": frac}
    print(f"  replay targets match gold on {frac:.3f} -- these are predictions, not gold", flush=True)
    if frac > 0.995:
        raise SystemExit("replay targets are indistinguishable from gold; check the label path")

    train_rows = [item for _, item in labelled] + replay_rows
    rng.shuffle(train_rows)
    print(f"== train: {len(labelled)} composed + {len(replay_rows)} replay = {len(train_rows)} ==", flush=True)

    del model
    torch.cuda.empty_cache()
    model, tok = load_adapter_for_training(args.model, Path(args.adapter))
    torch.manual_seed(args.seed)
    tr = train_lora(model=model, tokenizer=tok, examples=train_rows, output_dir=args.out_dir,
                    max_length=1024, max_steps=args.max_steps, learning_rate=args.learning_rate,
                    micro_batch_size=args.micro_batch_size, effective_batch_size=16, seed=args.seed)
    report["training"] = {k: v for k, v in tr.items() if k != "log_history"}

    print("== after: transfer to the REAL test split ==", flush=True)
    rng = random.Random(args.seed)          # same eval sample as `before`
    report["test_after"] = eval_test(model, tok, args.test_csv, args.per_k, args.batch_size, rng, "after")

    b, a = report["test_before"]["bands"], report["test_after"]["bands"]
    report["delta"] = {k: a[k]["accuracy"] - b[k]["accuracy"] for k in b}
    (args.out_dir / "round1_pool.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"before": b, "after": a, "delta": report["delta"]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

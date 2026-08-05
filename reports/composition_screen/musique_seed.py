#!/usr/bin/env python3
"""Train a one-hop MuSiQue seed, then re-run composed-vs-direct with no oracle.

The retrieval control left MuSiQue undecided.  Composition is load-bearing
(corrupting the upstream entity costs .417) and parts are genuinely easier than
composites (+.185/+.203/+.260), but with both arms searching the same 20
paragraphs the headroom was +.017 -- nothing.  That screen ran on the BASE model,
where single-hop accuracy is only .72-.78.  The method assumes an atomic seed is
trained first; on CLUTRR that step moved parts from .403 to .950 and was the
difference between a null screen and a real frontier gap.

So this is the experiment that decides the benchmark: does a one-hop seed lift
part accuracy enough that composing beats direct prediction?

Design choices that matter:

* The seed trains on single-hop sub-questions posed over the FULL 20-paragraph
  context of their parent instance -- the same condition the composed arm faces
  at inference.  Training on the gold supporting paragraph would teach answering
  but not retrieval, and the eval would then be off-distribution.
* Seed data comes from the TRAIN split, evaluation from dev.  Verified at run
  time: zero overlap in instance ids, multihop questions, and -- the one that
  matters, since MuSiQue composes its items from a shared single-hop pool --
  zero overlap in filled (sub-question, answer) pairs.
* Gold `#N` substitution is used for SEED TRAINING ONLY.  That is atomic gold
  supervision, exactly as CLUTRR trained on gold k<=4 and addition on a gold
  seed range.  Every evaluation arm is self-fed.
* Per-instance outcomes are logged, which the retrieval control failed to do,
  so the composed-vs-direct contrast gets McNemar and a paired bootstrap CI
  instead of an eyeballed difference.
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
from musique_isolation import fill, full_context, prompt, score_em

from self.coding.atomic_data import AtomicExample
from self.coding.training import (
    generate_predictions,
    load_qwen_lora_model,
    load_qwen_tokenizer,
    train_lora,
)


def ex(context: str, question: str, target: str, sid: str, k: int) -> AtomicExample:
    return AtomicExample(
        task="musique", source_id=sid, source_group_id=sid, split="x",
        messages=({"role": "user", "content": prompt(context, question)},),
        target=target, evaluator={}, metadata={"k": k},
    )


def usable(r: dict) -> bool:
    return all(s.get("paragraph_support_idx") is not None
               and s["paragraph_support_idx"] < len(r["paragraphs"])
               for s in r["question_decomposition"])


def gold_filled_steps(r: dict):
    """(question, answer) for each step, with gold upstream answers substituted."""
    gold = {str(i): s["answer"] for i, s in enumerate(r["question_decomposition"], 1)}
    for i, s in enumerate(r["question_decomposition"], 1):
        q = fill(s["question"], {j: a for j, a in gold.items() if int(j) < i})
        if "#" not in q:
            yield q, s["answer"]


def mcnemar(pairs) -> dict:
    """Exact-ish McNemar on paired binary outcomes [(composed_ok, direct_ok), ...]."""
    b = sum(1 for c, d in pairs if c and not d)      # composed wins
    c = sum(1 for x, d in pairs if not x and d)      # direct wins
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    # two-sided binomial test against p=0.5, computed directly
    from math import comb
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / (2 ** n)
    return {"b": b, "c": c, "p_value": min(1.0, 2 * tail)}


def boot_ci(pairs, rng, iters=5000) -> tuple:
    """Paired bootstrap CI on composed - direct."""
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    diffs = []
    for _ in range(iters):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(c for c, _ in s) / n - sum(d for _, d in s) / n)
    diffs.sort()
    return (diffs[int(0.025 * iters)], diffs[int(0.975 * iters)])


def evaluate(model, tok, sample_by_k, bs, tag, records):
    """direct, gold-fed part accuracy, and the self-fed no-oracle chain."""
    out = {}
    for k, sample in sorted(sample_by_k.items()):
        n = len(sample)
        d_items = [ex(full_context(r), r["question"], r["answer"], r["id"], k) for r in sample]
        d_pred = generate_predictions(model=model, tokenizer=tok, examples=d_items,
                                      batch_size=bs, max_new_tokens=32)
        d_ok = [score_em(p, [r["answer"], *r.get("answer_aliases", [])])
                for p, r in zip(d_pred, sample)]

        p_items, p_gold = [], []
        for r in sample:
            for q, a in gold_filled_steps(r):
                p_items.append(ex(full_context(r), q, a, r["id"], k))
                p_gold.append([a])
        p_pred = generate_predictions(model=model, tokenizer=tok, examples=p_items,
                                      batch_size=bs, max_new_tokens=32)
        p_ok = sum(score_em(p, g) for p, g in zip(p_pred, p_gold))

        # self-fed chain, full context at every step: no retrieval oracle
        state = [dict() for _ in sample]
        last = [None] * n
        for i in range(1, k + 1):
            idx, items = [], []
            for j, r in enumerate(sample):
                step = r["question_decomposition"][i - 1]
                q = fill(step["question"], state[j])
                if "#" in q:
                    continue
                idx.append(j)
                items.append(ex(full_context(r), q, step["answer"], f"{r['id']}|{i}", k))
            if not items:
                continue
            preds = generate_predictions(model=model, tokenizer=tok, examples=items,
                                         batch_size=bs, max_new_tokens=32)
            for j, p in zip(idx, preds):
                a = p.split("\n")[0].strip()
                state[j][str(i)] = a
                last[j] = a
        c_ok = [l is not None and score_em(l, [r["answer"], *r.get("answer_aliases", [])])
                for l, r in zip(last, sample)]

        for r, do, co in zip(sample, d_ok, c_ok):
            records.append({"phase": tag, "k": k, "id": r["id"],
                            "direct_ok": bool(do), "composed_ok": bool(co)})

        out[k] = {"n": n, "direct": sum(d_ok) / n, "part_full_ctx": p_ok / len(p_gold),
                  "composed_no_oracle": sum(c_ok) / n,
                  "headroom": (sum(c_ok) - sum(d_ok)) / n}
        v = out[k]
        print(f"  [{tag}] k={k}: direct={v['direct']:.3f} part={v['part_full_ctx']:.3f} "
              f"composed={v['composed_no_oracle']:.3f} ({v['headroom']:+.3f})", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("--model", "--train-data", "--dev-data"):
        ap.add_argument(a, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--micro-batch-size", type=int, default=1)
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="required to fit seq-4096 training on a 48 GB card")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [json.loads(l) for l in open(args.train_data)]
    dev_rows = [json.loads(l) for l in open(args.dev_data)]

    # --- leakage gate: MuSiQue composes items from a shared single-hop pool ---
    def filled_pairs(rows):
        return {(q.strip().lower(), str(a).strip().lower())
                for r in rows for q, a in gold_filled_steps(r)}
    tr_pairs, dv_pairs = filled_pairs(train_rows), filled_pairs(dev_rows)
    leak = {
        "instance_ids": len({r["id"] for r in train_rows} & {r["id"] for r in dev_rows}),
        "multihop_questions": len({r["question"].strip().lower() for r in train_rows}
                                  & {r["question"].strip().lower() for r in dev_rows}),
        "filled_single_hop_pairs": len(tr_pairs & dv_pairs),
    }
    print("leakage check:", leak, flush=True)
    if any(leak.values()):
        raise SystemExit(f"train/dev leakage detected: {leak}")

    # --- dev sample: same seed and draw order as the earlier screens ---
    by_k = defaultdict(list)
    for r in dev_rows:
        if usable(r):
            by_k[len(r["question_decomposition"])].append(r)
    rng = random.Random(args.seed)
    sample_by_k = {}
    for k in sorted(by_k):
        pool = by_k[k]
        sample_by_k[k] = pool if len(pool) <= args.per_k else rng.sample(pool, args.per_k)

    # --- seed training data: one-hop questions over the FULL parent context ---
    seed_items = []
    for r in train_rows:
        if not usable(r):
            continue
        ctx = full_context(r)
        for q, a in gold_filled_steps(r):
            seed_items.append(ex(ctx, q, str(a), f"seed|{r['id']}", 1))
    rng.shuffle(seed_items)

    tok = load_qwen_tokenizer(args.model)

    # `train_lora` REJECTS over-length examples rather than truncating, so filter
    # here.  The tail is thin -- 1.27% of the pool exceeds 4096 tokens -- and the
    # pool is far larger than train_size, so this costs nothing but must be done
    # on true tokenized length, not a character estimate.
    kept, dropped = [], 0
    for item in seed_items:
        if len(kept) >= args.train_size:
            break
        n_tok = len(tok(item.messages[0]["content"] + str(item.target))["input_ids"])
        if n_tok + 32 > args.max_length:          # margin for chat-template wrapping
            dropped += 1
            continue
        kept.append(item)
    seed_items = kept
    print(f"seed training examples: {len(seed_items)} (dropped {dropped} over "
          f"{args.max_length} tokens)", flush=True)
    if len(seed_items) < args.train_size:
        raise SystemExit(f"only {len(seed_items)} examples fit; need {args.train_size}")
    model = load_qwen_lora_model(args.model)
    torch.manual_seed(args.seed)

    records: list[dict] = []
    report: dict = {"config": vars(args) | {"out_dir": str(args.out_dir)}, "leakage": leak}

    print("== BEFORE (base; should reproduce musique_retrieval_control.json) ==", flush=True)
    report["before"] = evaluate(model, tok, sample_by_k, args.batch_size, "before", records)

    print("== training one-hop seed ==", flush=True)
    # The before-eval generates with long prompts and leaves a large allocator
    # pool behind; training at 4096 tokens then OOMs on the logits tensor
    # (seq x vocab x fp32) even though the live model is only ~8 GB.
    torch.cuda.empty_cache()
    print(f"  cuda reserved after empty_cache: "
          f"{torch.cuda.memory_reserved() / 2**30:.1f} GiB", flush=True)
    tr = train_lora(model=model, tokenizer=tok, examples=seed_items, output_dir=args.out_dir,
                    max_length=args.max_length, max_steps=args.max_steps,
                    learning_rate=args.learning_rate, micro_batch_size=args.micro_batch_size,
                    effective_batch_size=16, seed=args.seed,
                    gradient_checkpointing=args.gradient_checkpointing)
    report["training"] = {k: v for k, v in tr.items() if k != "log_history"}

    print("== AFTER ==", flush=True)
    report["after"] = evaluate(model, tok, sample_by_k, args.batch_size, "after", records)

    # --- paired statistics on the contrast that decides the benchmark ---
    stats = {}
    for phase in ("before", "after"):
        rs = [r for r in records if r["phase"] == phase]
        pairs = [(r["composed_ok"], r["direct_ok"]) for r in rs]
        srng = random.Random(args.seed)
        lo, hi = boot_ci(pairs, srng)
        stats[phase] = {
            "n": len(pairs),
            "composed": sum(c for c, _ in pairs) / len(pairs),
            "direct": sum(d for _, d in pairs) / len(pairs),
            "headroom": (sum(c for c, _ in pairs) - sum(d for _, d in pairs)) / len(pairs),
            "ci95": [lo, hi], "mcnemar": mcnemar(pairs),
        }
        by_hop = {}
        for k in sorted({r["k"] for r in rs}):
            kp = [(r["composed_ok"], r["direct_ok"]) for r in rs if r["k"] == k]
            klo, khi = boot_ci(kp, random.Random(args.seed))
            by_hop[k] = {"n": len(kp),
                         "headroom": (sum(c for c, _ in kp) - sum(d for _, d in kp)) / len(kp),
                         "ci95": [klo, khi], "mcnemar": mcnemar(kp)}
        stats[phase]["by_hop"] = by_hop
    report["stats"] = stats

    (args.out_dir / "per_instance.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n")
    (args.out_dir / "musique_seed.json").write_text(json.dumps(report, indent=2, default=str))

    for phase in ("before", "after"):
        s = stats[phase]
        print(f"\n{phase.upper()}: composed={s['composed']:.3f} direct={s['direct']:.3f} "
              f"headroom={s['headroom']:+.3f} 95%CI[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}] "
              f"McNemar p={s['mcnemar']['p_value']:.4f}", flush=True)
    d = stats["after"]["headroom"] - stats["before"]["headroom"]
    print(f"\nSEED EFFECT ON HEADROOM: {d:+.3f}", flush=True)
    print("VERDICT:", "composition wins after seeding"
          if stats["after"]["headroom"] > 0 and stats["after"]["ci95"][0] > 0
          else "no significant headroom after seeding", flush=True)


if __name__ == "__main__":
    main()

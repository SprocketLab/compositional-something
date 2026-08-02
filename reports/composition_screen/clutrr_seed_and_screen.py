#!/usr/bin/env python3
"""Train a CLUTRR seed on k=2-4, then screen k=5-10.

The first screen was uninformative because the base model scored .447 at k=2 --
there was no reliable seed regime, so the k-curve measured noise around a weak
baseline rather than compositional degradation.  CLUTRR ships 12,065 train rows
at k=2-4 precisely so a seed can be trained, which mirrors the atomic
calibration step used for BFCL.

Reports, for k=2..10: accuracy, and the implied per-step retention
acc(k)^(1/(k-1)).  Flat retention means errors compound independently and
decomposition buys nothing (the BFCL outcome).  Retention that FALLS with k is
the property the method needs.  Predictions are saved so degenerate
majority-class guessing can be ruled out -- an omission in the first screen.
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

import torch

from self.coding.atomic_data import AtomicExample
from self.coding.training import (
    generate_predictions,
    load_qwen_lora_model,
    load_qwen_tokenizer,
    train_lora,
)

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


def score(pred: str, gold: str) -> bool:
    p = normalize(pred)
    hits = [r for r in RELATIONS if re.search(rf"\b{re.escape(normalize(r))}\b", p)]
    return bool(hits) and normalize(max(hits, key=len)) == normalize(gold)


def to_example(row: dict, split: str) -> AtomicExample:
    a, b = ast.literal_eval(row["query"])
    return AtomicExample(
        task="clutrr",
        source_id=str(row["id"]),
        source_group_id=str(row["id"]),
        split=split,
        messages=({"role": "user", "content": prompt_for(row["clean_story"], a, b)},),
        target=str(row["target_text"]),
        evaluator={"gold": str(row["target_text"])},
        component_count=len(ast.literal_eval(row["edge_types"])),
        metadata={"k": len(ast.literal_eval(row["edge_types"]))},
    )


def load(path: Path, split: str) -> list[AtomicExample]:
    return [to_example(r, split) for r in csv.DictReader(open(path))]


def evaluate(model, tokenizer, examples, batch_size, tag) -> tuple[float, list[dict]]:
    preds = generate_predictions(
        model=model, tokenizer=tokenizer, examples=examples,
        batch_size=batch_size, max_new_tokens=12,
    )
    rows = [
        {"source_id": e.source_id, "k": e.metadata["k"], "gold": e.target,
         "prediction": p, "correct": score(p, e.target)}
        for e, p in zip(examples, preds)
    ]
    acc = sum(r["correct"] for r in rows) / max(len(rows), 1)
    print(f"  [{tag}] n={len(rows)} acc={acc:.3f}", flush=True)
    return acc, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--train-size", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train_all = load(Path(args.train_csv), "train")
    rng.shuffle(train_all)
    holdout = train_all[: args.per_k * 3]          # k=2-4 validation, seed-regime check
    train = train_all[args.per_k * 3 :][: args.train_size]
    print(f"train={len(train)} holdout={len(holdout)}", flush=True)

    test_by_k = defaultdict(list)
    for e in load(Path(args.test_csv), "test"):
        test_by_k[e.metadata["k"]].append(e)
    test = []
    for k in sorted(test_by_k):
        pool = test_by_k[k]
        test += pool if len(pool) <= args.per_k else rng.sample(pool, args.per_k)

    tokenizer = load_qwen_tokenizer(args.model)
    model = load_qwen_lora_model(args.model)
    torch.manual_seed(args.seed)

    report: dict = {"config": vars(args) | {"out_dir": str(args.out_dir)}}
    print("== before training ==", flush=True)
    report["before"] = {}
    acc, rows = evaluate(model, tokenizer, holdout, args.eval_batch_size, "holdout k2-4 BEFORE")
    report["before"]["holdout_k2to4"] = acc

    training = train_lora(
        model=model, tokenizer=tokenizer, examples=train, output_dir=args.out_dir,
        max_length=1024, max_steps=args.max_steps, learning_rate=args.learning_rate,
        micro_batch_size=args.micro_batch_size, effective_batch_size=16, seed=args.seed,
    )
    report["training"] = {k: v for k, v in training.items() if k != "log_history"}

    print("== after training ==", flush=True)
    acc, hold_rows = evaluate(model, tokenizer, holdout, args.eval_batch_size, "holdout k2-4 AFTER")
    report["after"] = {"holdout_k2to4": acc, "by_k": {}}

    all_rows = list(hold_rows)
    by_k = defaultdict(list)
    for e in test:
        by_k[e.metadata["k"]].append(e)
    for k in sorted(by_k):
        acc, rows = evaluate(model, tokenizer, by_k[k], args.eval_batch_size, f"test k={k}")
        retention = acc ** (1 / (k - 1)) if k > 1 and acc > 0 else None
        dist = Counter(normalize(r["prediction"])[:24] for r in rows).most_common(3)
        report["after"]["by_k"][k] = {
            "n": len(rows), "accuracy": acc, "per_step_retention": retention,
            "top_predictions": dist,
        }
        all_rows += rows

    (args.out_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in all_rows) + "\n"
    )
    (args.out_dir / "clutrr_screen.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["after"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Screening diagnostic for compositional self-improvement task candidates.

The question a candidate must answer: does per-part accuracy DEGRADE as the
composite grows?  If parts are solved just as reliably inside a k-part problem
as alone, then composing k predictions reproduces p^k and decomposition buys
nothing -- the BFCL outcome, where per-call accuracy was flat (.897/.863/.899
at k=2/4/8) and observed accuracy tracked p^k to within a few percent.

CLUTRR   : parts cannot be isolated from the story, so we report acc(k) for
           k=2..10 and the implied per-step retention acc(k)^(1/(k-1)).
MuSiQue  : question_decomposition supplies each single-hop atom and its gold
           answer, so per-part accuracy is measured directly by asking the
           model the sub-questions with earlier answers substituted in.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CLUTRR_RELATIONS = [
    "aunt", "brother", "daughter", "daughter-in-law", "father", "father-in-law",
    "granddaughter", "grandfather", "grandmother", "grandson", "mother",
    "mother-in-law", "nephew", "niece", "sister", "son", "son-in-law", "uncle",
]


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


class Runner:
    def __init__(self, model_name: str, batch_size: int = 16):
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16
        ).to("cuda").eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def generate(self, prompts: list[str], max_new_tokens: int = 32) -> list[str]:
        out = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            texts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
                for p in batch
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            width = enc["input_ids"].shape[1]
            gen = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id,
            )
            out += [self.tok.decode(s[width:], skip_special_tokens=True).strip() for s in gen]
            print(f"    ...{min(i+self.batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return out


# ----------------------------------------------------------------- CLUTRR ---
def clutrr_prompt(story: str, a: str, b: str) -> str:
    return (
        "Read the story and state the family relation.\n\n"
        f"Story: {story}\n\n"
        f"Question: How is {b} related to {a}? In other words, {b} is {a}'s what?\n"
        f"Answer with exactly one word from this list: {', '.join(CLUTRR_RELATIONS)}.\n"
        "Answer:"
    )


def score_clutrr(pred: str, gold: str) -> bool:
    # normalize() strips hyphens, so compare relations in normalized form too --
    # otherwise "daughter-in-law" becomes "daughterinlaw" and never matches.
    p = normalize(pred)
    hits = [r for r in CLUTRR_RELATIONS if re.search(rf"\b{re.escape(normalize(r))}\b", p)]
    if not hits:
        return False
    # prefer the longest match so "daughter-in-law" is not read as "daughter"
    return normalize(max(hits, key=len)) == normalize(gold)


def run_clutrr(runner: Runner, path: Path, per_k: int, seed: int) -> dict:
    rows = list(csv.DictReader(open(path)))
    by_k = defaultdict(list)
    for r in rows:
        by_k[len(ast.literal_eval(r["edge_types"]))].append(r)
    rng = random.Random(seed)
    out = {}
    for k in sorted(by_k):
        sample = by_k[k]
        if len(sample) > per_k:
            sample = rng.sample(sample, per_k)
        prompts, golds = [], []
        for r in sample:
            a, b = ast.literal_eval(r["query"])
            prompts.append(clutrr_prompt(r["clean_story"], a, b))
            golds.append(r["target_text"])
        print(f"  CLUTRR k={k}: {len(prompts)} instances", flush=True)
        preds = runner.generate(prompts, max_new_tokens=16)
        acc = sum(score_clutrr(p, g) for p, g in zip(preds, golds)) / len(golds)
        out[k] = {"n": len(golds), "accuracy": acc}
        print(f"  CLUTRR k={k}: acc={acc:.3f}", flush=True)
    return out


# ---------------------------------------------------------------- MuSiQue ---
def musique_context(row: dict, limit: int = 20) -> str:
    keep = [p for p in row["paragraphs"] if p.get("is_supporting")]
    keep += [p for p in row["paragraphs"] if not p.get("is_supporting")][: max(0, limit - len(keep))]
    return "\n\n".join(f"[{i+1}] {p['title']}: {p['paragraph_text']}" for i, p in enumerate(keep))


def musique_prompt(context: str, question: str) -> str:
    return (
        "Answer the question using the passages. Reply with the answer only -- "
        "a name or short phrase, no sentence.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )


def score_em(pred: str, golds: list[str]) -> bool:
    p = normalize(pred.split("\n")[0])
    return any(normalize(g) and normalize(g) in p for g in golds)


def run_musique(runner: Runner, path: Path, per_k: int, seed: int) -> dict:
    rows = [json.loads(l) for l in open(path)]
    by_k = defaultdict(list)
    for r in rows:
        by_k[len(r["question_decomposition"])].append(r)
    rng = random.Random(seed)
    out = {}
    for k in sorted(by_k):
        sample = by_k[k]
        if len(sample) > per_k:
            sample = rng.sample(sample, per_k)

        # composite: the full k-hop question
        comp_prompts = [musique_prompt(musique_context(r), r["question"]) for r in sample]
        comp_golds = [[r["answer"], *r.get("answer_aliases", [])] for r in sample]
        print(f"  MuSiQue k={k}: {len(comp_prompts)} composites", flush=True)
        comp_pred = runner.generate(comp_prompts, max_new_tokens=32)
        q = sum(score_em(p, g) for p, g in zip(comp_pred, comp_golds)) / len(comp_golds)

        # parts: each single-hop sub-question, with earlier gold answers substituted
        part_prompts, part_golds = [], []
        for r in sample:
            answers = {}
            for idx, step in enumerate(r["question_decomposition"], start=1):
                text = step["question"]
                for ref, ans in answers.items():
                    text = text.replace(f"#{ref}", ans)
                if "#" in text:            # unresolved reference; skip this part
                    answers[idx] = step["answer"]
                    continue
                part_prompts.append(musique_prompt(musique_context(r), text))
                part_golds.append([step["answer"]])
                answers[idx] = step["answer"]
        print(f"  MuSiQue k={k}: {len(part_prompts)} isolated parts", flush=True)
        part_pred = runner.generate(part_prompts, max_new_tokens=32)
        p = sum(score_em(x, g) for x, g in zip(part_pred, part_golds)) / len(part_golds)

        out[k] = {
            "n_composite": len(comp_golds), "n_parts": len(part_golds),
            "composite_accuracy": q, "per_part_accuracy": p,
            "independent_prediction": p ** k, "ratio_observed_over_independent": q / (p ** k) if p else None,
        }
        print(f"  MuSiQue k={k}: composite={q:.3f}  per-part={p:.3f}  p^{k}={p**k:.3f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--clutrr")
    ap.add_argument("--musique")
    ap.add_argument("--per-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runner = Runner(args.model, args.batch_size)
    report = {"model": args.model, "per_k": args.per_k, "seed": args.seed}
    if args.clutrr:
        report["clutrr"] = run_clutrr(runner, Path(args.clutrr), args.per_k, args.seed)
    if args.musique:
        report["musique"] = run_musique(runner, Path(args.musique), args.per_k, args.seed)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

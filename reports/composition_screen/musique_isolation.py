#!/usr/bin/env python3
"""MuSiQue isolation-gain screen: is a sub-question EASIER on its own paragraph?

This is the measurement that decided CLUTRR, applied before committing to a new
benchmark.  Compositional self-improvement needs
    p(part solved alone)  >  p(part solved inside the composite)
On CLUTRR that gap was zero (.446 vs .445) because the story's entities
interleave and a sub-chain cannot be given its own text.  MuSiQue's release
carries `paragraph_support_idx` for every step, so isolation is constructible
for 100% of steps -- the question is whether it buys anything.

The previous screen (`screen_composition.py:run_musique`) could not answer this.
It asked every sub-question over the FULL 20-paragraph context and substituted
GOLD upstream answers into `#N` references, so its "per_part_accuracy" was
p(part in situ, oracle-fed) -- optimistic on one axis and in-situ on the other.

Five arms, all paired on the same instances:

  direct              full multi-hop question, full context
  part_situ_gold      sub-question, full context,  gold #N        [old screen]
  part_iso_gold       sub-question, OWN paragraph, gold #N        [isolation]
  part_iso_self       sub-question, OWN paragraph, model's own #N [realistic]
  composed_self       last step of part_iso_self = the pseudo-label

isolation gain  = part_iso_gold - part_situ_gold
error cascade   = part_iso_gold - part_iso_self
usable headroom = composed_self - direct
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def score_em(pred: str, golds) -> bool:
    """Substring containment, matching the previous screen so numbers compare."""
    p = normalize(pred.split("\n")[0])
    return any(normalize(g) and normalize(g) in p for g in golds)


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
                self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
                for p in batch
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            width = enc["input_ids"].shape[1]
            gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id,
                                      eos_token_id=self.tok.eos_token_id)
            out += [self.tok.decode(s[width:], skip_special_tokens=True).strip() for s in gen]
        print(f"      generated {len(prompts)}", flush=True)
        return out


def full_context(row: dict, limit: int = 20) -> str:
    keep = [p for p in row["paragraphs"] if p.get("is_supporting")]
    keep += [p for p in row["paragraphs"] if not p.get("is_supporting")][: max(0, limit - len(keep))]
    return "\n\n".join(f"[{i+1}] {p['title']}: {p['paragraph_text']}" for i, p in enumerate(keep))


def own_paragraph(row: dict, step: dict) -> str:
    p = row["paragraphs"][step["paragraph_support_idx"]]
    return f"[1] {p['title']}: {p['paragraph_text']}"


def prompt(context: str, question: str) -> str:
    return ("Answer the question using the passages. Reply with the answer only -- "
            "a name or short phrase, no sentence.\n\n"
            f"{context}\n\nQuestion: {question}\nAnswer:")


def fill(text: str, answers: dict) -> str:
    """Substitute #N references, longest index first so #10 beats #1."""
    for ref in sorted(answers, key=lambda x: -int(x)):
        text = text.replace(f"#{ref}", answers[ref])
    return text


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
        print(f"== {k} hops: {len(sample)} instances ==", flush=True)

        # --- direct ---
        d_pred = runner.generate([prompt(full_context(r), r["question"]) for r in sample])
        d_ok = sum(score_em(p, [r["answer"], *r.get("answer_aliases", [])])
                   for p, r in zip(d_pred, sample))

        # --- gold-fed arms: every step independent, one batch ---
        situ, iso, golds = [], [], []
        for r in sample:
            gold_ans = {str(i): s["answer"] for i, s in enumerate(r["question_decomposition"], 1)}
            for i, s in enumerate(r["question_decomposition"], 1):
                q = fill(s["question"], {j: a for j, a in gold_ans.items() if int(j) < i})
                if "#" in q:
                    continue                       # forward reference; unusable either way
                situ.append(prompt(full_context(r), q))
                iso.append(prompt(own_paragraph(r, s), q))
                golds.append([s["answer"]])
        print(f"  gold-fed parts: {len(golds)}", flush=True)
        s_pred = runner.generate(situ)
        i_pred = runner.generate(iso)
        situ_ok = sum(score_em(p, g) for p, g in zip(s_pred, golds))
        iso_ok = sum(score_em(p, g) for p, g in zip(i_pred, golds))

        # --- self-fed arm: sequential, step i needs the model's answer to i-1 ---
        state = [dict() for _ in sample]           # per instance: {step_index: model answer}
        last = [None] * len(sample)
        self_ok = self_n = 0
        for i in range(1, k + 1):
            idx, prompts_i, golds_i = [], [], []
            for j, r in enumerate(sample):
                step = r["question_decomposition"][i - 1]
                q = fill(step["question"], state[j])
                if "#" in q:
                    continue
                idx.append(j)
                prompts_i.append(prompt(own_paragraph(r, step), q))
                golds_i.append([step["answer"]])
            if not prompts_i:
                continue
            preds = runner.generate(prompts_i)
            for j, p, g in zip(idx, preds, golds_i):
                ans = p.split("\n")[0].strip()
                state[j][str(i)] = ans
                last[j] = ans
                self_n += 1
                self_ok += score_em(p, g)

        comp_ok = sum(l is not None and score_em(l, [r["answer"], *r.get("answer_aliases", [])])
                      for l, r in zip(last, sample))

        n = len(sample)
        v = {
            "n": n, "n_parts": len(golds),
            "direct": d_ok / n,
            "part_situ_gold": situ_ok / len(golds),
            "part_iso_gold": iso_ok / len(golds),
            "part_iso_self": self_ok / max(self_n, 1),
            "composed_self": comp_ok / n,
        }
        v["isolation_gain"] = v["part_iso_gold"] - v["part_situ_gold"]
        v["error_cascade"] = v["part_iso_gold"] - v["part_iso_self"]
        v["usable_headroom"] = v["composed_self"] - v["direct"]
        report["by_hops"][k] = v
        print(f"  direct={v['direct']:.3f} | part situ={v['part_situ_gold']:.3f} "
              f"iso={v['part_iso_gold']:.3f} (gain {v['isolation_gain']:+.3f}) "
              f"iso-self={v['part_iso_self']:.3f} | composed={v['composed_self']:.3f} "
              f"({v['usable_headroom']:+.3f})", flush=True)

    tot = sum(v["n"] for v in report["by_hops"].values())
    tp = sum(v["n_parts"] for v in report["by_hops"].values())
    report["overall"] = {
        m: sum(v[m] * (v["n_parts"] if "part" in m else v["n"]) for v in report["by_hops"].values())
           / (tp if "part" in m else tot)
        for m in ("direct", "part_situ_gold", "part_iso_gold", "part_iso_self", "composed_self")
    }
    o = report["overall"]
    o["isolation_gain"] = o["part_iso_gold"] - o["part_situ_gold"]
    o["usable_headroom"] = o["composed_self"] - o["direct"]
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(o, indent=2), flush=True)
    print(f"\nISOLATION GAIN {o['isolation_gain']:+.3f}   "
          f"USABLE HEADROOM {o['usable_headroom']:+.3f}", flush=True)


if __name__ == "__main__":
    main()

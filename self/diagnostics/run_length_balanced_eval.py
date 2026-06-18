#!/usr/bin/env python3
"""Balanced-answer evaluation for plain-output run-length diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from self.core.data_io import ensure_dir, sanitize_json_value
from self.core.evaluation import build_generation_encodings, parse_prediction
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.recipes import RECIPE_ALGORITHMIC_SELF_IMPROVE_V1
from self.tasks.bit import RUN_LENGTH_ALPHABET_SYMBOLS
from self.tasks.run_length import RunLengthTask
from self.tasks.run_length import (
    RunLengthExample,
    generate_run_length_example,
)
from self.tasks.run_length import compute_run_stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run balanced-answer evaluation for run-length models.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-bits", type=int, required=True)
    parser.add_argument("--max-bits", type=int, required=True)
    parser.add_argument("--frontier-min-bits", type=int, default=None)
    parser.add_argument("--symbol-alphabet-size", type=int, default=2)
    parser.add_argument("--per-answer", type=int, default=50)
    parser.add_argument("--natural-per-bit", type=int, default=100)
    parser.add_argument("--min-supported-count", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--debug-sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--recipe",
        choices=("none", RECIPE_ALGORITHMIC_SELF_IMPROVE_V1),
        default=RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
    )
    parser.add_argument("--format-version", choices=("legacy", "symbolic_v1"), default="legacy")
    parser.add_argument("--target-mode", choices=("default", "plain_output"), default="plain_output")
    return parser


def _no_repeat_string(
    length: int,
    alphabet: str,
    rng: random.Random,
    *,
    disallowed_first: Optional[str] = None,
    disallowed_last: Optional[str] = None,
) -> str:
    if length <= 0:
        return ""
    result: List[str] = []
    for idx in range(length):
        choices = list(alphabet)
        if idx == 0 and disallowed_first is not None:
            choices = [ch for ch in choices if ch != disallowed_first]
        if idx > 0:
            choices = [ch for ch in choices if ch != result[-1]]
        if idx == length - 1 and disallowed_last is not None:
            choices = [ch for ch in choices if ch != disallowed_last]
        if not choices:
            raise ValueError("Unable to construct nonrepeating filler with the requested boundary constraints.")
        result.append(rng.choice(choices))
    return "".join(result)


def construct_run_length_string(bits: int, target_max_run: int, alphabet: str, rng: random.Random) -> str:
    """Construct a string whose longest run is exactly ``target_max_run``."""

    if target_max_run < 1 or target_max_run > bits:
        raise ValueError(f"target_max_run must be in [1, {bits}], got {target_max_run}.")
    if target_max_run == 1:
        return _no_repeat_string(bits, alphabet, rng)
    symbol = rng.choice(alphabet)
    if target_max_run == bits:
        return symbol * bits

    start = rng.randint(0, bits - target_max_run)
    left_len = start
    right_len = bits - target_max_run - start
    left = _no_repeat_string(left_len, alphabet, rng, disallowed_last=symbol)
    right = _no_repeat_string(right_len, alphabet, rng, disallowed_first=symbol)
    return left + (symbol * target_max_run) + right


def make_run_length_example(bitstring: str, *, format_version: str, target_mode: str) -> RunLengthExample:
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=len(bitstring),
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=format_version,
        target_mode=target_mode,
    )


def generate_balanced_examples(
    *,
    min_bits: int,
    max_bits: int,
    alphabet: str,
    per_answer: int,
    rng: random.Random,
    format_version: str,
    target_mode: str,
    max_attempts_per_cell: int = 10_000,
) -> Tuple[List[RunLengthExample], Dict[str, Dict[str, int]], List[Dict[str, int]]]:
    examples: List[RunLengthExample] = []
    counts: Dict[str, Dict[str, int]] = {}
    underfilled: List[Dict[str, int]] = []

    for bits in range(min_bits, max_bits + 1):
        counts[str(bits)] = {}
        for answer in range(1, bits + 1):
            seen: set[str] = set()
            attempts = 0
            while len(seen) < per_answer and attempts < max_attempts_per_cell:
                attempts += 1
                bitstring = construct_run_length_string(bits, answer, alphabet, rng)
                if bitstring in seen:
                    continue
                example = make_run_length_example(
                    bitstring,
                    format_version=format_version,
                    target_mode=target_mode,
                )
                if example.max_run != answer:
                    raise AssertionError(
                        f"Constructed invalid run-length example: bits={bits} target={answer} got={example.max_run}"
                    )
                seen.add(bitstring)
                examples.append(example)
            counts[str(bits)][str(answer)] = len(seen)
            if len(seen) < per_answer:
                underfilled.append(
                    {
                        "bits": bits,
                        "answer": answer,
                        "requested": per_answer,
                        "retained": len(seen),
                        "attempts": attempts,
                    }
                )
    rng.shuffle(examples)
    return examples, counts, underfilled


def generate_natural_examples(
    *,
    min_bits: int,
    max_bits: int,
    alphabet: str,
    per_bit: int,
    rng: random.Random,
    format_version: str,
    target_mode: str,
) -> List[RunLengthExample]:
    examples: List[RunLengthExample] = []
    for bits in range(min_bits, max_bits + 1):
        for _ in range(per_bit):
            examples.append(
                generate_run_length_example(
                    bits,
                    rng,
                    format_version=format_version,
                    target_mode=target_mode,
                    alphabet=alphabet,
                )
            )
    rng.shuffle(examples)
    return examples


def evaluate_with_predictions(
    *,
    model: Any,
    tokenizer: Any,
    task: RunLengthTask,
    examples: Sequence[RunLengthExample],
    batch_size: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    if not examples:
        return []
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = list(examples[start : start + batch_size])
            prompts = [example.prompt() for example in batch]
            encodings = build_generation_encodings(tokenizer, prompts, device)
            output_ids = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt_width = encodings["input_ids"].shape[1]
            for idx, example in enumerate(batch):
                generated_slice = output_ids[idx, prompt_width:].tolist()
                decoded = tokenizer.decode(generated_slice, skip_special_tokens=True)
                prediction = parse_prediction(task.prediction_parser, decoded, example)
                target = example.target()
                rows.append(
                    {
                        "bits": example.bits,
                        "answer": int(target),
                        "target": target,
                        "prediction": prediction,
                        "correct": prediction == target,
                        "prompt": example.prompt(),
                        "decoded_output": decoded,
                        "bitstring": example.bitstring,
                    }
                )
    if model_was_training:
        model.train()
    return rows


def summarize_prediction_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_supported_count: int,
    frontier_min_bits: Optional[int] = None,
) -> Dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "micro_accuracy": math.nan,
            "macro_answer_accuracy": math.nan,
            "macro_supported_accuracy": math.nan,
            "per_bit_macro_accuracy": {},
            "per_answer_accuracy": {},
            "answer_counts": {},
            "cell_metrics": {},
        }

    def summarize_subset(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {
                "count": 0,
                "micro_accuracy": math.nan,
                "macro_answer_accuracy": math.nan,
                "macro_supported_accuracy": math.nan,
                "per_bit_macro_accuracy": {},
                "per_answer_accuracy": {},
                "answer_counts": {},
                "cell_metrics": {},
            }
        correct_total = sum(1 for row in subset if row["correct"])
        cell_totals: Dict[Tuple[int, int], int] = defaultdict(int)
        cell_correct: Dict[Tuple[int, int], int] = defaultdict(int)
        answer_totals: Dict[int, int] = defaultdict(int)
        answer_correct: Dict[int, int] = defaultdict(int)
        for row in subset:
            key = (int(row["bits"]), int(row["answer"]))
            cell_totals[key] += 1
            answer_totals[key[1]] += 1
            if row["correct"]:
                cell_correct[key] += 1
                answer_correct[key[1]] += 1

        cell_metrics: Dict[str, Dict[str, Any]] = {}
        bit_to_cell_accs: Dict[int, List[float]] = defaultdict(list)
        all_cell_accs: List[float] = []
        supported_cell_accs: List[float] = []
        for key in sorted(cell_totals):
            bits, answer = key
            count = cell_totals[key]
            accuracy = cell_correct[key] / count if count else math.nan
            cell_metrics[f"{bits}:{answer}"] = {
                "bits": bits,
                "answer": answer,
                "count": count,
                "accuracy": accuracy,
            }
            if not math.isnan(accuracy):
                all_cell_accs.append(accuracy)
                bit_to_cell_accs[bits].append(accuracy)
                if count >= min_supported_count:
                    supported_cell_accs.append(accuracy)

        return {
            "count": len(subset),
            "micro_accuracy": correct_total / len(subset),
            "macro_answer_accuracy": sum(all_cell_accs) / len(all_cell_accs) if all_cell_accs else math.nan,
            "macro_supported_accuracy": (
                sum(supported_cell_accs) / len(supported_cell_accs) if supported_cell_accs else math.nan
            ),
            "supported_cell_count": len(supported_cell_accs),
            "cell_count": len(all_cell_accs),
            "per_bit_macro_accuracy": {
                str(bits): sum(values) / len(values) for bits, values in sorted(bit_to_cell_accs.items())
            },
            "per_answer_accuracy": {
                str(answer): answer_correct[answer] / count for answer, count in sorted(answer_totals.items())
            },
            "answer_counts": {str(answer): count for answer, count in sorted(answer_totals.items())},
            "cell_metrics": cell_metrics,
        }

    payload = summarize_subset(rows)
    if frontier_min_bits is not None:
        payload["frontier"] = summarize_subset([row for row in rows if int(row["bits"]) >= frontier_min_bits])
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True

    if args.symbol_alphabet_size < 2 or args.symbol_alphabet_size > len(RUN_LENGTH_ALPHABET_SYMBOLS):
        raise ValueError("Unsupported symbol alphabet size.")
    if args.min_bits > args.max_bits:
        raise ValueError("--min-bits must be <= --max-bits.")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    rng = random.Random(args.seed)
    alphabet = RUN_LENGTH_ALPHABET_SYMBOLS[: args.symbol_alphabet_size]

    task = RunLengthTask()
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode="auto",
        recipe=args.recipe,
    )
    if getattr(model, "generation_config", None) is not None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    balanced_examples, balanced_counts, underfilled_cells = generate_balanced_examples(
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        alphabet=alphabet,
        per_answer=args.per_answer,
        rng=rng,
        format_version=args.format_version,
        target_mode=args.target_mode,
    )
    natural_examples = generate_natural_examples(
        min_bits=args.min_bits,
        max_bits=args.max_bits,
        alphabet=alphabet,
        per_bit=args.natural_per_bit,
        rng=rng,
        format_version=args.format_version,
        target_mode=args.target_mode,
    )

    balanced_rows = evaluate_with_predictions(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=balanced_examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    natural_rows = evaluate_with_predictions(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=natural_examples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    payload = {
        "model_name": args.model_name,
        "min_bits": args.min_bits,
        "max_bits": args.max_bits,
        "frontier_min_bits": args.frontier_min_bits,
        "symbol_alphabet_size": args.symbol_alphabet_size,
        "alphabet": alphabet,
        "per_answer": args.per_answer,
        "natural_per_bit": args.natural_per_bit,
        "min_supported_count": args.min_supported_count,
        "balanced": summarize_prediction_rows(
            balanced_rows,
            min_supported_count=args.min_supported_count,
            frontier_min_bits=args.frontier_min_bits,
        ),
        "natural": summarize_prediction_rows(
            natural_rows,
            min_supported_count=args.min_supported_count,
            frontier_min_bits=args.frontier_min_bits,
        ),
        "answer_counts": balanced_counts,
        "underfilled_cells": underfilled_cells,
    }
    with (output_dir / "balanced_eval_results.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)

    debug_rows = balanced_rows[: args.debug_sample_size]
    with (output_dir / "balanced_eval_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in debug_rows:
            json.dump(sanitize_json_value(row), handle)
            handle.write("\n")

    summary = {
        "balanced_micro": payload["balanced"]["micro_accuracy"],
        "balanced_macro_answer": payload["balanced"]["macro_answer_accuracy"],
        "balanced_macro_supported": payload["balanced"]["macro_supported_accuracy"],
        "natural_micro": payload["natural"]["micro_accuracy"],
    }
    print(json.dumps(sanitize_json_value(summary)), flush=True)
    print(f"[INFO] Saved balanced eval to {output_dir / 'balanced_eval_results.json'}", flush=True)


if __name__ == "__main__":
    main()

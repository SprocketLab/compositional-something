#!/usr/bin/env python3
"""Seed-fit experiment for rectangular multiplication partitions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import Trainer, set_seed

from self.core.data_io import ensure_dir, sanitize_json_value
from self.core.evaluation import build_generation_encodings, parse_prediction, resolve_max_new_tokens
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import (
    CausalLMDataCollator,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    make_training_args,
)
from self.tasks.bit_parsing import parse_multiplication_prediction


PartitionKey = Tuple[int, int]


@dataclass(frozen=True, slots=True)
class RectangularMultiplicationExample:
    a: int
    b: int
    a_digits: int
    b_digits: int
    format_version: str = "legacy"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"{self.a:0{self.a_digits}d}×{self.b:0{self.b_digits}d}="
        return f"Q: {self.a:0{self.a_digits}d} * {self.b:0{self.b_digits}d} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        value = self.a * self.b
        if self.format_version == "symbolic_v1":
            return f"{value:0{self.a_digits + self.b_digits}d}"
        return str(value)

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def parse_rectangular_multiplication_prediction(
    text: str,
    example: Optional[RectangularMultiplicationExample] = None,
) -> Optional[str]:
    value = parse_multiplication_prediction(text)
    if value is None:
        return None
    if example is None or example.format_version != "symbolic_v1":
        return value
    return f"{int(value):0{example.a_digits + example.b_digits}d}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rectangular multiplication seed-fit experiment.")
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--format-version", type=str, default="legacy", choices=("legacy", "symbolic_v1"))
    parser.add_argument("--min-a-digits", type=int, default=1)
    parser.add_argument("--max-a-digits", type=int, default=3)
    parser.add_argument("--min-b-digits", type=int, default=1)
    parser.add_argument("--max-b-digits", type=int, default=3)
    parser.add_argument(
        "--eval-per-partition",
        type=int,
        default=100,
        help="Validation/test holdout per partition when enough unique examples exist.",
    )
    parser.add_argument(
        "--small-partition-eval-count",
        type=int,
        default=10,
        help="Fallback validation/test holdout for tiny partitions such as 1x1.",
    )

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=250)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional max training steps; 0 means run the full requested epochs.",
    )
    parser.add_argument("--decode-max-new-tokens", type=int, default=16)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--init-from-scratch", action="store_true")
    parser.add_argument("--tokenizer-mode", type=str, choices=("auto", "fixed_char"), default="fixed_char")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def values_for_digits(num_digits: int) -> range:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    if num_digits == 1:
        return range(0, 10)
    start = 10 ** (num_digits - 1)
    stop = 10**num_digits
    return range(start, stop)


def partition_label(key: PartitionKey) -> str:
    return f"{key[0]}x{key[1]}"


def holdout_counts(total: int, requested_eval: int, small_eval: int) -> Tuple[int, int]:
    if total >= (2 * requested_eval) + 1:
        return requested_eval, requested_eval
    fallback = min(small_eval, max(1, (total - 1) // 3))
    return fallback, fallback


def build_exhaustive_splits(
    *,
    min_a_digits: int,
    max_a_digits: int,
    min_b_digits: int,
    max_b_digits: int,
    eval_per_partition: int,
    small_partition_eval_count: int,
    format_version: str,
    rng: random.Random,
) -> Tuple[Dict[str, List[RectangularMultiplicationExample]], Dict[str, Dict[str, int]]]:
    splits: Dict[str, List[RectangularMultiplicationExample]] = {"train": [], "validation": [], "test": []}
    counts: Dict[str, Dict[str, int]] = {"train": {}, "validation": {}, "test": {}}

    for a_digits in range(min_a_digits, max_a_digits + 1):
        a_values = list(values_for_digits(a_digits))
        for b_digits in range(min_b_digits, max_b_digits + 1):
            b_values = list(values_for_digits(b_digits))
            label = partition_label((a_digits, b_digits))
            pairs = [(a, b) for a in a_values for b in b_values]
            rng.shuffle(pairs)

            validation_count, test_count = holdout_counts(
                len(pairs),
                eval_per_partition,
                small_partition_eval_count,
            )
            train_count = len(pairs) - validation_count - test_count
            if train_count <= 0:
                raise ValueError(f"Partition {label} has no train examples after holdout.")

            boundaries = {
                "validation": pairs[:validation_count],
                "test": pairs[validation_count : validation_count + test_count],
                "train": pairs[validation_count + test_count :],
            }
            for split, chunk in boundaries.items():
                counts[split][label] = len(chunk)
                splits[split].extend(
                    RectangularMultiplicationExample(
                        a=a,
                        b=b,
                        a_digits=a_digits,
                        b_digits=b_digits,
                        format_version=format_version,
                    )
                    for a, b in chunk
                )

    rng.shuffle(splits["train"])
    rng.shuffle(splits["validation"])
    rng.shuffle(splits["test"])
    return splits, counts


def example_partition(example: RectangularMultiplicationExample) -> str:
    return partition_label((example.a_digits, example.b_digits))


def evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[RectangularMultiplicationExample],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    total = len(examples)
    correct = 0
    partition_totals: Dict[str, int] = defaultdict(int)
    partition_correct: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for start in range(0, total, batch_size):
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
                label = example_partition(example)
                partition_totals[label] += 1
                generated_slice = output_ids[idx, prompt_width:].tolist()
                raw_text = tokenizer.decode(generated_slice, skip_special_tokens=True)
                prediction = parse_prediction(parse_rectangular_multiplication_prediction, raw_text, example)
                if prediction == example.target():
                    correct += 1
                    partition_correct[label] += 1

    if model_was_training:
        model.train()

    per_partition_accuracy = {
        label: partition_correct[label] / count if count > 0 else math.nan
        for label, count in sorted(partition_totals.items())
    }
    min_partition_accuracy = min(per_partition_accuracy.values()) if per_partition_accuracy else None
    return {
        "count": total,
        "accuracy": correct / total if total > 0 else math.nan,
        "per_partition_accuracy": per_partition_accuracy,
        "min_partition_accuracy": min_partition_accuracy,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] No precision flag provided; defaulting to bf16 on CUDA.", flush=True)
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    with (output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(vars(args)), handle, indent=2)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    splits, partition_counts = build_exhaustive_splits(
        min_a_digits=args.min_a_digits,
        max_a_digits=args.max_a_digits,
        min_b_digits=args.min_b_digits,
        max_b_digits=args.max_b_digits,
        eval_per_partition=args.eval_per_partition,
        small_partition_eval_count=args.small_partition_eval_count,
        format_version=args.format_version,
        rng=rng,
    )

    print(
        "[INFO] Rectangular dataset sizes -- train: {} | validation: {} | test: {}".format(
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        ),
        flush=True,
    )
    print(f"[INFO] Train counts by partition: {partition_counts['train']}", flush=True)
    print(f"[INFO] Validation counts by partition: {partition_counts['validation']}", flush=True)
    print(f"[INFO] Test counts by partition: {partition_counts['test']}", flush=True)

    token_initializers = {"×": "*"} if args.format_version == "symbolic_v1" else {}
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        token_initializers=token_initializers,
        init_from_scratch=args.init_from_scratch,
        tokenizer_mode=args.tokenizer_mode,
    )
    if getattr(model, "generation_config", None) is not None and tokenizer.pad_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        eval_steps=None,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(
            output_dir / "trainer",
            config,
            bf16=args.bf16,
            fp16=args.fp16,
            skip_save=True,
            seed=args.seed,
        ),
        train_dataset=TokenizedPromptTargetDataset(splits["train"], tokenizer),
        eval_dataset=None,
        data_collator=CausalLMDataCollator(tokenizer),
    )
    train_result = trainer.train()
    model = trainer.model

    decode_max_new_tokens = max(
        resolve_max_new_tokens(splits["train"], config.decode_max_new_tokens),
        resolve_max_new_tokens(splits["validation"], config.decode_max_new_tokens),
        resolve_max_new_tokens(splits["test"], config.decode_max_new_tokens),
    )

    validation_results = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=splits["validation"],
        batch_size=config.per_device_eval_batch_size,
        max_new_tokens=decode_max_new_tokens,
    )
    test_results = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=splits["test"],
        batch_size=config.per_device_eval_batch_size,
        max_new_tokens=decode_max_new_tokens,
    )

    payload = {
        "task": "multiplication_rectangular",
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "seed": args.seed,
        "format_version": args.format_version,
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "partition_counts": partition_counts,
        "training": {
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps if args.max_steps > 0 else None,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.per_device_train_batch_size * args.gradient_accumulation_steps,
            "train_result_metrics": sanitize_json_value(train_result.metrics),
            "log_history": sanitize_json_value(trainer.state.log_history),
            "final_epoch": sanitize_json_value(float(trainer.state.epoch) if trainer.state.epoch is not None else None),
        },
        "results": {
            "validation": validation_results,
            "test": test_results,
        },
    }
    with (output_dir / "rectangular_seed_fit_results.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)
    print(json.dumps(
        {
            "task": payload["task"],
            "train_examples": payload["train_examples"],
            "validation_min_partition_accuracy": validation_results["min_partition_accuracy"],
            "test_min_partition_accuracy": test_results["min_partition_accuracy"],
        }
    ))
    print(f"[INFO] Saved rectangular seed-fit results to {output_dir / 'rectangular_seed_fit_results.json'}", flush=True)


if __name__ == "__main__":
    main()

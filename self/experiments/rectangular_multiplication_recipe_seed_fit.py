#!/usr/bin/env python3
"""Sampled rectangular multiplication seed-fit experiment with recipe support."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from transformers import set_seed

from self.tasks.rectangular_multiplication import (
    RECTANGULAR_MULTIPLICATION_FORMATS,
    RectangularMultiplicationExample,
    build_sampled_rectangular_dataset,
    iter_partition_grid,
    parse_partition_spec,
    partition_bucket_id,
    partition_label,
    prediction_matches_example,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples
from self.core.evaluation import build_generation_encodings, resolve_max_new_tokens
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.task_protocols import JsonDict
from self.core.training import (
    CausalLMDataCollator,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    build_trainer,
    make_training_args,
)
from self.core.recipes import (
    PaddingAwareCausalLMDataCollator,
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
    recipe_enabled,
    resolve_self_improvement_recipe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sampled rectangular multiplication seed-fit experiment.")
    parser.add_argument("--model-name", type=str, default="meta/models/tiny_gpt2_8l_384d")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--format-version",
        type=str,
        default="symbolic_v1",
        choices=sorted(RECTANGULAR_MULTIPLICATION_FORMATS),
    )
    parser.add_argument(
        "--recipe",
        type=str,
        choices=("none", RECIPE_ARITHMETIC_SELF_IMPROVE_V1, "algorithmic_self_improve_v1", RECIPE_MULTIPLICATION_SELF_IMPROVE_V1),
        default=RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help="Optional comma-separated AxB partition list. When set, overrides the min/max digit grid.",
    )
    parser.add_argument("--min-a-digits", type=int, default=1)
    parser.add_argument("--max-a-digits", type=int, default=6)
    parser.add_argument("--min-b-digits", type=int, default=1)
    parser.add_argument("--max-b-digits", type=int, default=6)
    parser.add_argument("--train-per-partition", type=int, default=50_000)
    parser.add_argument("--heldout-per-partition", type=int, default=200)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--decode-max-new-tokens", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--init-from-scratch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer-mode", type=str, choices=("auto", "fixed_char"), default="auto")
    parser.add_argument(
        "--auto-find-batch-size",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--bucket-train-batches-by-partition",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-train-eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the expensive full-train generative evaluation and only score validation/test splits.",
    )
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


@contextmanager
def tokenizer_padding_side(tokenizer: Any, side: str) -> Iterator[None]:
    original = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = side
    try:
        yield
    finally:
        tokenizer.padding_side = original


def evaluate_examples(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[RectangularMultiplicationExample],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    if not examples:
        return {
            "count": 0,
            "accuracy": math.nan,
            "per_partition_accuracy": {},
            "min_partition_accuracy": None,
        }

    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()

    total = len(examples)
    correct = 0
    partition_totals: Dict[str, int] = {}
    partition_correct: Dict[str, int] = {}

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
            for index, example in enumerate(batch):
                partition = partition_label((example.a_digits, example.b_digits))
                partition_totals[partition] = partition_totals.get(partition, 0) + 1
                generated_slice = output_ids[index, prompt_width:].tolist()
                raw_text = tokenizer.decode(generated_slice, skip_special_tokens=True)
                if prediction_matches_example(raw_text, example):
                    correct += 1
                    partition_correct[partition] = partition_correct.get(partition, 0) + 1

    if model_was_training:
        model.train()

    per_partition_accuracy = {
        partition: partition_correct.get(partition, 0) / count if count > 0 else math.nan
        for partition, count in sorted(partition_totals.items())
    }
    min_partition_accuracy = min(per_partition_accuracy.values()) if per_partition_accuracy else None
    return {
        "count": total,
        "accuracy": correct / total if total > 0 else math.nan,
        "per_partition_accuracy": per_partition_accuracy,
        "min_partition_accuracy": min_partition_accuracy,
    }


def skipped_evaluation_summary(total: int) -> Dict[str, Any]:
    return {
        "count": total,
        "accuracy": None,
        "per_partition_accuracy": {},
        "min_partition_accuracy": None,
        "skipped": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")
    if args.recipe == RECIPE_MULTIPLICATION_SELF_IMPROVE_V1 and args.format_version != "symbolic_v1":
        raise ValueError("multiplication_self_improve_v1 only supports format_version='symbolic_v1'.")

    recipe_preset = resolve_self_improvement_recipe(args.recipe) if recipe_enabled(args.recipe) else None
    if recipe_preset is not None:
        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = recipe_preset.per_device_train_batch_size
        if args.per_device_eval_batch_size is None:
            args.per_device_eval_batch_size = recipe_preset.per_device_eval_batch_size
        if args.max_steps is None:
            args.max_steps = recipe_preset.seed_phase.max_steps
        if args.auto_find_batch_size is None:
            args.auto_find_batch_size = recipe_preset.auto_find_batch_size
        if args.decode_max_new_tokens == 64:
            args.decode_max_new_tokens = recipe_preset.decode_max_new_tokens
        if args.learning_rate is None:
            args.learning_rate = recipe_preset.seed_phase.learning_rate
    else:
        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = 32
        if args.per_device_eval_batch_size is None:
            args.per_device_eval_batch_size = 64
        if args.auto_find_batch_size is None:
            args.auto_find_batch_size = False
        if args.learning_rate is None:
            args.learning_rate = 5e-5

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    with (output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(vars(args)), handle, indent=2)

    if args.partitions:
        partitions = parse_partition_spec(args.partitions)
    else:
        partitions = iter_partition_grid(
            args.min_a_digits,
            args.max_a_digits,
            args.min_b_digits,
            args.max_b_digits,
        )
    dry_run_payload = {
        "recipe": args.recipe,
        "format_version": args.format_version,
        "partitions": [partition_label(partition) for partition in partitions],
        "train_per_partition": args.train_per_partition,
        "heldout_per_partition": args.heldout_per_partition,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "max_steps": args.max_steps,
        "skip_train_eval": args.skip_train_eval,
    }
    if args.dry_run:
        with (output_dir / "dry_run_plan.json").open("w", encoding="utf-8") as handle:
            json.dump(dry_run_payload, handle, indent=2)
        print(json.dumps(dry_run_payload, indent=2))
        return

    set_seed(args.seed)
    rng = random.Random(args.seed)
    record_keys = {name: set() for name in ("train", "validation", "test")}
    splits = build_sampled_rectangular_dataset(
        partitions=partitions,
        per_partition_counts={
            "train": args.train_per_partition,
            "validation": args.heldout_per_partition,
            "test": args.heldout_per_partition,
        },
        rng=rng,
        format_version=args.format_version,
        record_keys=record_keys,
        progress_name="rectangular-seed",
    )

    data_dir = output_dir / "data"
    save_examples(data_dir / "train_examples.jsonl", splits["train"], lambda example: {
        "a": example.a,
        "b": example.b,
        "a_digits": example.a_digits,
        "b_digits": example.b_digits,
        "format_version": example.format_version,
        "target_override": example.target_override,
    })
    save_examples(data_dir / "validation_examples.jsonl", splits["validation"], lambda example: {
        "a": example.a,
        "b": example.b,
        "a_digits": example.a_digits,
        "b_digits": example.b_digits,
        "format_version": example.format_version,
        "target_override": example.target_override,
    })
    save_examples(data_dir / "test_examples.jsonl", splits["test"], lambda example: {
        "a": example.a,
        "b": example.b,
        "a_digits": example.a_digits,
        "b_digits": example.b_digits,
        "format_version": example.format_version,
        "target_override": example.target_override,
    })

    model, tokenizer = instantiate_model_and_tokenizer(
        args.model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=bool(args.init_from_scratch),
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    training_config = TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        eval_steps=None,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )
    training_args = make_training_args(
        output_dir / "trainer",
        training_config,
        bf16=args.bf16,
        fp16=args.fp16,
        skip_save=not bool(args.save_model),
        seed=args.seed,
        recipe=args.recipe,
        recipe_phase_name="seed",
        recipe_phase_overrides={"learning_rate": args.learning_rate} if recipe_enabled(args.recipe) else None,
    )
    if hasattr(training_args, "auto_find_batch_size"):
        setattr(training_args, "auto_find_batch_size", bool(args.auto_find_batch_size))

    train_dataset = TokenizedPromptTargetDataset(splits["train"], tokenizer)
    if recipe_enabled(args.recipe):
        data_collator = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")
    else:
        data_collator = CausalLMDataCollator(tokenizer)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        seed=args.seed,
        size_getter=partition_bucket_id,
        bucket_train_batches_by_size=bool(args.bucket_train_batches_by_partition),
        recipe=args.recipe,
        recipe_phase_name="seed",
    )
    train_result = trainer.train()
    if args.save_model:
        trainer.save_model(str(output_dir / "model"))
        tokenizer.save_pretrained(output_dir / "model")

    eval_budget = max(
        resolve_max_new_tokens(splits["validation"], args.decode_max_new_tokens),
        resolve_max_new_tokens(splits["test"], args.decode_max_new_tokens),
    )
    if not args.skip_train_eval:
        eval_budget = max(
            eval_budget,
            resolve_max_new_tokens(splits["train"], args.decode_max_new_tokens),
        )
    with tokenizer_padding_side(tokenizer, "left"):
        if args.skip_train_eval:
            train_summary = skipped_evaluation_summary(len(splits["train"]))
        else:
            train_summary = evaluate_examples(
                model=trainer.model,
                tokenizer=tokenizer,
                examples=splits["train"],
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
        validation_summary = evaluate_examples(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=splits["validation"],
            batch_size=args.per_device_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        test_summary = evaluate_examples(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=splits["test"],
            batch_size=args.per_device_eval_batch_size,
            max_new_tokens=eval_budget,
        )

    payload: JsonDict = {
        "task": "multiplication_rectangular_seed_fit",
        "recipe": args.recipe,
        "format_version": args.format_version,
        "train_examples": len(splits["train"]),
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "partitions": [partition_label(partition) for partition in partitions],
        "training": sanitize_json_value(train_result.metrics),
        "results": {
            "train": train_summary,
            "validation": validation_summary,
            "test": test_summary,
        },
        "model_dir": str(output_dir / "model") if args.save_model else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)

    print(
        json.dumps(
            {
                "task": payload["task"],
                "validation_min_partition_accuracy": validation_summary["min_partition_accuracy"],
                "test_min_partition_accuracy": test_summary["min_partition_accuracy"],
            }
        )
    )


if __name__ == "__main__":
    main()

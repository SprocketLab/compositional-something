#!/usr/bin/env python3
"""Frontier diagnostic for rectangular multiplication with direct or composed pseudo labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from transformers import set_seed

from self.tasks.rectangular_multiplication import (
    EDGE_ONLY_MULTIPLICATION_PARTITIONS,
    RECTANGULAR_MULTIPLICATION_FORMATS,
    RectangularMultiplicationExample,
    build_partition_supported_components,
    build_sampled_rectangular_dataset,
    compose_target_from_weighted_component_values,
    normalize_rectangular_prediction_for_training,
    parse_partition_spec,
    parse_rectangular_multiplication_final_value,
    partition_bucket_id,
    partition_label,
    prediction_matches_example,
    rectangular_multiplication_key,
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
from self.core.recipe_presets import (
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
    recipe_enabled,
    resolve_self_improvement_recipe,
)
from self.core.recipe_training import PaddingAwareCausalLMDataCollator


DEFAULT_EDGE_PARTITIONS_SPEC = ",".join(partition_label(partition) for partition in EDGE_ONLY_MULTIPLICATION_PARTITIONS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose multiplication frontier fits with upstream-style composition.")
    parser.add_argument("--seed-model", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--recipe",
        type=str,
        choices=("none", RECIPE_ARITHMETIC_SELF_IMPROVE_V1, "algorithmic_self_improve_v1", RECIPE_MULTIPLICATION_SELF_IMPROVE_V1),
        default=RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
    )
    parser.add_argument("--tokenizer-mode", type=str, choices=("auto", "fixed_char"), default="auto")
    parser.add_argument(
        "--format-version",
        type=str,
        default="symbolic_v1",
        choices=sorted(RECTANGULAR_MULTIPLICATION_FORMATS),
    )
    parser.add_argument(
        "--seed-partitions",
        type=str,
        default=DEFAULT_EDGE_PARTITIONS_SPEC,
        help="Comma-separated AxB seed partitions used to determine compose leaves.",
    )
    parser.add_argument("--frontier-partitions", type=str, required=True, help="Comma-separated AxB pairs, e.g. 1x6,6x1.")
    parser.add_argument("--train-per-partition", type=int, default=5_000)
    parser.add_argument("--heldout-per-partition", type=int, default=200)
    parser.add_argument("--mode", type=str, choices=("gold", "direct", "compose"), default="compose")
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--decode-max-new-tokens", type=int, default=64)
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


def serialize_example(example: RectangularMultiplicationExample) -> JsonDict:
    return {
        "a": example.a,
        "b": example.b,
        "a_digits": example.a_digits,
        "b_digits": example.b_digits,
        "format_version": example.format_version,
        "target_override": example.target_override,
    }


def clone_with_override(
    example: RectangularMultiplicationExample,
    override: Optional[str],
) -> RectangularMultiplicationExample:
    if override is None:
        return example
    return RectangularMultiplicationExample(
        a=example.a,
        b=example.b,
        a_digits=example.a_digits,
        b_digits=example.b_digits,
        format_version=example.format_version,
        target_override=override,
    )


def generate_raw_output_map(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[RectangularMultiplicationExample],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[Tuple[int, int, int, int], str]:
    if not examples:
        return {}

    device = next(model.parameters()).device
    unique_examples: Dict[Tuple[int, int, int, int], RectangularMultiplicationExample] = {}
    for example in examples:
        key = rectangular_multiplication_key(example)
        if key not in unique_examples:
            unique_examples[key] = example

    keys = list(unique_examples.keys())
    values = [unique_examples[key] for key in keys]
    predictions: Dict[Tuple[int, int, int, int], str] = {}

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = values[start : start + batch_size]
            prompts = [example.prompt() for example in batch]
            encodings = build_generation_encodings(tokenizer, prompts, device)
            output_ids = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt_width = encodings["input_ids"].shape[1]
            for index, example in enumerate(batch):
                generated_slice = output_ids[index, prompt_width:].tolist()
                predictions[rectangular_multiplication_key(example)] = tokenizer.decode(
                    generated_slice,
                    skip_special_tokens=True,
                )

    if model_was_training:
        model.train()
    return predictions


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


def build_direct_pseudo_examples(
    *,
    train_examples: Sequence[RectangularMultiplicationExample],
    raw_output_map: Dict[Tuple[int, int, int, int], str],
) -> Tuple[List[RectangularMultiplicationExample], List[JsonDict], JsonDict]:
    pseudo_examples: List[RectangularMultiplicationExample] = []
    debug_rows: List[JsonDict] = []
    missing_total = 0
    exact_target_total = 0
    correct_value_total = 0

    for example in train_examples:
        key = rectangular_multiplication_key(example)
        raw_output = raw_output_map.get(key)
        if raw_output is None:
            missing_total += 1
            continue
        override = normalize_rectangular_prediction_for_training(raw_output, example)
        if override is None:
            missing_total += 1
            continue
        pseudo_example = clone_with_override(example, override)
        pseudo_examples.append(pseudo_example)
        if override == example.target():
            exact_target_total += 1
        if parse_rectangular_multiplication_final_value(override, example) == (example.a * example.b):
            correct_value_total += 1
        debug_rows.append(
            {
                "partition": partition_label((example.a_digits, example.b_digits)),
                "prompt": example.prompt(),
                "gold_target": example.target(),
                "raw_output": raw_output,
                "pseudo_target": override,
                "gold_value": example.a * example.b,
                "pseudo_value": parse_rectangular_multiplication_final_value(override, example),
                "correct_value": parse_rectangular_multiplication_final_value(override, example) == (example.a * example.b),
                "exact_target_match": override == example.target(),
            }
        )

    diagnostics: JsonDict = {
        "mode": "direct",
        "candidate_total": len(train_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "exact_target_total": exact_target_total,
        "correct_value_total": correct_value_total,
    }
    return pseudo_examples, debug_rows, diagnostics


def build_composed_pseudo_examples(
    *,
    train_examples: Sequence[RectangularMultiplicationExample],
    raw_output_map: Dict[Tuple[int, int, int, int], str],
    supported_partitions: Sequence[Tuple[int, int]],
) -> Tuple[List[RectangularMultiplicationExample], List[JsonDict], JsonDict]:
    pseudo_examples: List[RectangularMultiplicationExample] = []
    debug_rows: List[JsonDict] = []
    missing_total = 0
    exact_target_total = 0
    correct_value_total = 0
    zero_shortcut_component_total = 0

    for example in train_examples:
        weighted_values: List[Tuple[int, int]] = []
        component_rows: List[JsonDict] = []
        missing_component = False
        for component in build_partition_supported_components(
            example,
            supported_partitions=supported_partitions,
        ):
            component_example = component.example
            if component_example.a == 0 or component_example.b == 0:
                parsed_value = 0
                raw_output = None
                zero_shortcut_component_total += 1
            else:
                raw_output = raw_output_map.get(rectangular_multiplication_key(component_example))
                if raw_output is None:
                    missing_component = True
                    break
                parsed_value = parse_rectangular_multiplication_final_value(raw_output, component_example)
                if parsed_value is None:
                    missing_component = True
                    break
            weighted_values.append((component.shift_digits, parsed_value))
            component_rows.append(
                {
                    "shift_digits": component.shift_digits,
                    "component_prompt": component_example.prompt(),
                    "component_gold_target": component_example.target(),
                    "component_raw_output": raw_output,
                    "component_source": "zero_shortcut" if component_example.a == 0 or component_example.b == 0 else "model",
                    "component_parsed_value": parsed_value,
                    "component_gold_value": component_example.a * component_example.b,
                }
            )
        if missing_component:
            missing_total += 1
            continue
        override = compose_target_from_weighted_component_values(example, weighted_values)
        pseudo_example = clone_with_override(example, override)
        pseudo_examples.append(pseudo_example)
        pseudo_value = parse_rectangular_multiplication_final_value(override, example)
        if override == example.target():
            exact_target_total += 1
        if pseudo_value == (example.a * example.b):
            correct_value_total += 1
        debug_rows.append(
            {
                "partition": partition_label((example.a_digits, example.b_digits)),
                "prompt": example.prompt(),
                "gold_target": example.target(),
                "pseudo_target": override,
                "gold_value": example.a * example.b,
                "pseudo_value": pseudo_value,
                "correct_value": pseudo_value == (example.a * example.b),
                "exact_target_match": override == example.target(),
                "components": component_rows,
            }
        )

    diagnostics: JsonDict = {
        "mode": "compose",
        "candidate_total": len(train_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "exact_target_total": exact_target_total,
        "correct_value_total": correct_value_total,
        "zero_shortcut_component_total": zero_shortcut_component_total,
    }
    return pseudo_examples, debug_rows, diagnostics


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.recipe == RECIPE_MULTIPLICATION_SELF_IMPROVE_V1 and args.format_version != "symbolic_v1":
        raise ValueError("multiplication_self_improve_v1 only supports format_version='symbolic_v1'.")

    if not Path(args.seed_model).exists():
        raise FileNotFoundError(f"Seed model path does not exist: {args.seed_model}")
    if args.mode != "gold" and args.format_version == "legacy":
        print("[WARN] Legacy multiplication frontier diagnostics only keep final answers as pseudo labels.", flush=True)

    recipe_preset = resolve_self_improvement_recipe(args.recipe) if recipe_enabled(args.recipe) else None
    if recipe_preset is not None:
        if args.per_device_train_batch_size is None:
            args.per_device_train_batch_size = recipe_preset.per_device_train_batch_size
        if args.per_device_eval_batch_size is None:
            args.per_device_eval_batch_size = recipe_preset.per_device_eval_batch_size
        if args.max_steps is None:
            args.max_steps = recipe_preset.self_improve_phase.max_steps
        if args.auto_find_batch_size is None:
            args.auto_find_batch_size = recipe_preset.auto_find_batch_size
        if args.decode_max_new_tokens == 64:
            args.decode_max_new_tokens = recipe_preset.decode_max_new_tokens
        if args.learning_rate is None:
            args.learning_rate = recipe_preset.self_improve_phase.learning_rate
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

    partitions = parse_partition_spec(args.frontier_partitions)
    seed_partitions = parse_partition_spec(args.seed_partitions)
    dry_run_payload = {
        "seed_model": args.seed_model,
        "recipe": args.recipe,
        "format_version": args.format_version,
        "mode": args.mode,
        "seed_partitions": [partition_label(partition) for partition in seed_partitions],
        "frontier_partitions": [partition_label(partition) for partition in partitions],
        "train_per_partition": args.train_per_partition,
        "heldout_per_partition": args.heldout_per_partition,
        "max_steps": args.max_steps,
    }
    if args.dry_run:
        with (output_dir / "dry_run_plan.json").open("w", encoding="utf-8") as handle:
            json.dump(dry_run_payload, handle, indent=2)
        print(json.dumps(dry_run_payload, indent=2))
        return

    set_seed(args.seed)
    rng = random.Random(args.seed)
    splits = build_sampled_rectangular_dataset(
        partitions=partitions,
        per_partition_counts={
            "train": args.train_per_partition,
            "validation": args.heldout_per_partition,
            "test": args.heldout_per_partition,
        },
        rng=rng,
        format_version=args.format_version,
        progress_name="frontier",
    )

    data_dir = output_dir / "data"
    save_examples(data_dir / "frontier_train_examples.jsonl", splits["train"], serialize_example)
    save_examples(data_dir / "frontier_validation_examples.jsonl", splits["validation"], serialize_example)
    save_examples(data_dir / "frontier_test_examples.jsonl", splits["test"], serialize_example)

    model, tokenizer = instantiate_model_and_tokenizer(
        args.seed_model,
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )

    with tokenizer_padding_side(tokenizer, "left"):
        if args.mode == "gold":
            pseudo_examples = list(splits["train"])
            debug_rows = [
                {
                    "partition": partition_label((example.a_digits, example.b_digits)),
                    "prompt": example.prompt(),
                    "gold_target": example.target(),
                    "pseudo_target": example.target(),
                    "gold_value": example.a * example.b,
                    "pseudo_value": example.a * example.b,
                    "correct_value": True,
                    "exact_target_match": True,
                }
                for example in splits["train"]
            ]
            pseudo_diagnostics: JsonDict = {
                "mode": "gold",
                "candidate_total": len(splits["train"]),
                "retained_total": len(splits["train"]),
                "missing_total": 0,
                "exact_target_total": len(splits["train"]),
                "correct_value_total": len(splits["train"]),
            }
        elif args.mode == "direct":
            raw_output_map = generate_raw_output_map(
                model=model,
                tokenizer=tokenizer,
                examples=splits["train"],
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=args.decode_max_new_tokens,
            )
            pseudo_examples, debug_rows, pseudo_diagnostics = build_direct_pseudo_examples(
                train_examples=splits["train"],
                raw_output_map=raw_output_map,
            )
        else:
            component_examples: List[RectangularMultiplicationExample] = []
            for example in splits["train"]:
                for component in build_partition_supported_components(
                    example,
                    supported_partitions=seed_partitions,
                ):
                    if component.example.a == 0 or component.example.b == 0:
                        continue
                    component_examples.append(component.example)
            raw_output_map = generate_raw_output_map(
                model=model,
                tokenizer=tokenizer,
                examples=component_examples,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=args.decode_max_new_tokens,
            )
            pseudo_examples, debug_rows, pseudo_diagnostics = build_composed_pseudo_examples(
                train_examples=splits["train"],
                raw_output_map=raw_output_map,
                supported_partitions=seed_partitions,
            )

    if not pseudo_examples:
        raise RuntimeError(f"No training examples survived pseudo generation for mode={args.mode!r}.")

    save_examples(data_dir / "pseudo_train_examples.jsonl", pseudo_examples, serialize_example)
    with (data_dir / "pseudo_generation_debug.jsonl").open("w", encoding="utf-8") as handle:
        for row in debug_rows:
            json.dump(sanitize_json_value(row), handle)
            handle.write("\n")

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
        bf16=True if torch.cuda.is_available() else False,
        fp16=False,
        skip_save=not bool(args.save_model),
        seed=args.seed,
        recipe=args.recipe,
        recipe_phase_name="frontier",
        recipe_phase_overrides={"learning_rate": args.learning_rate} if recipe_enabled(args.recipe) else None,
    )
    if hasattr(training_args, "auto_find_batch_size"):
        setattr(training_args, "auto_find_batch_size", bool(args.auto_find_batch_size))

    train_dataset = TokenizedPromptTargetDataset(pseudo_examples, tokenizer)
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
        recipe_phase_name="frontier",
    )
    train_result = trainer.train()
    if args.save_model:
        trainer.save_model(str(output_dir / "model"))
        tokenizer.save_pretrained(output_dir / "model")

    eval_budget = max(
        resolve_max_new_tokens(pseudo_examples, args.decode_max_new_tokens),
        resolve_max_new_tokens(splits["validation"], args.decode_max_new_tokens),
        resolve_max_new_tokens(splits["test"], args.decode_max_new_tokens),
    )
    with tokenizer_padding_side(tokenizer, "left"):
        train_summary = evaluate_examples(
            model=trainer.model,
            tokenizer=tokenizer,
            examples=pseudo_examples,
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

    summary = {
        "task": "multiplication_rectangular_frontier_diagnostic",
        "recipe": args.recipe,
        "format_version": args.format_version,
        "mode": args.mode,
        "frontier_partitions": [partition_label(partition) for partition in partitions],
        "pseudo_generation": pseudo_diagnostics,
        "train_examples": len(pseudo_examples),
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "training": sanitize_json_value(train_result.metrics),
        "results": {
            "train": train_summary,
            "validation": validation_summary,
            "test": test_summary,
        },
        "model_dir": str(output_dir / "model") if args.save_model else None,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(summary), handle, indent=2)

    print(
        json.dumps(
            {
                "task": summary["task"],
                "mode": args.mode,
                "validation_min_partition_accuracy": validation_summary["min_partition_accuracy"],
                "test_min_partition_accuracy": test_summary["min_partition_accuracy"],
                "pseudo_retained_total": pseudo_diagnostics["retained_total"],
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Iterative self-improvement for rectangular multiplication using an edge-only seed.

This path grows multiplier width while keeping the multiplicand width inside the
seed-supported range. The round-0 seed is assumed to cover the asymmetric edge
partitions `1x1..6 ∪ 1..6x1`. Each round:

1. Samples fresh direct-pseudo replay from the seed edge partitions.
2. Samples fresh frontier examples for the active non-seed rectangle.
3. Pseudo-labels the frontier either directly, compositionally, or with
   composition-time corruption.
4. Fine-tunes the current model for one recipe self-improvement phase.

The script saves round metrics in `self_improvement_results.json` plus per-round
artifacts under `round_XX/`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import set_seed

from self.tasks.rectangular_multiplication import (
    EDGE_ONLY_MULTIPLICATION_PARTITIONS,
    RECTANGULAR_MULTIPLICATION_FORMATS,
    RectangularMultiplicationExample,
    build_partition_supported_components,
    compose_target_from_weighted_component_values,
    build_sampled_rectangular_dataset,
    normalize_rectangular_prediction_for_training,
    parse_partition_spec,
    parse_rectangular_multiplication_final_value,
    partition_bucket_id,
    partition_label,
    rectangular_multiplication_key,
)
from self.diagnostics.rectangular_multiplication_compose_diagnostic import (
    clone_with_override,
    evaluate_examples,
    generate_raw_output_map,
    serialize_example,
    tokenizer_padding_side,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples
from self.core.evaluation import resolve_max_new_tokens
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
FRONTIER_ROW_PROFILES: Dict[str, Dict[int, int]] = {
    "uniform": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1},
    "hard_rows_v1": {1: 1, 2: 4, 3: 4, 4: 2, 5: 1, 6: 1},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rectangular multiplication self-improvement from an edge-only seed.")
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
        help="Comma-separated AxB seed partitions used for replay.",
    )
    parser.add_argument("--frontier-min-a-digits", type=int, default=1)
    parser.add_argument("--frontier-max-a-digits", type=int, default=6)
    parser.add_argument(
        "--frontier-min-b-digits",
        type=int,
        default=2,
        help="Minimum multiplier width included in the non-seed frontier.",
    )
    parser.add_argument(
        "--initial-max-b-digits",
        type=int,
        default=8,
        help="Initial frontier multiplier width after loading the seed.",
    )
    parser.add_argument("--num-expand-rounds", type=int, default=4)
    parser.add_argument("--expand-b-digits", type=int, default=2)
    parser.add_argument("--seed-replay-train-per-partition", type=int, default=2_000)
    parser.add_argument("--expand-train-per-partition", type=int, default=2_000)
    parser.add_argument(
        "--frontier-row-profile",
        type=str,
        choices=tuple(FRONTIER_ROW_PROFILES),
        default="uniform",
        help="Optional per-row frontier sampling multiplier profile.",
    )
    parser.add_argument("--heldout-per-partition", type=int, default=200)
    parser.add_argument(
        "--pseudo-label-mode",
        type=str,
        choices=("short_only", "direct", "compose", "compose_corrupt"),
        default="compose",
    )
    parser.add_argument("--corruption-rate", type=float, default=0.10)
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


def build_partition_grid(
    *,
    min_a_digits: int,
    max_a_digits: int,
    min_b_digits: int,
    max_b_digits: int,
    exclude: Optional[Iterable[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    excluded = set(exclude or ())
    partitions: List[Tuple[int, int]] = []
    for a_digits in range(min_a_digits, max_a_digits + 1):
        for b_digits in range(min_b_digits, max_b_digits + 1):
            partition = (a_digits, b_digits)
            if partition in excluded:
                continue
            partitions.append(partition)
    return partitions


def keys_for_examples(examples: Sequence[RectangularMultiplicationExample]) -> set[Tuple[int, int, int, int]]:
    return {rectangular_multiplication_key(example) for example in examples}


def filter_examples_by_max_b_digits(
    examples: Sequence[RectangularMultiplicationExample],
    max_b_digits: int,
) -> List[RectangularMultiplicationExample]:
    return [example for example in examples if example.b_digits <= max_b_digits]


def frontier_row_multiplier(frontier_row_profile: str, a_digits: int) -> int:
    profile = FRONTIER_ROW_PROFILES[frontier_row_profile]
    return profile.get(a_digits, 1)


def build_frontier_partition_train_counts(
    *,
    partitions: Sequence[Tuple[int, int]],
    base_count: int,
    frontier_row_profile: str,
) -> Dict[Tuple[int, int], int]:
    return {
        partition: base_count * frontier_row_multiplier(frontier_row_profile, partition[0])
        for partition in partitions
    }


def sample_frontier_train_examples(
    *,
    partitions: Sequence[Tuple[int, int]],
    base_count: int,
    frontier_row_profile: str,
    rng: random.Random,
    format_version: str,
    exclude_keys: Optional[set[Tuple[int, int, int, int]]] = None,
    progress_name: Optional[str] = None,
) -> List[RectangularMultiplicationExample]:
    per_partition_counts = build_frontier_partition_train_counts(
        partitions=partitions,
        base_count=base_count,
        frontier_row_profile=frontier_row_profile,
    )
    examples: List[RectangularMultiplicationExample] = []
    for partition in partitions:
        count = per_partition_counts[partition]
        if count <= 0:
            continue
        partition_splits = build_sampled_rectangular_dataset(
            partitions=[partition],
            per_partition_counts={"train": count, "validation": 0, "test": 0},
            rng=rng,
            format_version=format_version,
            exclude_keys=exclude_keys,
            progress_name=progress_name,
        )
        examples.extend(partition_splits["train"])
    rng.shuffle(examples)
    return examples


def summarize_accuracy_by_a_digits(summary: JsonDict) -> JsonDict:
    per_partition_accuracy = summary.get("per_partition_accuracy") or {}
    grouped: Dict[str, List[float]] = {}
    for partition, accuracy in per_partition_accuracy.items():
        if accuracy is None:
            continue
        try:
            numeric_accuracy = float(accuracy)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_accuracy):
            continue
        a_digits = partition.split("x", 1)[0]
        grouped.setdefault(a_digits, []).append(numeric_accuracy)

    payload = dict(summary)
    payload["mean_accuracy_by_a_digits"] = {
        a_digits: sum(values) / len(values)
        for a_digits, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    }
    payload["min_accuracy_by_a_digits"] = {
        a_digits: min(values)
        for a_digits, values in sorted(grouped.items(), key=lambda item: int(item[0]))
    }
    return payload


def build_direct_pseudo_examples(
    *,
    train_examples: Sequence[RectangularMultiplicationExample],
    raw_output_map: Dict[Tuple[int, int, int, int], str],
    diagnostics_mode: str,
) -> Tuple[List[RectangularMultiplicationExample], JsonDict]:
    pseudo_examples: List[RectangularMultiplicationExample] = []
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
        pseudo_examples.append(clone_with_override(example, override))
        if override == example.target():
            exact_target_total += 1
        if parse_rectangular_multiplication_final_value(override, example) == (example.a * example.b):
            correct_value_total += 1

    diagnostics: JsonDict = {
        "mode": diagnostics_mode,
        "candidate_total": len(train_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "exact_target_total": exact_target_total,
        "correct_value_total": correct_value_total,
    }
    return pseudo_examples, diagnostics


def build_composed_pseudo_examples(
    *,
    train_examples: Sequence[RectangularMultiplicationExample],
    raw_output_map: Dict[Tuple[int, int, int, int], str],
    supported_partitions: Sequence[Tuple[int, int]],
    corruption_rate: float,
    rng: random.Random,
    diagnostics_mode: str,
) -> Tuple[List[RectangularMultiplicationExample], JsonDict]:
    pseudo_examples: List[RectangularMultiplicationExample] = []
    missing_total = 0
    exact_target_total = 0
    correct_value_total = 0
    corrupted_component_total = 0
    corrupted_example_total = 0
    zero_shortcut_component_total = 0

    for example in train_examples:
        weighted_values: List[Tuple[int, int]] = []
        example_corrupted = False
        missing_component = False
        for component in build_partition_supported_components(
            example,
            supported_partitions=supported_partitions,
        ):
            component_example = component.example
            if component_example.a == 0 or component_example.b == 0:
                parsed_value = 0
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
            if diagnostics_mode == "compose_corrupt" and rng.random() < corruption_rate:
                parsed_value += 1
                corrupted_component_total += 1
                example_corrupted = True
            weighted_values.append((component.shift_digits, parsed_value))
        if missing_component:
            missing_total += 1
            continue

        override = compose_target_from_weighted_component_values(example, weighted_values)
        total_value = parse_rectangular_multiplication_final_value(override, example)
        if total_value is None:
            missing_total += 1
            continue

        pseudo_examples.append(clone_with_override(example, override))
        if override == example.target():
            exact_target_total += 1
        if total_value == (example.a * example.b):
            correct_value_total += 1
        if example_corrupted:
            corrupted_example_total += 1

    diagnostics: JsonDict = {
        "mode": diagnostics_mode,
        "candidate_total": len(train_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "exact_target_total": exact_target_total,
        "correct_value_total": correct_value_total,
        "corruption_rate": corruption_rate if diagnostics_mode == "compose_corrupt" else 0.0,
        "corrupted_component_total": corrupted_component_total,
        "corrupted_example_total": corrupted_example_total,
        "zero_shortcut_component_total": zero_shortcut_component_total,
    }
    return pseudo_examples, diagnostics


def summarize_round(
    *,
    round_index: int,
    max_b_digits: int,
    seed_replay_examples: Sequence[RectangularMultiplicationExample],
    frontier_examples: Sequence[RectangularMultiplicationExample],
    train_examples: Sequence[RectangularMultiplicationExample],
    seed_replay_stats: JsonDict,
    frontier_stats: JsonDict,
    seed_validation_summary: JsonDict,
    seed_test_summary: JsonDict,
    frontier_train_summary: JsonDict,
    train_summary: JsonDict,
    frontier_validation_summary: JsonDict,
    frontier_test_summary: JsonDict,
) -> JsonDict:
    payload: JsonDict = {
        "round": round_index,
        "max_b_digits": max_b_digits,
        "train_examples": len(train_examples),
        "seed_replay_pseudo_examples": len(seed_replay_examples),
        "frontier_pseudo_examples": len(frontier_examples),
        "seed_replay_stats": seed_replay_stats,
        "frontier_stats": frontier_stats,
        "results": {
            "train": train_summary,
            "seed_validation": seed_validation_summary,
            "seed_test": seed_test_summary,
            "frontier_train": summarize_accuracy_by_a_digits(frontier_train_summary),
            "frontier_validation": summarize_accuracy_by_a_digits(frontier_validation_summary),
            "frontier_test": summarize_accuracy_by_a_digits(frontier_test_summary),
        },
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.recipe == RECIPE_MULTIPLICATION_SELF_IMPROVE_V1 and args.format_version != "symbolic_v1":
        raise ValueError("multiplication_self_improve_v1 only supports format_version='symbolic_v1'.")
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")
    if args.corruption_rate < 0.0 or args.corruption_rate > 1.0:
        raise ValueError("corruption_rate must be within [0, 1].")
    if not Path(args.seed_model).exists():
        raise FileNotFoundError(f"Seed model path does not exist: {args.seed_model}")

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
        if args.max_steps is None:
            args.max_steps = 3000
        if args.auto_find_batch_size is None:
            args.auto_find_batch_size = False
        if args.learning_rate is None:
            args.learning_rate = 5e-5

    seed_partitions = parse_partition_spec(args.seed_partitions)
    final_max_b_digits = args.initial_max_b_digits + args.num_expand_rounds * args.expand_b_digits
    all_frontier_partitions = build_partition_grid(
        min_a_digits=args.frontier_min_a_digits,
        max_a_digits=args.frontier_max_a_digits,
        min_b_digits=args.frontier_min_b_digits,
        max_b_digits=final_max_b_digits,
        exclude=seed_partitions,
    )

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    with (output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(vars(args)), handle, indent=2)

    dry_run_payload = {
        "seed_model": args.seed_model,
        "recipe": args.recipe,
        "format_version": args.format_version,
        "seed_partitions": [partition_label(partition) for partition in seed_partitions],
        "frontier_partition_count": len(all_frontier_partitions),
        "initial_max_b_digits": args.initial_max_b_digits,
        "final_max_b_digits": final_max_b_digits,
        "num_expand_rounds": args.num_expand_rounds,
        "expand_b_digits": args.expand_b_digits,
        "seed_replay_train_per_partition": args.seed_replay_train_per_partition,
        "expand_train_per_partition": args.expand_train_per_partition,
        "frontier_row_profile": args.frontier_row_profile,
        "pseudo_label_mode": args.pseudo_label_mode,
    }
    if args.dry_run:
        with (output_dir / "dry_run_plan.json").open("w", encoding="utf-8") as handle:
            json.dump(dry_run_payload, handle, indent=2)
        print(json.dumps(dry_run_payload, indent=2))
        return

    set_seed(args.seed)
    base_rng = random.Random(args.seed)

    # Build fixed held-out sets once, then keep training data disjoint from them.
    seed_eval_splits = build_sampled_rectangular_dataset(
        partitions=seed_partitions,
        per_partition_counts={"train": 0, "validation": args.heldout_per_partition, "test": args.heldout_per_partition},
        rng=random.Random(base_rng.randint(0, 2**31 - 1)),
        format_version=args.format_version,
        progress_name="seed-heldout",
    )
    frontier_eval_splits = build_sampled_rectangular_dataset(
        partitions=all_frontier_partitions,
        per_partition_counts={"train": 0, "validation": args.heldout_per_partition, "test": args.heldout_per_partition},
        rng=random.Random(base_rng.randint(0, 2**31 - 1)),
        format_version=args.format_version,
        progress_name="frontier-heldout",
    )
    heldout_keys = set().union(
        keys_for_examples(seed_eval_splits["validation"]),
        keys_for_examples(seed_eval_splits["test"]),
        keys_for_examples(frontier_eval_splits["validation"]),
        keys_for_examples(frontier_eval_splits["test"]),
    )

    data_dir = output_dir / "data"
    save_examples(data_dir / "seed_validation_examples.jsonl", seed_eval_splits["validation"], serialize_example)
    save_examples(data_dir / "seed_test_examples.jsonl", seed_eval_splits["test"], serialize_example)
    save_examples(data_dir / "frontier_validation_examples.jsonl", frontier_eval_splits["validation"], serialize_example)
    save_examples(data_dir / "frontier_test_examples.jsonl", frontier_eval_splits["test"], serialize_example)

    model, tokenizer = instantiate_model_and_tokenizer(
        args.seed_model,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
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

    round_records: List[JsonDict] = []

    for round_index in range(args.num_expand_rounds + 1):
        current_max_b_digits = args.initial_max_b_digits + round_index * args.expand_b_digits
        round_dir = output_dir / f"round_{round_index:02d}"
        ensure_dir(round_dir)

        frontier_partitions = build_partition_grid(
            min_a_digits=args.frontier_min_a_digits,
            max_a_digits=args.frontier_max_a_digits,
            min_b_digits=args.frontier_min_b_digits,
            max_b_digits=current_max_b_digits,
            exclude=seed_partitions,
        )

        round_rng = random.Random(base_rng.randint(0, 2**31 - 1))
        seed_replay_splits = build_sampled_rectangular_dataset(
            partitions=seed_partitions,
            per_partition_counts={"train": args.seed_replay_train_per_partition, "validation": 0, "test": 0},
            rng=round_rng,
            format_version=args.format_version,
            exclude_keys=heldout_keys,
            progress_name=f"seed-replay-round-{round_index}",
        )
        seed_replay_raw = list(seed_replay_splits["train"])

        frontier_raw: List[RectangularMultiplicationExample] = []
        if args.pseudo_label_mode != "short_only" and frontier_partitions:
            frontier_raw = sample_frontier_train_examples(
                partitions=frontier_partitions,
                base_count=args.expand_train_per_partition,
                frontier_row_profile=args.frontier_row_profile,
                rng=round_rng,
                format_version=args.format_version,
                exclude_keys=heldout_keys,
                progress_name=f"frontier-round-{round_index}",
            )

        save_examples(round_dir / "seed_replay_raw_examples.jsonl", seed_replay_raw, serialize_example)
        save_examples(round_dir / "frontier_raw_examples.jsonl", frontier_raw, serialize_example)

        with tokenizer_padding_side(tokenizer, "left"):
            seed_raw_output_map = generate_raw_output_map(
                model=model,
                tokenizer=tokenizer,
                examples=seed_replay_raw,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=args.decode_max_new_tokens,
            )
            seed_replay_pseudo, seed_replay_stats = build_direct_pseudo_examples(
                train_examples=seed_replay_raw,
                raw_output_map=seed_raw_output_map,
                diagnostics_mode="seed_replay_direct",
            )

            if args.pseudo_label_mode == "short_only":
                frontier_pseudo: List[RectangularMultiplicationExample] = []
                frontier_stats: JsonDict = {
                    "mode": "short_only",
                    "candidate_total": len(frontier_raw),
                    "retained_total": 0,
                    "missing_total": 0,
                }
            elif args.pseudo_label_mode == "direct":
                frontier_raw_output_map = generate_raw_output_map(
                    model=model,
                    tokenizer=tokenizer,
                    examples=frontier_raw,
                    batch_size=args.per_device_eval_batch_size,
                    max_new_tokens=args.decode_max_new_tokens,
                )
                frontier_pseudo, frontier_stats = build_direct_pseudo_examples(
                    train_examples=frontier_raw,
                    raw_output_map=frontier_raw_output_map,
                    diagnostics_mode="frontier_direct",
                )
            else:
                component_examples: List[RectangularMultiplicationExample] = []
                for example in frontier_raw:
                    for component in build_partition_supported_components(
                        example,
                        supported_partitions=seed_partitions,
                    ):
                        if component.example.a == 0 or component.example.b == 0:
                            continue
                        component_examples.append(component.example)
                component_raw_output_map = generate_raw_output_map(
                    model=model,
                    tokenizer=tokenizer,
                    examples=component_examples,
                    batch_size=args.per_device_eval_batch_size,
                    max_new_tokens=args.decode_max_new_tokens,
                )
                frontier_pseudo, frontier_stats = build_composed_pseudo_examples(
                    train_examples=frontier_raw,
                    raw_output_map=component_raw_output_map,
                    supported_partitions=seed_partitions,
                    corruption_rate=args.corruption_rate,
                    rng=round_rng,
                    diagnostics_mode=args.pseudo_label_mode,
                )

        train_examples = [*seed_replay_pseudo, *frontier_pseudo]
        if not train_examples:
            raise RuntimeError(f"Round {round_index} generated no training examples.")

        save_examples(round_dir / "seed_replay_pseudo_examples.jsonl", seed_replay_pseudo, serialize_example)
        save_examples(round_dir / "frontier_pseudo_examples.jsonl", frontier_pseudo, serialize_example)
        save_examples(round_dir / "train_examples.jsonl", train_examples, serialize_example)

        train_dataset = TokenizedPromptTargetDataset(train_examples, tokenizer)
        if recipe_enabled(args.recipe):
            data_collator = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")
        else:
            data_collator = CausalLMDataCollator(tokenizer)
        training_args = make_training_args(
            round_dir,
            training_config,
            bf16=args.bf16,
            fp16=args.fp16,
            skip_save=True,
            seed=args.seed + round_index * 9973,
            recipe=args.recipe,
            recipe_phase_name="self_improve",
            recipe_phase_overrides={"learning_rate": args.learning_rate} if recipe_enabled(args.recipe) else None,
        )
        if hasattr(training_args, "auto_find_batch_size"):
            setattr(training_args, "auto_find_batch_size", bool(args.auto_find_batch_size))
        trainer = build_trainer(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            seed=args.seed + round_index * 9973,
            size_getter=partition_bucket_id,
            bucket_train_batches_by_size=bool(args.bucket_train_batches_by_partition),
            recipe=args.recipe,
            recipe_phase_name="self_improve",
        )
        train_result = trainer.train()
        model = trainer.model
        if args.save_model:
            trainer.save_model(str(round_dir / "model"))
            tokenizer.save_pretrained(round_dir / "model")

        eval_budget = max(
            resolve_max_new_tokens(train_examples, args.decode_max_new_tokens),
            resolve_max_new_tokens(seed_eval_splits["validation"], args.decode_max_new_tokens),
            resolve_max_new_tokens(seed_eval_splits["test"], args.decode_max_new_tokens),
            resolve_max_new_tokens(frontier_eval_splits["validation"], args.decode_max_new_tokens),
            resolve_max_new_tokens(frontier_eval_splits["test"], args.decode_max_new_tokens),
        )
        active_frontier_validation = filter_examples_by_max_b_digits(frontier_eval_splits["validation"], current_max_b_digits)
        active_frontier_test = filter_examples_by_max_b_digits(frontier_eval_splits["test"], current_max_b_digits)

        with tokenizer_padding_side(tokenizer, "left"):
            train_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=train_examples,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
            frontier_train_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=frontier_pseudo,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
            seed_validation_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=seed_eval_splits["validation"],
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
            seed_test_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=seed_eval_splits["test"],
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
            frontier_validation_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=active_frontier_validation,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )
            frontier_test_summary = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=active_frontier_test,
                batch_size=args.per_device_eval_batch_size,
                max_new_tokens=eval_budget,
            )

        round_payload = summarize_round(
            round_index=round_index,
            max_b_digits=current_max_b_digits,
            seed_replay_examples=seed_replay_pseudo,
            frontier_examples=frontier_pseudo,
            train_examples=train_examples,
            seed_replay_stats=seed_replay_stats,
            frontier_stats=frontier_stats,
            seed_validation_summary=seed_validation_summary,
            seed_test_summary=seed_test_summary,
            frontier_train_summary=frontier_train_summary,
            train_summary=train_summary,
            frontier_validation_summary=frontier_validation_summary,
            frontier_test_summary=frontier_test_summary,
        )
        round_payload["training"] = sanitize_json_value(train_result.metrics)
        round_payload["model_dir"] = str(round_dir / "model") if args.save_model else None
        round_records.append(round_payload)

        with (round_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(sanitize_json_value(round_payload), handle, indent=2)

        print(
            json.dumps(
                {
                    "round": round_index,
                    "max_b_digits": current_max_b_digits,
                    "seed_test_accuracy": seed_test_summary["accuracy"],
                    "frontier_test_accuracy": frontier_test_summary["accuracy"],
                    "frontier_train_accuracy": frontier_train_summary["accuracy"],
                    "train_examples": len(train_examples),
                }
            ),
            flush=True,
        )

        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_payload = {
        "task": "rectangular_multiplication_self_improvement",
        "seed_model": args.seed_model,
        "recipe": args.recipe,
        "format_version": args.format_version,
        "seed_partitions": [partition_label(partition) for partition in seed_partitions],
        "frontier_a_digits": [args.frontier_min_a_digits, args.frontier_max_a_digits],
        "frontier_b_digits": [args.frontier_min_b_digits, final_max_b_digits],
        "pseudo_label_mode": args.pseudo_label_mode,
        "rounds": sanitize_json_value(round_records),
    }
    with (output_dir / "self_improvement_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print(f"[INFO] Saved self-improvement results to {output_dir / 'self_improvement_results.json'}", flush=True)


if __name__ == "__main__":
    main()

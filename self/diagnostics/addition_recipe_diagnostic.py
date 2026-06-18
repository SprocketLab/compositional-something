#!/usr/bin/env python3
"""Diagnostic runner for the arithmetic-self-improve addition recipe."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import set_seed

from self.core.recipes import (
    apply_recipe_runtime_settings,
    build_recipe_tokenizer,
    instantiate_recipe_model,
    load_recipe_model,
    tokenizer_padding_side,
)
from self.core.recipes import (
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    RecipeTrainingPhaseConfig,
    resolve_addition_recipe,
    resolve_recipe_phase,
)
from self.core.recipes import (
    PaddingAwareCausalLMDataCollator,
    WarmupStableDecayTrainer,
    make_recipe_training_args,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples
from self.core.evaluation import evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.training import TokenizedPromptTargetDataset
from self.tasks.addition import AdditionTask


DEFAULT_FRONTIER_SOURCE_ROOT = (
    Path("artifacts/runs/self_improvement_addition_round1_fullpack_5000expand_20260418_193114")
    / "addition"
    / "with_carry_filtered"
)
DEFAULT_BUCKETED_REFERENCE = Path(
    "artifacts/runs/addition_wcf_digits8_9_buckettrain_from_seed_20260419_000909/summary.json"
)
DEFAULT_GOLD_REFERENCE = Path(
    "artifacts/runs/addition_wcf_frontier_overfit_compare_20260418_230437/gold/summary.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the arithmetic-self-improve addition recipe while preserving this repo's prompt format."
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--recipe",
        type=str,
        choices=(RECIPE_ARITHMETIC_SELF_IMPROVE_V1,),
        default=RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    )
    parser.add_argument("--device-target", type=str, default="local_a100_40gb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--seed-min-digits", type=int, default=3)
    parser.add_argument("--seed-max-digits", type=int, default=7)
    parser.add_argument("--seed-train-per-digit", type=int, default=50_000)
    parser.add_argument("--seed-heldout-per-digit", type=int, default=200)
    parser.add_argument("--seed-viability-threshold", type=float, default=0.95)

    parser.add_argument(
        "--frontier-source-root",
        type=str,
        default=str(DEFAULT_FRONTIER_SOURCE_ROOT),
        help="Existing with_carry_filtered artifact root that provides composed frontier/eval pools.",
    )
    parser.add_argument(
        "--bucketed-reference-summary",
        type=str,
        default=str(DEFAULT_BUCKETED_REFERENCE),
    )
    parser.add_argument(
        "--gold-reference-summary",
        type=str,
        default=str(DEFAULT_GOLD_REFERENCE),
    )
    parser.add_argument(
        "--frontier-train-limit-per-digit",
        type=int,
        default=None,
        help="Optional per-digit cap for smoke tests; by default the full frontier pool is used.",
    )

    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed-max-steps", type=int, default=None)
    parser.add_argument("--frontier-max-steps", type=int, default=None)
    parser.add_argument(
        "--auto-find-batch-size",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)


def min_accuracy(per_digit_accuracy: Dict[int, float]) -> Optional[float]:
    valid = [value for value in per_digit_accuracy.values() if value is not None and not math.isnan(value)]
    if not valid:
        return None
    return min(valid)


def filter_examples_by_digits(
    examples: Sequence[Any],
    min_digits: int,
    max_digits: int,
    *,
    per_digit_limit: Optional[int] = None,
) -> List[Any]:
    selected: List[Any] = []
    per_digit_counts: Dict[int, int] = {}
    for example in examples:
        digits = int(example.digits)
        if digits < min_digits or digits > max_digits:
            continue
        if per_digit_limit is not None and per_digit_counts.get(digits, 0) >= per_digit_limit:
            continue
        selected.append(example)
        per_digit_counts[digits] = per_digit_counts.get(digits, 0) + 1
    return selected


def filter_component_map_by_examples(
    task: AdditionTask,
    examples: Sequence[Any],
    component_map: Dict[Any, List[Any]],
) -> Dict[Any, List[Any]]:
    keys = {task.key_for_example(example) for example in examples}
    return {key: value for key, value in component_map.items() if key in keys}


def evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    task: AdditionTask,
    name: str,
    examples: Sequence[Any],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    accuracy, per_digit_accuracy = evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        size_getter=task.size_of,
        prediction_parser=task.prediction_parser,
    )
    return {
        "name": name,
        "count": len(examples),
        "accuracy": accuracy,
        "per_digit_accuracy": {str(digit): value for digit, value in sorted(per_digit_accuracy.items())},
        "min_per_digit_accuracy": min_accuracy(per_digit_accuracy),
    }


def save_stage_examples(
    *,
    stage_dir: Path,
    task: AdditionTask,
    train_examples: Sequence[Any],
    heldout_eval_examples: Sequence[Any],
    composed_eval_examples: Sequence[Any],
) -> None:
    data_dir = stage_dir / "data"
    save_examples(data_dir / "train_examples.jsonl", train_examples, task.serialize_example)
    save_examples(data_dir / "heldout_eval_examples.jsonl", heldout_eval_examples, task.serialize_example)
    save_examples(data_dir / "heldout_composed_eval_examples.jsonl", composed_eval_examples, task.serialize_example)


def train_model_for_stage(
    *,
    stage_dir: Path,
    task: AdditionTask,
    train_examples: Sequence[Any],
    heldout_eval_examples: Sequence[Any],
    composed_eval_examples: Sequence[Any],
    composed_eval_component_map: Dict[Any, List[Any]],
    load_from: Optional[Path],
    recipe_name: str,
    recipe_phase_name: str,
    recipe_train_batch_size: int,
    recipe_eval_batch_size: int,
    recipe_max_steps: int,
    auto_find_batch_size: bool,
    seed: int,
) -> Dict[str, Any]:
    preset = resolve_addition_recipe(recipe_name)
    phase = resolve_recipe_phase(preset, recipe_phase_name)
    bf16 = bool(preset.bf16 and torch.cuda.is_available())
    fp16 = False
    tokenizer = build_recipe_tokenizer(preset)
    if load_from is None:
        model = instantiate_recipe_model(tokenizer, preset, bf16=bf16, fp16=fp16)
    else:
        model = load_recipe_model(load_from, tokenizer, bf16=bf16, fp16=fp16)

    train_dataset = TokenizedPromptTargetDataset(train_examples, tokenizer)
    trainer = WarmupStableDecayTrainer(
        model=model,
        args=make_recipe_training_args(
            output_dir=stage_dir / "trainer",
            preset=preset,
            phase=phase,
            phase_overrides=None,
            seed=seed,
            bf16=bf16,
            fp16=fp16,
            per_device_train_batch_size=recipe_train_batch_size,
            per_device_eval_batch_size=recipe_eval_batch_size,
            max_steps=recipe_max_steps,
            auto_find_batch_size=auto_find_batch_size,
        ),
        train_dataset=train_dataset,
        data_collator=PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right"),
        num_stable_steps=phase.num_stable_steps,
        num_decay_steps=phase.num_decay_steps,
        min_lr_ratio=preset.min_lr_ratio,
    )
    train_result = trainer.train()
    trainer.save_model(str(stage_dir / "model"))
    tokenizer.save_pretrained(stage_dir / "model")

    eval_budget = max(
        resolve_max_new_tokens(train_examples, preset.decode_max_new_tokens),
        resolve_max_new_tokens(heldout_eval_examples, preset.decode_max_new_tokens),
        resolve_max_new_tokens(composed_eval_examples, preset.decode_max_new_tokens),
    )

    with tokenizer_padding_side(tokenizer, "left"):
        train_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="train",
            examples=train_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        heldout_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="heldout_eval",
            examples=heldout_eval_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        composed_all_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="composed_eval_all",
            examples=composed_eval_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        composed_slices = task.split_composed_eval_slices(composed_eval_examples, composed_eval_component_map)
        composed_boundary_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="composed_eval_boundary",
            examples=composed_slices.get("boundary_carry", []),
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        composed_no_boundary_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="composed_eval_no_boundary",
            examples=composed_slices.get("no_boundary_carry", []),
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        composed_unknown_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="composed_eval_unknown",
            examples=composed_slices.get("unknown", []),
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )

    summary = {
        "recipe": recipe_name,
        "training": train_result.metrics,
        "train": train_summary,
        "heldout_eval": heldout_summary,
        "composed_eval_all": composed_all_summary,
        "composed_eval_boundary": composed_boundary_summary,
        "composed_eval_no_boundary": composed_no_boundary_summary,
        "composed_eval_unknown": composed_unknown_summary,
        "model_dir": str(stage_dir / "model"),
    }
    write_json(stage_dir / "summary.json", summary)
    return summary


def summarize_seed_stage(
    *,
    stage_dir: Path,
    task: AdditionTask,
    train_examples: Sequence[Any],
    validation_examples: Sequence[Any],
    test_examples: Sequence[Any],
    recipe_name: str,
    recipe_train_batch_size: int,
    recipe_eval_batch_size: int,
    recipe_max_steps: int,
    auto_find_batch_size: bool,
    seed: int,
) -> Dict[str, Any]:
    preset = resolve_addition_recipe(recipe_name)
    phase = resolve_recipe_phase(preset, "seed")
    bf16 = bool(preset.bf16 and torch.cuda.is_available())
    fp16 = False
    tokenizer = build_recipe_tokenizer(preset)
    model = instantiate_recipe_model(tokenizer, preset, bf16=bf16, fp16=fp16)

    trainer = WarmupStableDecayTrainer(
        model=model,
        args=make_recipe_training_args(
            output_dir=stage_dir / "trainer",
            preset=preset,
            phase=phase,
            phase_overrides=None,
            seed=seed,
            bf16=bf16,
            fp16=fp16,
            per_device_train_batch_size=recipe_train_batch_size,
            per_device_eval_batch_size=recipe_eval_batch_size,
            max_steps=recipe_max_steps,
            auto_find_batch_size=auto_find_batch_size,
        ),
        train_dataset=TokenizedPromptTargetDataset(train_examples, tokenizer),
        data_collator=PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right"),
        num_stable_steps=phase.num_stable_steps,
        num_decay_steps=phase.num_decay_steps,
        min_lr_ratio=preset.min_lr_ratio,
    )
    train_result = trainer.train()
    trainer.save_model(str(stage_dir / "model"))
    tokenizer.save_pretrained(stage_dir / "model")

    eval_budget = max(
        resolve_max_new_tokens(train_examples, preset.decode_max_new_tokens),
        resolve_max_new_tokens(validation_examples, preset.decode_max_new_tokens),
        resolve_max_new_tokens(test_examples, preset.decode_max_new_tokens),
    )
    with tokenizer_padding_side(tokenizer, "left"):
        train_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="train",
            examples=train_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        validation_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="validation",
            examples=validation_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )
        test_summary = evaluate_split(
            model=trainer.model,
            tokenizer=tokenizer,
            task=task,
            name="test",
            examples=test_examples,
            batch_size=recipe_eval_batch_size,
            max_new_tokens=eval_budget,
        )

    worst_case = min(
        value
        for value in (
            validation_summary["min_per_digit_accuracy"],
            test_summary["min_per_digit_accuracy"],
        )
        if value is not None
    )
    summary = {
        "recipe": recipe_name,
        "training": train_result.metrics,
        "results": {
            "train": train_summary,
            "validation": validation_summary,
            "test": test_summary,
        },
        "validation_min_per_digit_accuracy": validation_summary["min_per_digit_accuracy"],
        "test_min_per_digit_accuracy": test_summary["min_per_digit_accuracy"],
        "worst_case_heldout_min_per_digit_accuracy": worst_case,
        "model_dir": str(stage_dir / "model"),
    }
    write_json(stage_dir / "summary.json", summary)
    return summary


def load_frontier_source(
    task: AdditionTask,
    source_root: Path,
    *,
    min_digits: int,
    max_digits: int,
    train_limit_per_digit: Optional[int],
) -> Tuple[List[Any], List[Any], List[Any], Dict[Any, List[Any]]]:
    data_root = source_root / "data"
    required_paths = [
        data_root / "composed_pool.jsonl",
        data_root / "evaluation.jsonl",
        data_root / "composed_evaluation.jsonl",
        data_root / "composed_evaluation_component_map.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Frontier source root is missing required files:\n" + "\n".join(missing)
        )
    train_examples = filter_examples_by_digits(
        load_examples_from_jsonl(task, data_root / "composed_pool.jsonl"),
        min_digits,
        max_digits,
        per_digit_limit=train_limit_per_digit,
    )
    heldout_eval_examples = filter_examples_by_digits(
        load_examples_from_jsonl(task, data_root / "evaluation.jsonl"),
        min_digits,
        max_digits,
    )
    composed_eval_examples = filter_examples_by_digits(
        load_examples_from_jsonl(task, data_root / "composed_evaluation.jsonl"),
        min_digits,
        max_digits,
    )
    composed_eval_component_map = filter_component_map_by_examples(
        task,
        composed_eval_examples,
        task.load_component_map(data_root / "composed_evaluation_component_map.json"),
    )
    return train_examples, heldout_eval_examples, composed_eval_examples, composed_eval_component_map


def load_examples_from_jsonl(task: AdditionTask, path: Path) -> List[Any]:
    examples: List[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(task.deserialize_example(json.loads(line)))
    return examples


def build_stage_plan(args: argparse.Namespace, recipe_name: str) -> Dict[str, Any]:
    return {
        "recipe": recipe_name,
        "device_target": args.device_target,
        "seed_stage": {
            "digits": [args.seed_min_digits, args.seed_max_digits],
            "train_per_digit": args.seed_train_per_digit,
            "heldout_per_digit": args.seed_heldout_per_digit,
            "max_steps": args.seed_max_steps,
        },
        "frontier_8_9": {
            "digits": [8, 9],
            "source_root": args.frontier_source_root,
            "train_limit_per_digit": args.frontier_train_limit_per_digit,
            "max_steps": args.frontier_max_steps,
        },
        "frontier_8_12": {
            "digits": [8, 12],
            "source_root": args.frontier_source_root,
            "train_limit_per_digit": args.frontier_train_limit_per_digit,
            "max_steps": args.frontier_max_steps,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    recipe = resolve_addition_recipe(args.recipe)
    apply_recipe_runtime_settings(recipe)

    recipe_train_batch_size = args.per_device_train_batch_size or recipe.per_device_train_batch_size
    recipe_eval_batch_size = args.per_device_eval_batch_size or recipe.per_device_eval_batch_size
    args.seed_max_steps = args.seed_max_steps or args.max_steps or recipe.seed_phase.max_steps
    args.frontier_max_steps = args.frontier_max_steps or args.max_steps or recipe.self_improve_phase.max_steps
    if args.auto_find_batch_size is None:
        args.auto_find_batch_size = recipe.auto_find_batch_size

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    write_json(output_dir / "config_args.json", vars(args))

    if args.dry_run:
        plan = build_stage_plan(args, recipe.name)
        write_json(output_dir / "dry_run_plan.json", plan)
        print(json.dumps(plan, indent=2))
        return

    set_seed(args.seed)
    rng = random.Random(args.seed)
    task = AdditionTask()

    seed_stage_dir = output_dir / "seed"
    seed_data_dir = seed_stage_dir / "data"
    ensure_dir(seed_data_dir)
    seed_args = argparse.Namespace(
        initial_min_size=args.seed_min_digits,
        initial_max_size=args.seed_max_digits,
        initial_train_per_size=args.seed_train_per_digit,
        initial_eval_per_size=args.seed_heldout_per_digit,
        composition_error_percent=0.0,
        corruption_rate=0.0,
    )
    seed_splits, _ = task.prepare_initial_splits(rng, seed_args)
    save_examples(seed_data_dir / "initial_train.jsonl", seed_splits["train"], task.serialize_example)
    save_examples(seed_data_dir / "initial_validation.jsonl", seed_splits["validation"], task.serialize_example)
    save_examples(seed_data_dir / "initial_test.jsonl", seed_splits["test"], task.serialize_example)

    print(
        "[INFO] Seed stage: digits {}..{} train_per_digit={} heldout_per_digit={}".format(
            args.seed_min_digits,
            args.seed_max_digits,
            args.seed_train_per_digit,
            args.seed_heldout_per_digit,
        ),
        flush=True,
    )
    seed_summary = summarize_seed_stage(
        stage_dir=seed_stage_dir,
        task=task,
        train_examples=seed_splits["train"],
        validation_examples=seed_splits["validation"],
        test_examples=seed_splits["test"],
        recipe_name=recipe.name,
        recipe_train_batch_size=recipe_train_batch_size,
        recipe_eval_batch_size=recipe_eval_batch_size,
        recipe_max_steps=args.seed_max_steps,
        auto_find_batch_size=bool(args.auto_find_batch_size),
        seed=args.seed,
    )

    worst_case = seed_summary["worst_case_heldout_min_per_digit_accuracy"]
    seed_viable = worst_case is not None and worst_case >= args.seed_viability_threshold
    if not seed_viable:
        overall_summary = {
            "recipe": recipe.name,
            "seed": seed_summary,
            "seed_viable": False,
            "seed_viability_threshold": args.seed_viability_threshold,
            "stop_reason": "seed_non_viable",
        }
        write_json(output_dir / "summary.json", overall_summary)
        raise SystemExit(
            "Seed stage did not meet viability threshold: "
            f"worst_case={worst_case} threshold={args.seed_viability_threshold}"
        )

    source_root = Path(args.frontier_source_root)
    frontier_stage_specs = [
        ("frontier_8_9", 8, 9, Path(args.bucketed_reference_summary)),
        ("frontier_8_12", 8, 12, Path(args.gold_reference_summary)),
    ]
    frontier_results: Dict[str, Any] = {}
    comparison_payload: Dict[str, Any] = {}

    for stage_name, min_digits, max_digits, reference_summary_path in frontier_stage_specs:
        stage_dir = output_dir / stage_name
        print(f"[INFO] {stage_name}: digits {min_digits}..{max_digits} from {source_root}", flush=True)
        train_examples, heldout_eval_examples, composed_eval_examples, composed_eval_component_map = load_frontier_source(
            task,
            source_root,
            min_digits=min_digits,
            max_digits=max_digits,
            train_limit_per_digit=args.frontier_train_limit_per_digit,
        )
        save_stage_examples(
            stage_dir=stage_dir,
            task=task,
            train_examples=train_examples,
            heldout_eval_examples=heldout_eval_examples,
            composed_eval_examples=composed_eval_examples,
        )
        task.save_component_map(stage_dir / "data" / "heldout_composed_eval_component_map.json", composed_eval_component_map)

        stage_summary = train_model_for_stage(
            stage_dir=stage_dir,
            task=task,
            train_examples=train_examples,
            heldout_eval_examples=heldout_eval_examples,
            composed_eval_examples=composed_eval_examples,
            composed_eval_component_map=composed_eval_component_map,
            load_from=seed_stage_dir / "model",
            recipe_name=recipe.name,
            recipe_phase_name="frontier",
            recipe_train_batch_size=recipe_train_batch_size,
            recipe_eval_batch_size=recipe_eval_batch_size,
            recipe_max_steps=args.frontier_max_steps,
            auto_find_batch_size=bool(args.auto_find_batch_size),
            seed=args.seed,
        )
        frontier_results[stage_name] = stage_summary
        comparison_payload[stage_name] = {
            "current_prompt_gpt2_reference": load_json(reference_summary_path),
            "their_recipe_current_prompt": stage_summary,
        }

    write_json(output_dir / "comparison.json", comparison_payload)
    overall_summary = {
        "recipe": recipe.name,
        "device_target": args.device_target,
        "seed_viable": True,
        "seed_viability_threshold": args.seed_viability_threshold,
        "seed": seed_summary,
        "frontier_8_9": frontier_results["frontier_8_9"],
        "frontier_8_12": frontier_results["frontier_8_12"],
        "comparison_file": str(output_dir / "comparison.json"),
    }
    write_json(output_dir / "summary.json", overall_summary)


if __name__ == "__main__":
    main()

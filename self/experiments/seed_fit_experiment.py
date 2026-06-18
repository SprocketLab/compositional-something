#!/usr/bin/env python3
"""Train on the seed dataset only and measure held-out seed accuracy."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import set_seed

from self.core.recipes import (
    RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
    recipe_enabled,
    resolve_self_improvement_recipe,
)
from self.core.recipes import PaddingAwareCausalLMDataCollator
from core.addition_pipeline import (
    ADDITION_SAMPLING_MODES,
    ADDITION_SAMPLING_NATURAL,
    ADDITION_WIDTH_EXACT_DIGITS,
    ADDITION_WIDTH_MODES,
    COMPOSITION_PATH_MODES,
    COMPOSITION_PATH_RANDOM,
)
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples
from self.core.evaluation import evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import (
    CausalLMDataCollator,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    build_trainer,
    make_training_args,
)
from self.tasks.addition import AdditionTask
from self.tasks.multiplication import MultiplicationTask
from self.tasks.run_length import RunLengthTask


TASK_REGISTRY = {
    "addition": AdditionTask,
    "run_length": RunLengthTask,
    "multiplication": MultiplicationTask,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed-only fit experiment for compositional self-improvement tasks.")

    parser.add_argument("--task", type=str, choices=sorted(TASK_REGISTRY), required=True)
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--format-version",
        type=str,
        default="legacy",
        help="Prompt/target serialization format for tasks that support alternate symbolic formats.",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        choices=("default", "plain_output", "symbol_run_pair", "run_state"),
        default="default",
        help="Use the task's default structured target or a plain-output diagnostic target.",
    )
    parser.add_argument(
        "--compose-arity",
        type=str,
        choices=("at_least2", "exact2"),
        default="at_least2",
        help="Composition arity setting forwarded to task adapters that use composed examples.",
    )
    parser.add_argument(
        "--bit-composition-path-mode",
        type=str,
        choices=("random", "fixed_binary"),
        default="random",
        help="Run-length bit-string composition path mode forwarded to task adapters that use composed examples.",
    )
    parser.add_argument(
        "--guarded-compose-rule",
        type=str,
        choices=("none", "run_length_no_boundary_continue", "run_length_unfiltered_pair"),
        default="none",
        help="Optional guarded composition rule forwarded to task adapters.",
    )
    parser.add_argument(
        "--symbol-alphabet-size",
        type=int,
        default=2,
        help=(
            "Number of input symbols for run_length diagnostics. "
            "The prompt alphabet becomes 0..k-1 while the target is the longest run of any repeated symbol."
        ),
    )

    parser.add_argument("--initial-min-size", type=int, required=True)
    parser.add_argument("--initial-max-size", type=int, required=True)
    parser.add_argument("--initial-train-per-size", type=int, required=True)
    parser.add_argument(
        "--initial-eval-per-size",
        type=int,
        default=100,
        help="Per-size held-out count for validation and test splits in the seed range.",
    )
    parser.add_argument(
        "--target-accuracy-threshold",
        type=float,
        default=0.95,
        help="Selection target for worst-case held-out seed accuracy.",
    )

    parser.add_argument("--expand-num-size", type=int, default=1)
    parser.add_argument("--expand-train-per-size", type=int, default=0)
    parser.add_argument("--eval-per-size", type=int, default=0)
    parser.add_argument("--composed-eval-per-size", type=int, default=0)
    parser.add_argument("--num-expand-rounds", type=int, default=0)
    parser.add_argument("--pseudo-label-mode", type=str, default="none")
    parser.add_argument("--corruption-rate", type=float, default=0.0)
    parser.add_argument("--composed-strategy", type=str, default="with_carry")
    parser.add_argument("--composition-error-percent", type=float, default=0.0)
    parser.add_argument(
        "--addition-width-mode",
        type=str,
        choices=ADDITION_WIDTH_MODES,
        default=ADDITION_WIDTH_EXACT_DIGITS,
        help="Addition-only operand width mode.",
    )
    parser.add_argument(
        "--addition-sampling-mode",
        type=str,
        choices=ADDITION_SAMPLING_MODES,
        default=ADDITION_SAMPLING_NATURAL,
        help="Addition-only seed/eval sampling mode.",
    )
    parser.add_argument(
        "--addition-composition-path-mode",
        type=str,
        choices=COMPOSITION_PATH_MODES,
        default=COMPOSITION_PATH_RANDOM,
        help="Addition-only composition path mode.",
    )

    parser.add_argument("--block-size", type=int, default=2)

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=960)
    parser.add_argument("--decode-max-new-tokens", type=int, default=16)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--init-from-scratch",
        action="store_true",
        help="Initialize the model from config.json instead of loading pretrained weights.",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        choices=("auto", "fixed_char"),
        default="auto",
        help="Tokenizer mode for scratch models.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        choices=("none", RECIPE_ALGORITHMIC_SELF_IMPROVE_V1, "arithmetic_self_improve_v1"),
        default="none",
        help="Optional recipe model/training preset for scratch or checkpointed seed-fit runs.",
    )
    parser.add_argument(
        "--bucket-train-batches-by-size",
        action="store_true",
        help="Bucket training batches so every batch contains examples from a single exact size.",
    )
    parser.add_argument(
        "--evaluate-train",
        action="store_true",
        help="Also evaluate accuracy on the seed training split.",
    )
    parser.add_argument(
        "--reserve-heldout-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve validation/test examples before train examples in finite seed universes.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Persist the trained model in the output directory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def get_task(task_name: str) -> Any:
    try:
        return TASK_REGISTRY[task_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported task: {task_name!r}") from exc


def min_accuracy(per_size_accuracy: Dict[int, float]) -> Optional[float]:
    valid = [score for score in per_size_accuracy.values() if score is not None and not math.isnan(score)]
    if not valid:
        return None
    return min(valid)


def evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    task: Any,
    split_name: str,
    examples: Sequence[Any],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    accuracy, per_size_accuracy = evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        size_getter=task.size_of,
        prediction_parser=task.prediction_parser,
    )
    return {
        "split": split_name,
        "count": len(examples),
        "accuracy": accuracy,
        "per_size_accuracy": {str(size): score for size, score in sorted(per_size_accuracy.items())},
        "min_per_size_accuracy": min_accuracy(per_size_accuracy),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] No precision flag provided; defaulting to bf16 on CUDA.", flush=True)
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of bf16 or fp16.")

    task = get_task(args.task)
    task.validate_args(args)
    if recipe_enabled(args.recipe) and args.task == "multiplication":
        raise ValueError("Recipe-backed seed-fit is only supported for addition and run_length.")

    recipe_preset = resolve_self_improvement_recipe(args.recipe) if recipe_enabled(args.recipe) else None
    if recipe_enabled(args.recipe) and args.tokenizer_mode != "auto":
        print("[INFO] Recipe-backed seed-fit ignores --tokenizer-mode and uses the recipe tokenizer.", flush=True)
    if recipe_preset is not None:
        if args.per_device_train_batch_size == 4:
            args.per_device_train_batch_size = recipe_preset.per_device_train_batch_size
        if args.per_device_eval_batch_size == 8:
            args.per_device_eval_batch_size = recipe_preset.per_device_eval_batch_size
        if args.max_steps == 960:
            args.max_steps = recipe_preset.seed_phase.max_steps

    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    ensure_dir(data_dir)

    with (output_dir / "config_args.json").open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(vars(args)), handle, indent=2)

    set_seed(args.seed)
    rng = random.Random(args.seed)
    splits, records = task.prepare_initial_splits(rng, args)

    save_examples(data_dir / "initial_train.jsonl", splits["train"], task.serialize_example)
    save_examples(data_dir / "initial_validation.jsonl", splits["validation"], task.serialize_example)
    save_examples(data_dir / "initial_test.jsonl", splits["test"], task.serialize_example)

    print(
        "[INFO] Seed dataset sizes -- train: {} | validation: {} | test: {}".format(
            len(splits["train"]),
            len(splits["validation"]),
            len(splits["test"]),
        ),
        flush=True,
    )

    token_initializers = task.token_initializers(args) if hasattr(task, "token_initializers") else {}
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        token_initializers=token_initializers,
        init_from_scratch=args.init_from_scratch,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
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
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )

    train_dataset = TokenizedPromptTargetDataset(splits["train"], tokenizer)
    training_args = make_training_args(
        output_dir,
        config,
        bf16=args.bf16,
        fp16=args.fp16,
        # Seed-fit runs only need the final explicit save below; skipping
        # trainer-managed checkpoints avoids unnecessary epoch checkpointing.
        skip_save=True,
        keep_checkpoints=False,
        seed=args.seed,
        recipe=args.recipe,
        recipe_phase_name="seed",
    )
    data_collator = (
        PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")
        if recipe_enabled(args.recipe)
        else CausalLMDataCollator(tokenizer)
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        seed=args.seed,
        size_getter=task.size_of,
        bucket_train_batches_by_size=bool(args.bucket_train_batches_by_size),
        recipe=args.recipe,
        recipe_phase_name="seed",
    )
    train_result = trainer.train()
    model = trainer.model

    if args.save_model:
        trainer.save_model(str(output_dir / "model"))
        tokenizer.save_pretrained(output_dir / "model")

    decode_max_new_tokens = max(
        resolve_max_new_tokens(splits["train"], config.decode_max_new_tokens),
        resolve_max_new_tokens(splits["validation"], config.decode_max_new_tokens),
        resolve_max_new_tokens(splits["test"], config.decode_max_new_tokens),
    )

    split_results: Dict[str, Dict[str, Any]] = {}
    if args.evaluate_train:
        split_results["train"] = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            task=task,
            split_name="train",
            examples=splits["train"],
            batch_size=config.per_device_eval_batch_size,
            max_new_tokens=decode_max_new_tokens,
        )
    split_results["validation"] = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        task=task,
        split_name="validation",
        examples=splits["validation"],
        batch_size=config.per_device_eval_batch_size,
        max_new_tokens=decode_max_new_tokens,
    )
    split_results["test"] = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        task=task,
        split_name="test",
        examples=splits["test"],
        batch_size=config.per_device_eval_batch_size,
        max_new_tokens=decode_max_new_tokens,
    )

    validation_min = split_results["validation"]["min_per_size_accuracy"]
    test_min = split_results["test"]["min_per_size_accuracy"]
    meets_threshold = (
        validation_min is not None
        and test_min is not None
        and validation_min >= args.target_accuracy_threshold
        and test_min >= args.target_accuracy_threshold
    )

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    train_examples = len(splits["train"])

    payload = {
        "task": args.task,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "seed": args.seed,
        "target_accuracy_threshold": args.target_accuracy_threshold,
        "initial_min_size": args.initial_min_size,
        "initial_max_size": args.initial_max_size,
        "initial_train_per_size": args.initial_train_per_size,
        "initial_eval_per_size": args.initial_eval_per_size,
        "train_examples": train_examples,
        "validation_examples": len(splits["validation"]),
        "test_examples": len(splits["test"]),
        "training": {
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "train_result_metrics": sanitize_json_value(train_result.metrics),
            "log_history": sanitize_json_value(trainer.state.log_history),
            "final_epoch": sanitize_json_value(float(trainer.state.epoch) if trainer.state.epoch is not None else None),
            "approx_effective_epochs_from_steps": (
                (args.max_steps * effective_batch_size / train_examples)
                if args.max_steps and train_examples > 0
                else None
            ),
        },
        "results": split_results,
        "validation_min_per_size_accuracy": validation_min,
        "test_min_per_size_accuracy": test_min,
        "meets_threshold": meets_threshold,
        "task_metadata": {
            "records": {split: len(record_set) for split, record_set in records.items()},
        },
    }

    results_path = output_dir / "seed_fit_results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)

    summary = {
        "task": args.task,
        "train_examples": train_examples,
        "validation_min_per_size_accuracy": validation_min,
        "test_min_per_size_accuracy": test_min,
        "meets_threshold": meets_threshold,
    }
    print(json.dumps(sanitize_json_value(summary)), flush=True)
    print(f"[INFO] Saved seed-fit results to {results_path}", flush=True)


if __name__ == "__main__":
    main()

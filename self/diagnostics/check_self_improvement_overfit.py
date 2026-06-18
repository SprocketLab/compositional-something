#!/usr/bin/env python3
"""Overfit sanity checks for self-improvement task supervision/evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import Trainer, set_seed

from self.legacy.addition_self_improvement import normalize_args as normalize_addition_args
from self.legacy.addition_self_improvement import parse_args as parse_addition_args
from self.legacy.multiplication_self_improvement import normalize_args as normalize_multiplication_args
from self.legacy.multiplication_self_improvement import parse_args as parse_multiplication_args
from self.legacy.run_length_self_improvement import normalize_args as normalize_run_length_args
from self.legacy.run_length_self_improvement import parse_args as parse_run_length_args
from self.core.evaluation import (
    build_generation_encodings,
    evaluate_accuracy_with_breakdown,
    parse_prediction,
    resolve_max_new_tokens,
)
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import (
    CausalLMDataCollator,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    make_training_args,
)
from self.tasks.addition import AdditionTask
from self.tasks.multiplication import MultiplicationTask
from self.tasks.run_length import RunLengthTask


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit 10-sample sanity checks for self-improvement tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("addition", "run_length", "multiplication"),
        default=("addition", "run_length", "multiplication"),
        help="Tasks to run.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        choices=("base", "composed"),
        default=("base", "composed"),
        help="Whether to overfit base or composed examples.",
    )
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument(
        "--format-version",
        type=str,
        default="legacy",
        help="Prompt/target serialization format for run_length/multiplication.",
    )
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="artifacts/overfit_sanity")
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
    parser.add_argument("--run-length-min-bits", type=int, default=4)
    parser.add_argument("--run-length-max-bits", type=int, default=8)
    parser.add_argument("--multiplication-min-digits", type=int, default=2)
    parser.add_argument("--multiplication-max-digits", type=int, default=3)
    return parser.parse_args(argv)


def build_task_bundle(
    task_name: str,
    model_name: str,
    seed: int,
    format_version: str,
    seed_args: argparse.Namespace,
) -> Tuple[Any, Any]:
    run_length_min_bits = getattr(seed_args, "run_length_min_bits", 4)
    run_length_max_bits = getattr(seed_args, "run_length_max_bits", 8)
    multiplication_min_digits = getattr(seed_args, "multiplication_min_digits", 2)
    multiplication_max_digits = getattr(seed_args, "multiplication_max_digits", 2)
    if task_name == "addition":
        args = normalize_addition_args(
            parse_addition_args(
                [
                    "--model-name",
                    model_name,
                    "--output-dir",
                    "artifacts/tmp_overfit_addition",
                    "--initial-min-digits",
                    "2",
                    "--initial-max-digits",
                    "4",
                    "--initial-train-per-digit",
                    "20",
                    "--initial-eval-per-digit",
                    "0",
                    "--num-expand-rounds",
                    "1",
                    "--expand-num-digits",
                    "2",
                    "--expand-train-per-digit",
                    "20",
                    "--eval-per-digit",
                    "0",
                    "--composed-eval-per-digit",
                    "0",
                    "--pseudo-label-mode",
                    "compose",
                    "--seed",
                    str(seed),
                ]
            )
        )
        return AdditionTask(), args
    if task_name == "run_length":
        args = normalize_run_length_args(
            parse_run_length_args(
                [
                    "--model-name",
                    model_name,
                    "--format-version",
                    format_version,
                    "--output-dir",
                    "artifacts/tmp_overfit_run_length",
                    "--initial-min-bits",
                    str(run_length_min_bits),
                    "--initial-max-bits",
                    str(run_length_max_bits),
                    "--initial-train-per-bit",
                    "20",
                    "--initial-eval-per-bit",
                    "0",
                    "--num-expand-rounds",
                    "1",
                    "--expand-num-bits",
                    "2",
                    "--expand-train-per-bit",
                    "20",
                    "--eval-per-bit",
                    "0",
                    "--composed-eval-per-bit",
                    "0",
                    "--pseudo-label-mode",
                    "compose",
                    "--seed",
                    str(seed),
                ]
            )
        )
        return RunLengthTask(), args
    if task_name == "multiplication":
        args = normalize_multiplication_args(
            parse_multiplication_args(
                [
                    "--model-name",
                    model_name,
                    "--format-version",
                    format_version,
                    "--output-dir",
                    "artifacts/tmp_overfit_multiplication",
                    "--block-size",
                    "2",
                    "--initial-min-digits",
                    str(multiplication_min_digits),
                    "--initial-max-digits",
                    str(multiplication_max_digits),
                    "--initial-train-per-digit",
                    "20",
                    "--initial-eval-per-digit",
                    "0",
                    "--num-expand-rounds",
                    "1",
                    "--expand-num-digits",
                    "2",
                    "--expand-train-per-digit",
                    "20",
                    "--eval-per-digit",
                    "0",
                    "--composed-eval-per-digit",
                    "0",
                    "--pseudo-label-mode",
                    "compose",
                    "--seed",
                    str(seed),
                ]
            )
        )
        return MultiplicationTask(), args
    raise ValueError(f"Unsupported task: {task_name}")


def prepare_examples(task_name: str, task: Any, args: Any, seed: int, setting: str, num_examples: int) -> List[Any]:
    rng = random.Random(seed)
    base_splits, base_records = task.prepare_initial_splits(rng, args)
    if setting == "base":
        pool = list(base_splits["train"])
    elif setting == "composed":
        composed_min_size = args.initial_max_size + 1
        composed_max_size = args.initial_max_size + args.expand_num_size
        pool, _, _ = task.prepare_composed_train(
            rng,
            args,
            base_splits=base_splits,
            base_records=base_records,
            min_size=composed_min_size,
            max_size=composed_max_size,
        )
    else:
        raise ValueError(f"Unsupported setting: {setting}")

    if len(pool) < num_examples:
        raise RuntimeError(
            f"{task_name}/{setting} only produced {len(pool)} examples, but {num_examples} were requested."
        )
    return list(pool[:num_examples])


def overfit_examples(
    *,
    task_name: str,
    setting: str,
    task: Any,
    task_args: Any,
    model_name: str,
    examples: Sequence[Any],
    base_output_dir: Path,
    max_steps: int,
    num_epochs: int,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    seed: int,
    bf16: bool,
    fp16: bool,
    init_from_scratch: bool,
    tokenizer_mode: str,
) -> Dict[str, Any]:
    run_dir = base_output_dir / task_name / setting
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    token_initializers = task.token_initializers(task_args) if hasattr(task, "token_initializers") else {}
    model, tokenizer = instantiate_model_and_tokenizer(
        model_name,
        bf16=bf16,
        fp16=fp16,
        token_initializers=token_initializers,
        init_from_scratch=init_from_scratch,
        tokenizer_mode=tokenizer_mode,
    )
    config = TrainingConfig(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=0.0,
        logging_steps=logging_steps,
        max_steps=max_steps,
        decode_max_new_tokens=resolve_max_new_tokens(examples, 16),
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(run_dir, config, bf16=bf16, fp16=fp16, skip_save=True, seed=seed),
        train_dataset=TokenizedPromptTargetDataset(examples, tokenizer),
        eval_dataset=None,
        data_collator=CausalLMDataCollator(tokenizer),
    )
    trainer.train()
    model = trainer.model

    decode_tokens = resolve_max_new_tokens(examples, config.decode_max_new_tokens)
    eval_accuracy, per_size_accuracy = evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=per_device_eval_batch_size,
        max_new_tokens=decode_tokens,
        size_getter=task.size_of,
        prediction_parser=task.prediction_parser,
    )

    raw_prediction_map: Dict[Any, str] = {}
    parsed_prediction_map: Dict[Any, str] = {}
    device = next(model.parameters()).device
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), per_device_eval_batch_size):
            batch = list(examples[start : start + per_device_eval_batch_size])
            prompts = [example.prompt() for example in batch]
            encodings = build_generation_encodings(tokenizer, prompts, device)
            output_ids = model.generate(
                **encodings,
                max_new_tokens=decode_tokens,
                do_sample=False,
            )
            prompt_width = encodings["input_ids"].shape[1]
            for idx, example in enumerate(batch):
                key = task.key_for_example(example)
                generated_slice = output_ids[idx, prompt_width:].tolist()
                raw_text = tokenizer.decode(generated_slice, skip_special_tokens=True)
                raw_prediction_map[key] = raw_text
                parsed = parse_prediction(task.prediction_parser, raw_text, example)
                if parsed is not None:
                    parsed_prediction_map[key] = parsed.strip()
    if model_was_training:
        model.train()

    decoded_examples = [
        {
            "prompt": example.prompt(),
            "target": example.target(),
            "raw_prediction": raw_prediction_map.get(task.key_for_example(example)),
            "prediction": parsed_prediction_map.get(task.key_for_example(example)),
        }
        for example in examples
    ]
    result = {
        "task": task_name,
        "setting": setting,
        "num_examples": len(examples),
        "eval_accuracy": eval_accuracy,
        "per_size_accuracy": per_size_accuracy,
        "examples": decoded_examples,
    }
    with (run_dir / "overfit_result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for task_name in args.tasks:
        task, task_args = build_task_bundle(task_name, args.model_name, args.seed, args.format_version, args)
        for setting in args.settings:
            examples = prepare_examples(task_name, task, task_args, args.seed, setting, args.num_examples)
            result = overfit_examples(
                task_name=task_name,
                setting=setting,
                task=task,
                task_args=task_args,
                model_name=args.model_name,
                examples=examples,
                base_output_dir=output_dir,
                max_steps=args.max_steps,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                per_device_train_batch_size=args.per_device_train_batch_size,
                per_device_eval_batch_size=args.per_device_eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                logging_steps=args.logging_steps,
                seed=args.seed,
                bf16=args.bf16,
                fp16=args.fp16,
                init_from_scratch=args.init_from_scratch,
                tokenizer_mode=args.tokenizer_mode,
            )
            results.append(result)
            print(
                f"[OVERFIT] task={task_name} setting={setting} "
                f"examples={len(examples)} eval_acc={result['eval_accuracy']:.4f}",
                flush=True,
            )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

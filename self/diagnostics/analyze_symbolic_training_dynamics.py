#!/usr/bin/env python3
"""Diagnose training-loss and exact-match dynamics for workshop self-improvement tasks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Trainer, TrainerCallback

from self.core.evaluation import evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import (
    CausalLMDataCollator,
    TokenizedPromptTargetDataset,
    TrainingConfig,
    make_training_args,
)
from self.diagnostics.check_self_improvement_overfit import build_task_bundle, prepare_examples


@dataclass
class AccuracyRecord:
    step: int
    accuracy: float
    per_size_accuracy: Dict[str, float]


@dataclass
class LossRecord:
    step: int
    loss: float


@dataclass
class EvalLossRecord:
    step: int
    eval_loss: float


class ExactMatchCurveCallback(TrainerCallback):
    def __init__(
        self,
        *,
        task: Any,
        examples: Sequence[Any],
        tokenizer: Any,
        batch_size: int,
        max_new_tokens: int,
        eval_every_steps: int,
        total_steps: int,
    ) -> None:
        self.task = task
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.eval_every_steps = max(1, eval_every_steps)
        self.total_steps = total_steps
        self.accuracy_history: List[AccuracyRecord] = []
        self.loss_history: List[LossRecord] = []
        self.eval_loss_history: List[EvalLossRecord] = []
        self._recorded_steps: set[int] = set()

    def _record_accuracy(self, step: int, model: Any) -> None:
        if step in self._recorded_steps:
            return
        accuracy, per_size_accuracy = evaluate_accuracy_with_breakdown(
            model=model,
            tokenizer=self.tokenizer,
            examples=self.examples,
            batch_size=self.batch_size,
            max_new_tokens=self.max_new_tokens,
            size_getter=self.task.size_of,
            prediction_parser=self.task.prediction_parser,
        )
        self.accuracy_history.append(
            AccuracyRecord(
                step=int(step),
                accuracy=float(accuracy),
                per_size_accuracy={str(size): float(score) for size, score in sorted(per_size_accuracy.items())},
            )
        )
        self._recorded_steps.add(int(step))

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        self._record_accuracy(0, model)

    def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        step = int(state.global_step)
        if step <= 0:
            return
        if step % self.eval_every_steps == 0 or step >= self.total_steps:
            self._record_accuracy(step, model)

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if not logs:
            return
        if "loss" in logs:
            try:
                loss_value = float(logs["loss"])
            except (TypeError, ValueError):
                loss_value = None
            if loss_value is not None:
                self.loss_history.append(LossRecord(step=int(state.global_step), loss=loss_value))
        if "eval_loss" in logs:
            try:
                eval_value = float(logs["eval_loss"])
            except (TypeError, ValueError):
                eval_value = None
            if eval_value is not None:
                self.eval_loss_history.append(EvalLossRecord(step=int(state.global_step), eval_loss=eval_value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze training dynamics for workshop self-improvement tasks.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("addition", "run_length", "multiplication"),
        default=("addition", "run_length"),
    )
    parser.add_argument("--settings", nargs="+", choices=("base", "composed"), default=("base", "composed"))
    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--format-version", type=str, default="legacy")
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument(
        "--num-val-examples",
        type=int,
        default=0,
        help="If >0, hold out this many examples for validation loss tracking.",
    )
    parser.add_argument(
        "--initial-min-size",
        type=int,
        default=None,
        help="Override task args initial_min_size (bits or digits, depending on task).",
    )
    parser.add_argument(
        "--initial-max-size",
        type=int,
        default=None,
        help="Override task args initial_max_size (bits or digits, depending on task).",
    )
    parser.add_argument(
        "--initial-train-per-size",
        type=int,
        default=None,
        help="Override task args initial_train_per_size.",
    )
    parser.add_argument(
        "--expand-num-size",
        type=int,
        default=None,
        help="Override task args expand_num_size for composed sampling.",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-every-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(
    *,
    output_path: Path,
    task_name: str,
    setting: str,
    loss_history: Sequence[LossRecord],
    eval_loss_history: Sequence[EvalLossRecord],
    accuracy_history: Sequence[AccuracyRecord],
) -> None:
    ensure_dir(output_path.parent)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )
    fig, ax_loss = plt.subplots(figsize=(8.5, 4.8))
    ax_acc = ax_loss.twinx()

    if loss_history:
        ax_loss.plot(
            [record.step for record in loss_history],
            [record.loss for record in loss_history],
            color="#1f77b4",
            linewidth=2.0,
            label="Training loss",
        )
    if eval_loss_history:
        ax_loss.plot(
            [record.step for record in eval_loss_history],
            [record.eval_loss for record in eval_loss_history],
            color="#2ca02c",
            linewidth=2.0,
            linestyle="--",
            label="Validation loss",
        )
    if accuracy_history:
        ax_acc.plot(
            [record.step for record in accuracy_history],
            [record.accuracy for record in accuracy_history],
            color="#d62728",
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Exact-match accuracy",
        )

    ax_loss.set_title(f"{task_name.title()} {setting} dynamics")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss", color="#1f77b4")
    ax_acc.set_ylabel("Exact-match accuracy", color="#d62728")
    ax_acc.set_ylim(-0.02, 1.02)
    ax_loss.grid(alpha=0.25, linewidth=0.7)

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_case(
    *,
    task_name: str,
    setting: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    task, task_args = build_task_bundle(task_name, args.model_name, args.seed, args.format_version, args)
    if args.initial_min_size is not None:
        task_args.initial_min_size = args.initial_min_size
    if args.initial_max_size is not None:
        task_args.initial_max_size = args.initial_max_size
    if args.initial_train_per_size is not None:
        task_args.initial_train_per_size = args.initial_train_per_size
    if args.expand_num_size is not None:
        task_args.expand_num_size = args.expand_num_size
    raw_examples: List[Any]
    try:
        raw_examples = prepare_examples(task_name, task, task_args, args.seed, setting, args.num_examples)
    except RuntimeError as exc:
        message = str(exc)
        match = re.search(r"only produced (\d+) examples", message)
        if match:
            fallback = int(match.group(1))
            if fallback <= 0:
                raise
            print(
                f"[WARN] {task_name}/{setting} capped examples from {args.num_examples} to {fallback}.",
                flush=True,
            )
            raw_examples = prepare_examples(task_name, task, task_args, args.seed, setting, fallback)
        else:
            raise
    if args.num_val_examples > 0:
        if len(raw_examples) <= args.num_val_examples:
            fallback_val = max(0, len(raw_examples) // 5)
            if fallback_val <= 0:
                print(
                    f"[WARN] {task_name}/{setting} has only {len(raw_examples)} examples; skipping validation split.",
                    flush=True,
                )
                examples = raw_examples
                val_examples = []
            else:
                print(
                    f"[WARN] {task_name}/{setting} reducing val_examples from {args.num_val_examples} to {fallback_val}.",
                    flush=True,
                )
                examples = raw_examples[: len(raw_examples) - fallback_val]
                val_examples = raw_examples[len(raw_examples) - fallback_val :]
        else:
            examples = raw_examples[: len(raw_examples) - args.num_val_examples]
            val_examples = raw_examples[len(raw_examples) - args.num_val_examples :]
    else:
        examples = raw_examples
        val_examples = []

    token_initializers = task.token_initializers(task_args) if hasattr(task, "token_initializers") else {}
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        token_initializers=token_initializers,
    )

    decode_max_new_tokens = resolve_max_new_tokens(examples, 16)
    callback = ExactMatchCurveCallback(
        task=task,
        examples=examples,
        tokenizer=tokenizer,
        batch_size=args.per_device_eval_batch_size,
        max_new_tokens=decode_max_new_tokens,
        eval_every_steps=args.eval_every_steps,
        total_steps=args.max_steps,
    )

    training_args = make_training_args(
        output_dir / "trainer",
        TrainingConfig(
            num_epochs=1,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            max_steps=args.max_steps,
            eval_steps=args.eval_every_steps if args.num_val_examples > 0 else None,
            decode_max_new_tokens=decode_max_new_tokens,
        ),
        bf16=args.bf16,
        fp16=args.fp16,
        skip_save=True,
        seed=args.seed,
    )
    if val_examples:
        if hasattr(training_args, "evaluation_strategy"):
            training_args.evaluation_strategy = "steps"
        elif hasattr(training_args, "eval_strategy"):
            training_args.eval_strategy = "steps"
        if hasattr(training_args, "eval_steps"):
            training_args.eval_steps = args.eval_every_steps

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=TokenizedPromptTargetDataset(examples, tokenizer),
        eval_dataset=TokenizedPromptTargetDataset(val_examples, tokenizer) if val_examples else None,
        data_collator=CausalLMDataCollator(tokenizer),
        callbacks=[callback],
    )
    trainer.train()

    result = {
        "task": task_name,
        "setting": setting,
        "num_examples": len(examples),
        "format_version": args.format_version,
        "examples": [{"prompt": example.prompt(), "target": example.target()} for example in examples],
        "val_examples": [{"prompt": example.prompt(), "target": example.target()} for example in val_examples],
        "accuracy_history": [asdict(record) for record in callback.accuracy_history],
        "loss_history": [asdict(record) for record in callback.loss_history],
        "eval_loss_history": [asdict(record) for record in callback.eval_loss_history],
    }
    with (output_dir / "dynamics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    save_plot(
        output_path=output_dir / "dynamics.png",
        task_name=task_name,
        setting=setting,
        loss_history=callback.loss_history,
        eval_loss_history=callback.eval_loss_history,
        accuracy_history=callback.accuracy_history,
    )
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_dir)
    ensure_dir(output_root)

    summary: List[Dict[str, Any]] = []
    for task_name in args.tasks:
        for setting in args.settings:
            case_dir = output_root / task_name / setting
            ensure_dir(case_dir)
            result = run_case(task_name=task_name, setting=setting, args=args, output_dir=case_dir)
            summary.append(
                {
                    "task": task_name,
                    "setting": setting,
                    "final_accuracy": result["accuracy_history"][-1]["accuracy"] if result["accuracy_history"] else None,
                    "peak_accuracy": max((entry["accuracy"] for entry in result["accuracy_history"]), default=None),
                }
            )

    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()

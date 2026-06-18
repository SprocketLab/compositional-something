#!/usr/bin/env python3
"""Multiplication-specific entrypoint for the shared self-improvement scaffold."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-improvement multiplication experiment (resumable)")

    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--output-dir", type=str, default="artifacts/runs/self_improvement_multiplication")
    parser.add_argument(
        "--format-version",
        type=str,
        choices=("legacy", "symbolic_v1"),
        default="legacy",
        help="Prompt/target serialization format.",
    )

    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--initial-min-digits", type=int, default=2)
    parser.add_argument("--initial-max-digits", type=int, default=3)
    parser.add_argument("--initial-train-per-digit", type=int, default=4000)
    parser.add_argument(
        "--initial-eval-per-digit",
        type=int,
        default=100,
        help="Per-digit holdout count for the initial digit range (unused for training).",
    )

    parser.add_argument("--num-expand-rounds", type=int, default=8)
    parser.add_argument("--expand-num-digits", type=int, default=2)
    parser.add_argument("--expand-train-per-digit", type=int, default=1200)
    parser.add_argument(
        "--eval-per-digit",
        type=int,
        default=100,
        help="Per-digit evaluation count sampled across the full digit range.",
    )
    parser.add_argument(
        "--composed-eval-per-digit",
        type=int,
        default=50,
        help="Per-digit count for held-out composed evaluation examples.",
    )
    parser.add_argument(
        "--pseudo-label-mode",
        type=str,
        choices=("none", "direct", "compose", "compose_corrupt"),
        default="compose",
        help="How to generate pseudo labels for long examples.",
    )
    parser.add_argument(
        "--corruption-rate",
        type=float,
        default=0.10,
        help="Corruption rate used for compose_corrupt baselines.",
    )
    parser.add_argument(
        "--composed-refresh-mode",
        type=str,
        choices=("dynamic", "static"),
        default="dynamic",
        help="Whether to regenerate composed pools every round (dynamic) or keep the initial pool (static).",
    )

    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--decode-max-new-tokens", type=int, default=48)

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
    parser.add_argument("--skip-save-model", action="store_true")
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Retain per-round model checkpoints instead of deleting them after completion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reset-in-each-round",
        action="store_true",
        help="Reload the base model weights from --model-name before every training round.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing artifacts in --output-dir (continues after the last completed round).",
    )
    parser.add_argument(
        "--resume-from-round",
        type=int,
        default=None,
        help="Resume starting from this round index (overrides --resume detected round).",
    )
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.initial_min_size = args.initial_min_digits
    args.initial_max_size = args.initial_max_digits
    args.initial_train_per_size = args.initial_train_per_digit
    args.initial_eval_per_size = args.initial_eval_per_digit
    args.expand_num_size = args.expand_num_digits
    args.expand_train_per_size = args.expand_train_per_digit
    args.eval_per_size = args.eval_per_digit
    args.composed_eval_per_size = args.composed_eval_per_digit
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    from self.nonadaptive.nonadaptive_loop import run_self_improvement
    from self.tasks import MultiplicationTask

    args = normalize_args(parse_args(argv))
    run_self_improvement(args, MultiplicationTask())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI helpers for legacy run-length bit-string self-improvement."""

from __future__ import annotations

import argparse

from self.self_improvement_recipe import RECIPE_ALGORITHMIC_SELF_IMPROVE_V1


def build_run_length_bit_parser(*, description: str, default_output_dir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--model-name", type=str, default="HuggingFaceTB/SmolLM2-360M")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument(
        "--format-version",
        type=str,
        choices=("legacy", "symbolic_v1"),
        default="legacy",
        help="Prompt/target serialization format.",
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
        help="Whether composed frontier examples may use any number of parts >=2 or exactly two parts.",
    )
    parser.add_argument(
        "--bit-composition-path-mode",
        type=str,
        choices=("random", "fixed_binary"),
        default="random",
        help=(
            "How run-length component sizes are chosen. random preserves the existing stochastic "
            "path selection; fixed_binary uses floor(L/2)+ceil(L/2) for each target length."
        ),
    )
    parser.add_argument(
        "--guarded-compose-rule",
        type=str,
        choices=("none", "run_length_no_boundary_continue", "run_length_unfiltered_pair"),
        default="none",
        help="Optional guarded composition rule for diagnostic plain-output runs.",
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

    parser.add_argument("--initial-min-bits", type=int, default=4)
    parser.add_argument("--initial-max-bits", type=int, default=8)
    parser.add_argument("--initial-train-per-bit", type=int, default=2000)
    parser.add_argument(
        "--initial-eval-per-bit",
        type=int,
        default=50,
        help="Per-bit holdout count for the initial length range (unused for training).",
    )

    parser.add_argument("--num-expand-rounds", type=int, default=4)
    parser.add_argument("--expand-num-bits", type=int, default=4)
    parser.add_argument(
        "--frontier-min-bits",
        type=int,
        default=None,
        help=(
            "Optional explicit minimum frontier size for round-1 expansion. "
            "When unset, the frontier starts at initial_max_bits + 1."
        ),
    )
    parser.add_argument("--expand-train-per-bit", type=int, default=1200)
    parser.add_argument(
        "--eval-per-bit",
        type=int,
        default=100,
        help="Per-bit evaluation count sampled across the full length range.",
    )
    parser.add_argument(
        "--composed-eval-per-bit",
        type=int,
        default=50,
        help="Per-bit count for held-out composed evaluation examples.",
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
    parser.add_argument(
        "--reserve-heldout-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve validation/test examples before filling the seed training split.",
    )
    parser.add_argument(
        "--reserve-shared-eval-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reserve the shared evaluation pool before constructing seed train/validation/test splits.",
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
    parser.add_argument(
        "--self-improve-learning-rate",
        type=float,
        default=None,
        help="Optional recipe override for learning rate in self-improvement rounds only.",
    )
    parser.add_argument(
        "--self-improve-lr-switch-round",
        type=int,
        default=None,
        help=(
            "Optional round index at which to switch to --self-improve-learning-rate-after-switch "
            "for recipe self-improvement rounds."
        ),
    )
    parser.add_argument(
        "--self-improve-learning-rate-after-switch",
        type=float,
        default=None,
        help="Learning rate used from --self-improve-lr-switch-round onward.",
    )
    parser.add_argument(
        "--self-improve-warmup-steps",
        type=int,
        default=None,
        help="Optional recipe override for warmup steps in self-improvement rounds only.",
    )
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
        choices=("none", RECIPE_ALGORITHMIC_SELF_IMPROVE_V1),
        default="none",
        help="Optional recipe model/training preset for run-length bit-string experiments.",
    )
    parser.add_argument(
        "--treat-seed-as-round-zero",
        action="store_true",
        help=(
            "Treat --model-name as a completed round_00 checkpoint: skip round-0 training, "
            "evaluate/save it into round_00, and start learning updates at the first expansion round."
        ),
    )
    parser.add_argument(
        "--bucket-train-batches-by-bits",
        action="store_true",
        help="Bucket training batches so every batch contains examples from a single exact bit length.",
    )
    parser.add_argument(
        "--save-model-policy",
        type=str,
        choices=("final_only", "all_rounds", "none"),
        default="all_rounds",
        help=(
            "Model persistence policy. final_only saves only the final round as a reloadable "
            "checkpoint, all_rounds saves every round, and none disables model saving."
        ),
    )
    parser.add_argument(
        "--skip-save-model",
        action="store_true",
        help="Deprecated compatibility flag equivalent to --save-model-policy none.",
    )
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
    parser.add_argument(
        "--stop-after-round",
        type=int,
        default=None,
        help=(
            "Stop after completing this round. Useful for round-wise seed sweeps: the round is trained, "
            "evaluated, and its next-round pseudo pool is written before exiting."
        ),
    )
    return parser


def normalize_run_length_bit_args(args: argparse.Namespace) -> argparse.Namespace:
    args.initial_min_size = args.initial_min_bits
    args.initial_max_size = args.initial_max_bits
    args.initial_train_per_size = args.initial_train_per_bit
    args.initial_eval_per_size = args.initial_eval_per_bit
    args.expand_num_size = args.expand_num_bits
    args.frontier_min_size = args.frontier_min_bits
    args.expand_train_per_size = args.expand_train_per_bit
    args.eval_per_size = args.eval_per_bit
    args.composed_eval_per_size = args.composed_eval_per_bit
    args.bucket_train_batches_by_size = args.bucket_train_batches_by_bits
    if getattr(args, "skip_save_model", False):
        args.save_model_policy = "none"
    if getattr(args, "save_model_policy", "final_only") == "none":
        args.skip_save_model = True
    return args

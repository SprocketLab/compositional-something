"""CLI parsing and argument normalization for the adaptive core driver."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from core.addition_pipeline import (
    ADDITION_SAMPLING_NATURAL,
    ADDITION_WIDTH_EXACT_DIGITS,
)
from self.adaptive.proposals.proposal_config_schema import PROPOSAL_OUTPUT_SCHEMAS
from self.adaptive.proposals.proposal_grpo_traces import PROPOSAL_GRPO_REWARD_MODES, PROPOSAL_GRPO_ZERO_VARIANCE_MODES
from self.tasks.bit import RUN_LENGTH_TARGET_RUN_STATE


TASK_CHOICES = ("addition", "run_length")
CONDITION_CHOICES = ("config", "program", "policy", "meta")
OUTCOME_TRACE_TARGET_MODES = ("none", "numeric", "textual", "numeric_textual")
CANDIDATE_EXECUTION_MODES = ("local_parallel", "slurm_array", "serial")
CONTROLLER_EXECUTION_MODES = ("local", "slurm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate adaptive composition candidates round-by-round.")
    parser.add_argument("--task", choices=TASK_CHOICES, default=None)
    parser.add_argument("--condition", choices=CONDITION_CHOICES, default="config")
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument(
        "--proposal-model-name",
        default="current",
        help="Use 'current' to sample config proposals from the current task checkpoint.",
    )
    parser.add_argument(
        "--proposal-output-schema",
        choices=PROPOSAL_OUTPUT_SCHEMAS,
        default="action_prediction",
        help=(
            "Config proposal completion schema. 'action_prediction' asks the model to emit "
            "both an executable proposal and its predicted outcome; 'plain' keeps the legacy "
            "left/right/guard JSON."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/adaptive_candidate_training"))
    parser.add_argument("--proposal-fixture-jsonl", type=Path, default=None)
    parser.add_argument("--num-rounds", type=int, default=100)
    parser.add_argument(
        "--max-attempt-rounds",
        type=int,
        default=None,
        help="Maximum proposal/candidate attempts. Defaults to 10 * num_rounds.",
    )
    parser.add_argument(
        "--no-selection-patience",
        type=int,
        default=None,
        help=(
            "Stop after this many consecutive attempts without a selected candidate. "
            "Defaults to max_attempt_rounds."
        ),
    )
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--initial-min-size", type=int, default=None)
    parser.add_argument("--initial-max-size", type=int, default=None)
    parser.add_argument("--frontier-max-size", type=int, default=None)
    parser.add_argument("--initial-train-per-size", type=int, default=None)
    parser.add_argument("--initial-eval-per-size", type=int, default=50)
    parser.add_argument("--candidate-train-per-size", type=int, default=None)
    parser.add_argument("--eval-per-size", type=int, default=100)
    parser.add_argument("--composed-eval-per-size", type=int, default=100)
    parser.add_argument("--allow-repeat-targets", action="store_true")

    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--decode-max-new-tokens", type=int, default=48)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--init-from-scratch", action="store_true")
    parser.add_argument("--tokenizer-mode", choices=("auto", "fixed_char"), default="auto")
    parser.add_argument("--recipe", default="none")
    parser.add_argument("--bucket-train-batches-by-size", action="store_true")
    parser.add_argument("--treat-seed-as-round-zero", action="store_true")
    parser.add_argument("--keep-all-candidate-models", action="store_true")
    parser.add_argument(
        "--controller-execution-mode",
        choices=CONTROLLER_EXECUTION_MODES,
        default="local",
        help=(
            "Run controller GPU phases locally, or dispatch them to a single-GPU SLURM worker "
            "so the parent controller can be CPU-only."
        ),
    )
    parser.add_argument(
        "--controller-worker-sbatch-script",
        type=Path,
        default=Path("launchers/self/run_adaptive_controller_worker_ailab.sbatch"),
        help="SBATCH script used when controller_execution_mode=slurm.",
    )
    parser.add_argument(
        "--controller-worker-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval while waiting for controller GPU workers.",
    )
    parser.add_argument(
        "--controller-worker-time-limit",
        default="04:00:00",
        help="Fallback SBATCH --time limit for controller GPU workers.",
    )
    parser.add_argument(
        "--controller-seed-worker-time-limit",
        default="04:00:00",
        help="SBATCH --time limit for seed-training controller GPU workers.",
    )
    parser.add_argument(
        "--controller-round-worker-time-limit",
        default="01:00:00",
        help="SBATCH --time limit for round eval/proposal/pseudolabel controller GPU workers.",
    )
    parser.add_argument(
        "--controller-grpo-worker-time-limit",
        default="01:00:00",
        help="SBATCH --time limit for proposal-GRPO controller GPU workers.",
    )
    parser.add_argument(
        "--candidate-execution-mode",
        choices=CANDIDATE_EXECUTION_MODES,
        default="local_parallel",
        help=(
            "Train candidates as local subprocesses by default, or use slurm_array/serial "
            "for compatibility."
        ),
    )
    parser.add_argument(
        "--candidate-local-parallelism",
        type=int,
        default=4,
        help="Maximum local candidate-worker subprocesses to run concurrently on the allocated node/GPU.",
    )
    parser.add_argument(
        "--candidate-local-pack-size",
        type=int,
        default=1,
        help=(
            "Number of candidates to run sequentially inside one local worker subprocess. "
            "Use 1 for the legacy one-candidate-per-process behavior."
        ),
    )
    parser.add_argument(
        "--candidate-local-cache-base-state",
        action="store_true",
        help=(
            "In packed local candidate workers, cache a CPU copy of the source checkpoint "
            "state after the first load and instantiate fresh candidate models from that "
            "cached state. This avoids repeated checkpoint weight reads but increases CPU "
            "memory use. Candidate training semantics remain isolated."
        ),
    )
    parser.add_argument(
        "--candidate-array-sbatch-script",
        type=Path,
        default=Path("launchers/self/run_adaptive_candidate_worker_ailab.sbatch"),
        help="SBATCH script used when candidate_execution_mode=slurm_array.",
    )
    parser.add_argument(
        "--candidate-array-max-parallel",
        type=int,
        default=4,
        help="Maximum simultaneously running candidate workers in the SLURM array. Use 0 for no array throttle.",
    )
    parser.add_argument(
        "--candidate-array-poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval while waiting for candidate-worker array metrics.",
    )
    parser.add_argument(
        "--candidate-array-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional controller-side timeout for one candidate-worker array. 0 disables it.",
    )
    parser.add_argument(
        "--candidate-array-time-limit",
        default="08:00:00",
        help="SBATCH --time limit for each candidate-worker array task.",
    )
    parser.add_argument("--run-candidate-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-worker-spec", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-candidate-pack-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--candidate-worker-pack-spec", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--run-controller-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--controller-worker-spec", type=Path, default=None, help=argparse.SUPPRESS)

    parser.add_argument("--proposal-max-new-tokens", type=int, default=512)
    parser.add_argument("--proposal-temperature", type=float, default=0.8)
    parser.add_argument("--proposal-top-p", type=float, default=0.95)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--program-timeout-seconds", type=float, default=1.0)
    parser.add_argument("--program-batch-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--lambda-final", type=float, default=0.1)
    parser.add_argument("--selection-min-reward", type=float, default=0.0)
    parser.add_argument("--init-final-accuracy", type=float, default=None)
    parser.add_argument("--max-traces-per-round", type=int, default=2)
    parser.add_argument(
        "--proposal-trace-replay-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of task train examples to add as repeated selected proposal traces. "
            "Set to 0 to disable proposal-trace rehearsal."
        ),
    )
    parser.add_argument(
        "--proposal-trace-replay-max-examples",
        type=int,
        default=256,
        help="Maximum replayed proposal-trace examples mixed into each candidate update.",
    )
    parser.add_argument(
        "--post-task-proposal-rehearsal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a bounded proposal-trace SFT phase after each candidate task SFT update.",
    )
    parser.add_argument(
        "--post-task-proposal-rehearsal-repeat-count",
        type=int,
        default=64,
        help="Nominal repeats per available proposal trace in the post-task proposal rehearsal phase.",
    )
    parser.add_argument(
        "--post-task-proposal-rehearsal-max-examples",
        type=int,
        default=256,
        help="Maximum proposal-trace examples in the post-task proposal rehearsal phase.",
    )
    parser.add_argument(
        "--outcome-trace-target-mode",
        choices=OUTCOME_TRACE_TARGET_MODES,
        default="numeric",
        help="Compact config outcome-supervision target format mixed into future candidate updates.",
    )
    parser.add_argument(
        "--outcome-trace-replay-ratio",
        type=float,
        default=0.10,
        help="Fraction of task train examples to add as replayed compact outcome traces.",
    )
    parser.add_argument(
        "--outcome-trace-replay-max-examples",
        type=int,
        default=4096,
        help="Maximum replayed outcome-trace examples mixed into each candidate update.",
    )
    parser.add_argument(
        "--invalid-outcome-reward",
        type=float,
        default=-0.1,
        help="Reward value used in compact outcome targets for invalid or untrained candidates.",
    )
    parser.add_argument(
        "--proposal-grpo-steps",
        type=int,
        default=None,
        help=(
            "Immediate GRPO-style proposal-validity update steps after each config attempt. "
            "Defaults to 1 for condition=config and 0 otherwise."
        ),
    )
    parser.add_argument(
        "--proposal-grpo-learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate for the lightweight proposal-validity GRPO update.",
    )
    parser.add_argument(
        "--proposal-grpo-kl-coef",
        type=float,
        default=0.01,
        help="Coefficient for the sampled-token KL proxy against cached pre-update logprobs.",
    )
    parser.add_argument(
        "--proposal-grpo-grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm for the proposal-validity GRPO update.",
    )
    parser.add_argument(
        "--proposal-grpo-zero-variance",
        choices=PROPOSAL_GRPO_ZERO_VARIANCE_MODES,
        default="fixed_baseline",
        help="How to handle equal rewards within one proposal group.",
    )
    parser.add_argument(
        "--proposal-grpo-reward-mode",
        choices=PROPOSAL_GRPO_REWARD_MODES,
        default="outcome",
        help=(
            "Use candidate outcome rewards for proposal GRPO by default, or the older "
            "validity-only verifier reward for ablations."
        ),
    )
    parser.add_argument(
        "--proposal-grpo-outcome-scale",
        type=float,
        default=0.05,
        help="Scale for clipping candidate outcome rewards into [-1, 1] for proposal GRPO.",
    )
    parser.add_argument(
        "--proposal-grpo-fixed-baseline",
        type=float,
        default=0.5,
        help="Baseline subtracted from rewards when all proposal rewards are equal.",
    )
    parser.add_argument(
        "--keep-all-proposal-grpo-checkpoints",
        action="store_true",
        help=(
            "Keep every full proposal-GRPO model checkpoint. By default, superseded "
            "proposal-GRPO model directories are pruned while JSON/JSONL logs are kept."
        ),
    )

    parser.add_argument("--format-version", choices=("legacy", "symbolic_v1"), default="legacy")
    parser.add_argument(
        "--target-mode",
        choices=("default", "plain_output", "symbol_run_pair", RUN_LENGTH_TARGET_RUN_STATE),
        default=RUN_LENGTH_TARGET_RUN_STATE,
        help="Run-length target mode. The candidate loop is primarily intended for run_state.",
    )
    parser.add_argument("--symbol-alphabet-size", type=int, default=2)
    parser.add_argument("--addition-width-mode", default=ADDITION_WIDTH_EXACT_DIGITS)
    parser.add_argument("--addition-sampling-mode", default=ADDITION_SAMPLING_NATURAL)
    parser.add_argument("--plan-log-path", type=Path, default=Path("plan/260603-self-improvement-init.md"))
    parser.add_argument(
        "--dry-run-data-only",
        action="store_true",
        help="Validate proposals and build exact-pair composed data, but skip model prediction/training.",
    )
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.task is None:
        raise ValueError("task must be set.")
    if args.num_rounds < 0:
        raise ValueError("num_rounds must be non-negative.")
    if args.max_attempt_rounds is None:
        args.max_attempt_rounds = max(args.num_rounds, args.num_rounds * 10)
    if args.max_attempt_rounds < args.num_rounds:
        raise ValueError("max_attempt_rounds must be >= num_rounds.")
    if args.no_selection_patience is None:
        args.no_selection_patience = args.max_attempt_rounds
    if args.no_selection_patience < 1:
        raise ValueError("no_selection_patience must be positive.")
    if args.num_candidates < 1:
        raise ValueError("num_candidates must be positive.")
    if args.candidate_local_parallelism < 1:
        raise ValueError("candidate_local_parallelism must be positive.")
    if args.candidate_local_pack_size < 1:
        raise ValueError("candidate_local_pack_size must be positive.")
    if args.candidate_array_max_parallel < 0:
        raise ValueError("candidate_array_max_parallel must be non-negative.")
    if args.candidate_array_poll_seconds <= 0.0:
        raise ValueError("candidate_array_poll_seconds must be positive.")
    if args.candidate_array_timeout_seconds < 0.0:
        raise ValueError("candidate_array_timeout_seconds must be non-negative.")
    if args.run_candidate_worker and args.candidate_worker_spec is None:
        raise ValueError("candidate_worker_spec is required with run_candidate_worker.")
    if args.run_candidate_pack_worker and args.candidate_worker_pack_spec is None:
        raise ValueError("candidate_worker_pack_spec is required with run_candidate_pack_worker.")
    if args.controller_worker_poll_seconds <= 0.0:
        raise ValueError("controller_worker_poll_seconds must be positive.")
    if args.run_controller_worker and args.controller_worker_spec is None:
        raise ValueError("controller_worker_spec is required with run_controller_worker.")
    if args.initial_eval_per_size < 0 or args.eval_per_size < 0 or args.composed_eval_per_size < 0:
        raise ValueError("Evaluation counts must be non-negative.")
    if args.candidate_train_per_size is not None and args.candidate_train_per_size < 0:
        raise ValueError("candidate_train_per_size must be non-negative.")
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 and --fp16.")
    if args.repair_attempts < 0:
        raise ValueError("repair_attempts must be non-negative.")
    if args.program_timeout_seconds <= 0.0:
        raise ValueError("program_timeout_seconds must be positive.")
    if args.program_batch_timeout_seconds <= 0.0:
        raise ValueError("program_batch_timeout_seconds must be positive.")
    if args.proposal_trace_replay_ratio < 0.0:
        raise ValueError("proposal_trace_replay_ratio must be non-negative.")
    if args.proposal_trace_replay_max_examples < 0:
        raise ValueError("proposal_trace_replay_max_examples must be non-negative.")
    if args.post_task_proposal_rehearsal_repeat_count < 0:
        raise ValueError("post_task_proposal_rehearsal_repeat_count must be non-negative.")
    if args.post_task_proposal_rehearsal_max_examples < 0:
        raise ValueError("post_task_proposal_rehearsal_max_examples must be non-negative.")
    if args.outcome_trace_replay_ratio < 0.0:
        raise ValueError("outcome_trace_replay_ratio must be non-negative.")
    if args.outcome_trace_replay_max_examples < 0:
        raise ValueError("outcome_trace_replay_max_examples must be non-negative.")
    if args.proposal_grpo_steps is None:
        args.proposal_grpo_steps = 1 if args.condition == "config" else 0
    if args.proposal_grpo_steps < 0:
        raise ValueError("proposal_grpo_steps must be non-negative.")
    if args.condition != "config" and args.proposal_grpo_steps > 0:
        raise ValueError("proposal_grpo_steps is currently supported only for condition=config.")
    if args.proposal_grpo_learning_rate <= 0.0:
        raise ValueError("proposal_grpo_learning_rate must be positive.")
    if args.proposal_grpo_kl_coef < 0.0:
        raise ValueError("proposal_grpo_kl_coef must be non-negative.")
    if args.proposal_grpo_grad_clip <= 0.0:
        raise ValueError("proposal_grpo_grad_clip must be positive.")
    if args.proposal_grpo_outcome_scale <= 0.0:
        raise ValueError("proposal_grpo_outcome_scale must be positive.")
    if not math.isfinite(args.proposal_grpo_fixed_baseline):
        raise ValueError("proposal_grpo_fixed_baseline must be finite.")

    if args.task == "addition":
        args.initial_min_size = args.initial_min_size if args.initial_min_size is not None else 3
        args.initial_max_size = args.initial_max_size if args.initial_max_size is not None else 7
        args.frontier_max_size = args.frontier_max_size if args.frontier_max_size is not None else 31
        args.initial_train_per_size = (
            args.initial_train_per_size if args.initial_train_per_size is not None else 5000
        )
        args.candidate_train_per_size = (
            args.candidate_train_per_size if args.candidate_train_per_size is not None else 5000
        )
        args.composed_strategy = "with_carry"
        args.composition_error_percent = 0.0
        args.corruption_rate = 0.0
        args.pseudo_label_mode = "compose"
        args.initial_min_digits = args.initial_min_size
        args.initial_max_digits = args.initial_max_size
        args.initial_train_per_digit = args.initial_train_per_size
        args.initial_eval_per_digit = args.initial_eval_per_size
        args.expand_train_per_digit = args.candidate_train_per_size
        args.eval_per_digit = args.eval_per_size
        args.composed_eval_per_digit = args.composed_eval_per_size
    else:
        args.initial_min_size = args.initial_min_size if args.initial_min_size is not None else 8
        args.initial_max_size = args.initial_max_size if args.initial_max_size is not None else 16
        args.frontier_max_size = args.frontier_max_size if args.frontier_max_size is not None else 48
        args.initial_train_per_size = (
            args.initial_train_per_size if args.initial_train_per_size is not None else 50000
        )
        args.candidate_train_per_size = (
            args.candidate_train_per_size if args.candidate_train_per_size is not None else 2000
        )
        args.corruption_rate = 0.0
        args.pseudo_label_mode = "compose"
        args.compose_arity = "exact2"
        args.bit_composition_path_mode = "random"
        args.guarded_compose_rule = "none"
        args.initial_min_bits = args.initial_min_size
        args.initial_max_bits = args.initial_max_size
        args.initial_train_per_bit = args.initial_train_per_size
        args.initial_eval_per_bit = args.initial_eval_per_size
        args.expand_train_per_bit = args.candidate_train_per_size
        args.eval_per_bit = args.eval_per_size
        args.composed_eval_per_bit = args.composed_eval_per_size

    args.expand_train_per_size = args.candidate_train_per_size
    args.frontier_min_size = args.initial_max_size + 1
    if args.initial_min_size < 1:
        raise ValueError("initial_min_size must be at least 1.")
    if args.initial_max_size < args.initial_min_size:
        raise ValueError("initial_max_size must be >= initial_min_size.")
    if args.frontier_max_size <= args.initial_max_size:
        raise ValueError("frontier_max_size must be greater than initial_max_size.")
    return args

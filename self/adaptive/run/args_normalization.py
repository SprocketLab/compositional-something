"""Argument validation and task-specific default normalization."""

from __future__ import annotations

import argparse
import math


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

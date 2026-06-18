from __future__ import annotations

from self.adaptive.run import args as adaptive_args


def test_args_normalization_preserves_task_defaults() -> None:
    parser = adaptive_args.build_parser()

    addition_args = adaptive_args.normalize_args(parser.parse_args(["--task", "addition"]))
    assert addition_args.initial_min_size == 3
    assert addition_args.initial_max_size == 7
    assert addition_args.frontier_min_size == 8
    assert addition_args.frontier_max_size == 31
    assert addition_args.initial_min_digits == addition_args.initial_min_size
    assert addition_args.expand_train_per_digit == addition_args.candidate_train_per_size
    assert addition_args.proposal_grpo_steps == 1

    run_length_args = adaptive_args.normalize_args(parser.parse_args(["--task", "run_length"]))
    assert run_length_args.initial_min_size == 8
    assert run_length_args.initial_max_size == 16
    assert run_length_args.frontier_min_size == 17
    assert run_length_args.frontier_max_size == 48
    assert run_length_args.initial_min_bits == run_length_args.initial_min_size
    assert run_length_args.expand_train_per_bit == run_length_args.candidate_train_per_size

    program_args = adaptive_args.normalize_args(
        parser.parse_args(["--task", "addition", "--condition", "program"])
    )
    assert program_args.proposal_grpo_steps == 0

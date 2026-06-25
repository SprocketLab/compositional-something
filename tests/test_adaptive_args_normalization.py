from __future__ import annotations

from self.adaptive import args as adaptive_args


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
    assert addition_args.proposal_prompt_action_history is True
    assert addition_args.proposal_grpo_learning_rate == 1e-6
    assert addition_args.proposal_grpo_novelty_bonus_beta == 0.05
    assert addition_args.max_attempt_rounds == 100
    assert addition_args.max_selected_rounds == 0
    assert addition_args.no_selection_patience == 100
    assert addition_args.proposal_format_mask_config_values is True

    run_length_args = adaptive_args.normalize_args(parser.parse_args(["--task", "run_length"]))
    assert run_length_args.initial_min_size == 8
    assert run_length_args.initial_max_size == 16
    assert run_length_args.frontier_min_size == 17
    assert run_length_args.frontier_max_size == 48
    assert run_length_args.initial_min_bits == run_length_args.initial_min_size
    assert run_length_args.expand_train_per_bit == run_length_args.candidate_train_per_size


def test_args_normalization_uses_attempt_budget_by_default() -> None:
    parser = adaptive_args.build_parser()

    args = adaptive_args.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--max-attempt-rounds",
                "10",
            ]
        )
    )

    assert args.max_selected_rounds == 0
    assert args.max_attempt_rounds == 10
    assert args.no_selection_patience == 10

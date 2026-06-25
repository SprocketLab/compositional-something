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
    assert addition_args.proposal_update_loss_mode == "merged_agent"
    assert addition_args.proposal_prompt_action_history is True
    assert addition_args.proposal_grpo_learning_rate == 1e-6
    assert addition_args.proposal_grpo_novelty_bonus_beta == 0.05
    assert addition_args.post_task_proposal_rehearsal is False
    assert addition_args.num_rounds is None
    assert addition_args.max_attempt_rounds == 100
    assert addition_args.max_selected_rounds == 0
    assert addition_args.no_selection_patience == 100
    assert addition_args.candidate_eval_backend == "transformers"
    assert addition_args.vllm_python_bin is None
    assert addition_args.vllm_gpu_memory_utilization == 0.80
    assert addition_args.vllm_dtype == "auto"
    assert addition_args.vllm_flashinfer_sampler == "off"
    assert addition_args.vllm_enforce_eager is False
    assert addition_args.vllm_max_model_len == 0
    assert addition_args.vllm_max_num_seqs == 0
    assert addition_args.vllm_max_num_batched_tokens == 0
    assert addition_args.proposal_format_mask_config_values is True

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

    legacy_args = adaptive_args.normalize_args(
        parser.parse_args(["--task", "addition", "--proposal-update-loss-mode", "legacy_grpo"])
    )
    assert legacy_args.post_task_proposal_rehearsal is True


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

    assert args.num_rounds is None
    assert args.max_selected_rounds == 0
    assert args.max_attempt_rounds == 10
    assert args.no_selection_patience == 10


def test_args_normalization_keeps_num_rounds_as_legacy_selected_cap_alias() -> None:
    parser = adaptive_args.build_parser()

    args = adaptive_args.normalize_args(parser.parse_args(["--task", "addition", "--num-rounds", "5"]))

    assert args.num_rounds == 5
    assert args.max_selected_rounds == 5
    assert args.max_attempt_rounds == 50
    assert args.no_selection_patience == 50

    explicit_args = adaptive_args.normalize_args(
        parser.parse_args(
            [
                "--task",
                "addition",
                "--num-rounds",
                "5",
                "--max-selected-rounds",
                "2",
                "--max-attempt-rounds",
                "10",
            ]
        )
    )

    assert explicit_args.num_rounds == 5
    assert explicit_args.max_selected_rounds == 2
    assert explicit_args.max_attempt_rounds == 10
    assert explicit_args.no_selection_patience == 10

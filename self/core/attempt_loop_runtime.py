"""Adaptive selected-round attempt loop orchestration."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from self.core.attempt_candidate_runtime import run_candidate_attempt
from self.core.attempt_loop_models import CandidateAttemptDeps, AttemptLoopDeps, AttemptLoopResult

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


def run_adaptive_attempt_loop(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    rng: random.Random,
    output_dir: Path,
    checkpoint_manager: Any,
    source_examples: list[Any],
    source_sizes: set[int],
    exclude_keys: set[Any],
    eval_examples: Sequence[Any],
    current_checkpoint: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    summary_records: list[JsonDict],
    deps: AttemptLoopDeps,
) -> AttemptLoopResult:
    selected_rounds = 0
    attempt_index = 0
    consecutive_no_selection = 0
    proposal_trace_buffer: list[Any] = []
    outcome_trace_buffer: list[Any] = []
    proposal_grpo_update_count = 0
    candidate_attempt_deps = CandidateAttemptDeps(
        run_round_model_dispatch=deps.run_round_model_dispatch,
        train_candidate_metrics=deps.train_candidate_metrics,
        select_candidate=deps.select_candidate,
        write_round_trace=deps.write_round_trace,
        handle_attempt_outcome=deps.handle_attempt_outcome,
        round_model_dispatch_deps=deps.round_model_dispatch_deps,
        attempt_outcome_deps=deps.attempt_outcome_deps,
    )

    while selected_rounds < args.num_rounds and attempt_index < args.max_attempt_rounds:
        attempt_index += 1
        selected_round_for_prompt = selected_rounds + 1
        round_dir = output_dir / f"attempt_{attempt_index:04d}"
        deps.ensure_dir(round_dir)
        print(
            f"[INFO] Starting adaptive candidate attempt {attempt_index} "
            f"(selected_round={selected_round_for_prompt}/{args.num_rounds})",
            flush=True,
        )

        attempt_prompt = deps.build_attempt_prompt(
            args=args,
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            init_final_accuracy=init_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            source_sizes=source_sizes,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            deps=deps.attempt_prompt_deps,
        )
        prompt = attempt_prompt.prompt
        default_program_pair = attempt_prompt.default_program_pair
        deps.write_json(round_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})

        if args.dry_run_data_only:
            dry_result = deps.run_dry_attempt(
                args=args,
                task=task,
                round_dir=round_dir,
                output_dir=output_dir,
                selected_round_for_prompt=selected_round_for_prompt,
                attempt_index=attempt_index,
                source_sizes=source_sizes,
                default_program_pair=default_program_pair,
                source_examples=source_examples,
                exclude_keys=exclude_keys,
                rng=rng,
                summary_records=summary_records,
                selected_rounds=selected_rounds,
                consecutive_no_selection=consecutive_no_selection,
                deps=deps.dry_run_attempt_deps,
            )
            selected_rounds = dry_result.selected_rounds
            consecutive_no_selection = dry_result.consecutive_no_selection
            if dry_result.should_stop:
                break
            continue

        outcome_result = run_candidate_attempt(
            args=args,
            task=task,
            config=config,
            output_dir=output_dir,
            round_dir=round_dir,
            checkpoint_manager=checkpoint_manager,
            source_examples=source_examples,
            source_sizes=source_sizes,
            exclude_keys=exclude_keys,
            eval_examples=eval_examples,
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_grpo_update_count=proposal_grpo_update_count,
            summary_records=summary_records,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            deps=candidate_attempt_deps,
        )
        selected_rounds = outcome_result.selected_rounds
        consecutive_no_selection = outcome_result.consecutive_no_selection
        current_checkpoint = outcome_result.current_checkpoint
        current_final_accuracy = outcome_result.current_final_accuracy
        current_per_size_accuracy = outcome_result.current_per_size_accuracy
        proposal_grpo_update_count = outcome_result.proposal_grpo_update_count
        if outcome_result.should_break:
            break

    return AttemptLoopResult(
        selected_rounds=selected_rounds,
        attempt_index=attempt_index,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        source_sizes=source_sizes,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_grpo_update_count=proposal_grpo_update_count,
    )

"""Model-backed candidate attempt body for adaptive selected-round runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from self.adaptive.attempts.attempt_loop_models import CandidateAttemptDeps
from self.adaptive.attempts.attempt_outcome_models import AttemptOutcomeResult

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


def run_candidate_attempt(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    output_dir: Path,
    round_dir: Path,
    checkpoint_manager: Any,
    source_examples: list[Any],
    source_sizes: set[int],
    exclude_keys: set[Any],
    eval_examples: Sequence[Any],
    current_checkpoint: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    proposal_trace_buffer: list[Any],
    outcome_trace_buffer: list[Any],
    proposal_grpo_update_count: int,
    summary_records: list[dict[str, Any]],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    deps: CandidateAttemptDeps,
) -> AttemptOutcomeResult:
    deleted_replaced_model_dirs: list[str] = []
    round_result = deps.run_round_model_dispatch(
        args=args,
        task=task,
        config=config,
        current_checkpoint=current_checkpoint,
        round_dir=round_dir,
        source_examples=source_examples,
        eval_examples=eval_examples,
        exclude_keys=exclude_keys,
        source_sizes=source_sizes,
        selected_round_for_prompt=selected_round_for_prompt,
        attempt_index=attempt_index,
        selected_rounds=selected_rounds,
        consecutive_no_selection=consecutive_no_selection,
        init_final_accuracy=init_final_accuracy,
        deps=deps.round_model_dispatch_deps,
    )
    current_final_accuracy = round_result.current_final_accuracy
    current_per_size_accuracy = round_result.current_per_size_accuracy
    prompt = round_result.prompt
    proposal_results = round_result.proposal_results
    work_items = round_result.work_items

    metrics = deps.train_candidate_metrics(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=prompt,
        round_index=selected_round_for_prompt,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        config=config,
        attempt_index=attempt_index,
    )
    selected = deps.select_candidate(metrics, args.selection_min_reward)
    trace_rows = deps.write_round_trace(
        args=args,
        task_name=args.task,
        round_index=selected_round_for_prompt,
        prompt=prompt,
        work_items=work_items,
        metrics=metrics,
        path=round_dir / "trace_examples.jsonl",
    )
    checkpoint_manager.cleanup_unselected_candidates(metrics=metrics, selected=selected)
    return deps.handle_attempt_outcome(
        args=args,
        task=task,
        output_dir=output_dir,
        round_dir=round_dir,
        attempt_index=attempt_index,
        selected_round_for_prompt=selected_round_for_prompt,
        selected_rounds=selected_rounds,
        consecutive_no_selection=consecutive_no_selection,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        source_sizes=source_sizes,
        source_examples=source_examples,
        exclude_keys=exclude_keys,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_grpo_update_count=proposal_grpo_update_count,
        deleted_replaced_model_dirs=deleted_replaced_model_dirs,
        summary_records=summary_records,
        prompt=prompt,
        proposal_results=proposal_results,
        metrics=metrics,
        work_items=work_items,
        selected=selected,
        trace_rows=trace_rows,
        checkpoint_manager=checkpoint_manager,
        deps=deps.attempt_outcome_deps,
    )


__all__ = ["CandidateAttemptDeps", "run_candidate_attempt"]

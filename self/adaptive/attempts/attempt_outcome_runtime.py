"""Selected/no-selection attempt outcome handling for adaptive runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from self.adaptive.attempts.attempt_no_selection_runtime import handle_no_selection_attempt as _handle_no_selection_attempt
from self.adaptive.attempts.attempt_outcome_models import AttemptOutcomeDeps, AttemptOutcomeResult, JsonDict
from self.adaptive.attempts.attempt_selected_runtime import handle_selected_attempt as _handle_selected_attempt
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposals.proposal_prompts import PromptBundle


def handle_attempt_outcome(
    *,
    args: argparse.Namespace,
    task: Any,
    output_dir: Path,
    round_dir: Path,
    attempt_index: int,
    selected_round_for_prompt: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    current_checkpoint: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    source_sizes: set[int],
    source_examples: list[Any],
    exclude_keys: set[Any],
    proposal_trace_buffer: list[Any],
    outcome_trace_buffer: list[Any],
    proposal_grpo_update_count: int,
    deleted_replaced_model_dirs: list[str],
    summary_records: list[JsonDict],
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    metrics: Sequence[CandidateMetrics],
    work_items: Sequence[CandidateWorkItem],
    selected: Optional[CandidateMetrics],
    trace_rows: Sequence[Mapping[str, Any]],
    checkpoint_manager: Any,
    deps: AttemptOutcomeDeps,
) -> AttemptOutcomeResult:
    selected_payload = selected.to_json_dict() if selected is not None else None
    deps.write_json(round_dir / "candidate_metrics.json", [metric.to_json_dict() for metric in metrics])
    deps.write_json(round_dir / "selected_candidate.json", selected_payload)
    frontier_min = args.frontier_min_size
    frontier_max = args.frontier_max_size
    outcome_traces = deps.build_round_outcome_trace_examples(
        args=args,
        task_name=args.task,
        condition=args.condition,
        round_index=selected_round_for_prompt,
        proposal_results=proposal_results,
        metrics=metrics,
        selected=selected,
        source_sizes=sorted(source_sizes),
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        current_final_accuracy=current_final_accuracy,
        init_final_accuracy=init_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
    )
    if outcome_traces:
        outcome_trace_buffer.extend(outcome_traces)
        deps.write_trace_jsonl(
            round_dir / "outcome_trace_examples.jsonl",
            [trace.to_json_dict() for trace in outcome_traces],
        )
        deps.write_trace_jsonl(
            output_dir / "outcome_trace_buffer.jsonl",
            [trace.to_json_dict() for trace in outcome_trace_buffer],
        )

    if selected is None:
        return _handle_no_selection_attempt(
            args=args,
            output_dir=output_dir,
            round_dir=round_dir,
            attempt_index=attempt_index,
            selected_round_for_prompt=selected_round_for_prompt,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_grpo_update_count=proposal_grpo_update_count,
            deleted_replaced_model_dirs=deleted_replaced_model_dirs,
            summary_records=summary_records,
            source_sizes=source_sizes,
            prompt=prompt,
            proposal_results=proposal_results,
            metrics=metrics,
            trace_rows=trace_rows,
            outcome_traces=outcome_traces,
            checkpoint_manager=checkpoint_manager,
            deps=deps,
        )

    return _handle_selected_attempt(
        args=args,
        task=task,
        output_dir=output_dir,
        round_dir=round_dir,
        attempt_index=attempt_index,
        selected_rounds=selected_rounds,
        current_checkpoint=current_checkpoint,
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
        selected_payload=selected_payload,
        trace_rows=trace_rows,
        outcome_traces=outcome_traces,
        checkpoint_manager=checkpoint_manager,
        deps=deps,
    )

"""Selected-candidate attempt outcome handling for adaptive runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from self.core.attempt_outcome_models import AttemptOutcomeDeps, AttemptOutcomeResult, JsonDict
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposal_prompts import PromptBundle


def handle_selected_attempt(
    *,
    args: argparse.Namespace,
    task: Any,
    output_dir: Path,
    round_dir: Path,
    attempt_index: int,
    selected_rounds: int,
    current_checkpoint: str,
    init_final_accuracy: float,
    source_sizes: set[int],
    source_examples: list[Any],
    exclude_keys: set[Any],
    proposal_trace_buffer: list[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_grpo_update_count: int,
    deleted_replaced_model_dirs: list[str],
    summary_records: list[JsonDict],
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    metrics: Sequence[CandidateMetrics],
    work_items: Sequence[CandidateWorkItem],
    selected: CandidateMetrics,
    selected_payload: JsonDict,
    trace_rows: Sequence[Mapping[str, Any]],
    outcome_traces: Sequence[Any],
    checkpoint_manager: Any,
    deps: AttemptOutcomeDeps,
) -> AttemptOutcomeResult:
    selected_item = next(item for item in work_items if item.index == selected.index)
    selected_rounds += 1
    selected_trace = deps.build_selected_proposal_trace_example(
        task_name=args.task,
        condition=args.condition,
        round_index=selected_rounds,
        prompt=prompt,
        selected_item=selected_item,
        selected=selected,
    )
    proposal_trace_buffer.append(selected_trace)
    deps.write_trace_jsonl(round_dir / "selected_proposal_trace.jsonl", [selected_trace.to_json_dict()])
    deps.write_trace_jsonl(
        output_dir / "selected_proposal_trace_buffer.jsonl",
        [trace.to_json_dict() for trace in proposal_trace_buffer],
    )
    source_examples.extend(selected_item.pseudo_examples)
    exclude_keys.update(selected_item.composed.keys)
    source_sizes.add(selected.proposal.target)
    previous_checkpoint = current_checkpoint
    current_checkpoint = str(selected.model_dir)
    deleted_replaced_model_dirs.extend(
        checkpoint_manager.cleanup_replaced_checkpoint(
            old_checkpoint=previous_checkpoint,
            new_checkpoint=current_checkpoint,
        )
    )
    current_final_accuracy = selected.final_accuracy
    current_per_size_accuracy = selected.per_size_accuracy
    proposal_grpo_metrics = None
    if selected_rounds < args.num_rounds and attempt_index < args.max_attempt_rounds:
        previous_checkpoint = current_checkpoint
        next_checkpoint, proposal_grpo_metrics = deps.apply_or_dispatch_proposal_grpo_update(
            args=args,
            source_checkpoint=current_checkpoint,
            output_dir=round_dir / "proposal_grpo",
            prompt=prompt,
            proposal_results=proposal_results,
            candidate_metrics=metrics,
            seed=args.seed + attempt_index * 1543,
        )
        if not proposal_grpo_metrics.get("skipped", True):
            proposal_grpo_update_count += 1
        deleted_checkpoints = checkpoint_manager.cleanup_replaced_checkpoint(
            old_checkpoint=previous_checkpoint,
            new_checkpoint=next_checkpoint,
        )
        deleted_replaced_model_dirs.extend(deleted_checkpoints)
        if deleted_checkpoints:
            proposal_grpo_metrics["deleted_replaced_model_dirs"] = deleted_checkpoints
            deps.write_json(round_dir / "proposal_grpo" / "proposal_grpo_metrics.json", proposal_grpo_metrics)
        current_checkpoint = next_checkpoint
    if deleted_replaced_model_dirs:
        deps.write_json(round_dir / "deleted_replaced_model_dirs.json", deleted_replaced_model_dirs)
    deps.save_examples(
        round_dir / "selected_pseudo_examples.jsonl",
        selected_item.pseudo_examples,
        task.serialize_example,
    )
    deps.write_json(
        round_dir / "round_summary.json",
        {
            "attempt": attempt_index,
            "selected_round": selected_rounds,
            "selected": selected_payload,
            "source_sizes_after": sorted(source_sizes),
            "source_example_count_after": len(source_examples),
            "current_checkpoint": current_checkpoint,
            "trace_count": len(trace_rows),
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "outcome_trace_count": len(outcome_traces),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "proposal_grpo": proposal_grpo_metrics,
            "deleted_replaced_model_dirs": deleted_replaced_model_dirs,
        },
    )
    summary_records.append(
        {
            "attempt": attempt_index,
            "selected_round": selected_rounds,
            "selected": selected_payload,
            "candidate_count": len(metrics),
            "trace_count": len(trace_rows),
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "outcome_trace_count": len(outcome_traces),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "source_sizes": sorted(source_sizes),
            "source_example_count": len(source_examples),
            "current_checkpoint": current_checkpoint,
            "proposal_grpo": proposal_grpo_metrics,
            "deleted_replaced_model_dirs": deleted_replaced_model_dirs,
        }
    )
    deps.write_json(output_dir / "adaptive_candidate_training_results.json", summary_records)
    return AttemptOutcomeResult(
        selected_rounds=selected_rounds,
        consecutive_no_selection=0,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        proposal_grpo_update_count=proposal_grpo_update_count,
        should_break=False,
    )


__all__ = ["handle_selected_attempt"]

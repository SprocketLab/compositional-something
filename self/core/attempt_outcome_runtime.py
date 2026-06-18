"""Selected/no-selection attempt outcome handling for adaptive runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AttemptOutcomeDeps:
    build_round_outcome_trace_examples: Callable[..., list[Any]]
    build_selected_proposal_trace_example: Callable[..., Any]
    apply_or_dispatch_proposal_grpo_update: Callable[..., tuple[str, JsonDict]]
    write_json: Callable[[Path, Any], None]
    write_trace_jsonl: Callable[[Path, Sequence[Mapping[str, Any]]], None]
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]


@dataclass(frozen=True)
class AttemptOutcomeResult:
    selected_rounds: int
    consecutive_no_selection: int
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    proposal_grpo_update_count: int
    should_break: bool = False


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


def _handle_no_selection_attempt(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    round_dir: Path,
    attempt_index: int,
    selected_round_for_prompt: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    current_checkpoint: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_grpo_update_count: int,
    deleted_replaced_model_dirs: list[str],
    summary_records: list[JsonDict],
    source_sizes: set[int],
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    metrics: Sequence[CandidateMetrics],
    trace_rows: Sequence[Mapping[str, Any]],
    outcome_traces: Sequence[Any],
    checkpoint_manager: Any,
    deps: AttemptOutcomeDeps,
) -> AttemptOutcomeResult:
    print(
        f"[WARN] Attempt {attempt_index}: no candidate reached "
        f"selection_min_reward={args.selection_min_reward}; continuing.",
        flush=True,
    )
    consecutive_no_selection += 1
    proposal_grpo_metrics: Optional[JsonDict] = None
    will_continue = (
        consecutive_no_selection < args.no_selection_patience
        and attempt_index < args.max_attempt_rounds
        and selected_rounds < args.num_rounds
    )
    if will_continue:
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
    failure_record = {
        "attempt": attempt_index,
        "selected_round": selected_round_for_prompt,
        "selected": None,
        "candidate_count": len(metrics),
        "trace_count": len(trace_rows),
        "no_selection": True,
        "consecutive_no_selection": consecutive_no_selection,
        "proposal_trace_buffer_size": len(proposal_trace_buffer),
        "outcome_trace_count": len(outcome_traces),
        "outcome_trace_buffer_size": len(outcome_trace_buffer),
        "source_sizes": sorted(source_sizes),
        "current_checkpoint": current_checkpoint,
        "proposal_grpo": proposal_grpo_metrics,
        "deleted_replaced_model_dirs": deleted_replaced_model_dirs,
    }
    summary_records.append(failure_record)
    deps.write_json(
        round_dir / "attempt_summary.json",
        {
            **failure_record,
            "candidate_metrics_path": str(round_dir / "candidate_metrics.json"),
            "proposal_results_path": str(round_dir / "proposal_results.json"),
        },
    )
    deps.write_json(output_dir / "adaptive_candidate_training_results.json", summary_records)
    should_break = consecutive_no_selection >= args.no_selection_patience
    if should_break:
        print(
            f"[WARN] Reached no_selection_patience={args.no_selection_patience}; stopping.",
            flush=True,
        )
    return AttemptOutcomeResult(
        selected_rounds=selected_rounds,
        consecutive_no_selection=consecutive_no_selection,
        current_checkpoint=current_checkpoint,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        proposal_grpo_update_count=proposal_grpo_update_count,
        should_break=should_break,
    )


def _handle_selected_attempt(
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

"""No-selection attempt outcome handling for adaptive runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from self.adaptive.attempts.attempt_models import AttemptOutcomeDeps, AttemptOutcomeResult, JsonDict
from self.core.models import CandidateMetrics
from self.adaptive.proposals.proposal_prompts import PromptBundle


def handle_no_selection_attempt(
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


__all__ = ["handle_no_selection_attempt"]

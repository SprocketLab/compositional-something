"""Adaptive run summary and plan-log finalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


WriteJsonFn = Callable[[Path, Any], None]
AppendPlanLogFn = Callable[[Path, Iterable[str]], None]
SanitizeJsonFn = Callable[[Any], Any]


def finalize_adaptive_run(
    *,
    args: Any,
    output_dir: Path,
    summary_records: Sequence[Mapping[str, Any]],
    selected_rounds: int,
    attempt_index: int,
    current_checkpoint: str,
    source_sizes: set[int],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_grpo_update_count: int,
    init_final_accuracy: float,
    write_json: WriteJsonFn,
    append_plan_log: AppendPlanLogFn,
    sanitize_json_value: SanitizeJsonFn,
) -> Any:
    """Write final adaptive artifacts and return the sanitized summary."""

    results_path = output_dir / "adaptive_candidate_training_results.json"
    write_json(results_path, summary_records)
    append_plan_log(
        args.plan_log_path,
        [
            "Implemented/running adaptive candidate-training loop.",
            (
                f"Task: `{args.task}`; selected rounds requested: {args.num_rounds}; "
                f"attempts used: {attempt_index}; candidates per attempt: {args.num_candidates}."
            ),
            f"Output directory: `{output_dir}`.",
            f"Proposal output schema: `{args.proposal_output_schema}`.",
            f"Final source sizes tracked by driver: `{sorted(source_sizes)}`.",
            f"Selected proposal traces retained for replay: `{len(proposal_trace_buffer)}`.",
            (
                "Post-task proposal rehearsal: "
                f"`{args.post_task_proposal_rehearsal}`; repeat/max examples: "
                f"`{args.post_task_proposal_rehearsal_repeat_count}`/`{args.post_task_proposal_rehearsal_max_examples}`."
            ),
            f"Outcome trace target mode: `{args.outcome_trace_target_mode}`; retained outcome traces: `{len(outcome_trace_buffer)}`.",
            (
                f"Proposal GRPO updates: `{proposal_grpo_update_count}`; "
                f"steps/update: `{args.proposal_grpo_steps}`; reward mode: `{args.proposal_grpo_reward_mode}`; "
                f"zero-variance mode: `{args.proposal_grpo_zero_variance}`."
            ),
            f"Keep all proposal-GRPO checkpoints: `{args.keep_all_proposal_grpo_checkpoints}`.",
        ],
    )
    final_summary = {
        "task": args.task,
        "condition": args.condition,
        "output_dir": str(output_dir),
        "rounds_recorded": len(summary_records),
        "selected_rounds_completed": selected_rounds,
        "attempts_completed": attempt_index,
        "target_selected_rounds": args.num_rounds,
        "max_attempt_rounds": args.max_attempt_rounds,
        "no_selection_patience": args.no_selection_patience,
        "num_candidates": args.num_candidates,
        "current_checkpoint": current_checkpoint,
        "source_sizes": sorted(source_sizes),
        "proposal_trace_buffer_size": len(proposal_trace_buffer),
        "proposal_output_schema": args.proposal_output_schema,
        "proposal_trace_buffer_path": str(output_dir / "selected_proposal_trace_buffer.jsonl"),
        "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
        "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
        "post_task_proposal_rehearsal": args.post_task_proposal_rehearsal,
        "post_task_proposal_rehearsal_repeat_count": args.post_task_proposal_rehearsal_repeat_count,
        "post_task_proposal_rehearsal_max_examples": args.post_task_proposal_rehearsal_max_examples,
        "outcome_trace_target_mode": args.outcome_trace_target_mode,
        "outcome_trace_buffer_size": len(outcome_trace_buffer),
        "outcome_trace_buffer_path": str(output_dir / "outcome_trace_buffer.jsonl"),
        "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
        "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
        "proposal_grpo_update_count": proposal_grpo_update_count,
        "proposal_grpo_steps": args.proposal_grpo_steps,
        "proposal_grpo_learning_rate": args.proposal_grpo_learning_rate,
        "proposal_grpo_kl_coef": args.proposal_grpo_kl_coef,
        "proposal_grpo_zero_variance": args.proposal_grpo_zero_variance,
        "proposal_grpo_reward_mode": args.proposal_grpo_reward_mode,
        "proposal_grpo_outcome_scale": args.proposal_grpo_outcome_scale,
        "proposal_grpo_fixed_baseline": args.proposal_grpo_fixed_baseline,
        "keep_all_proposal_grpo_checkpoints": args.keep_all_proposal_grpo_checkpoints,
        "init_final_accuracy": init_final_accuracy,
        "results_path": str(results_path),
    }
    write_json(output_dir / "summary.json", final_summary)
    return sanitize_json_value(final_summary)

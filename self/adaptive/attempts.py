#!/usr/bin/env python3
"""Adaptive attempt prompt, loop, and outcome runtime."""

from __future__ import annotations

# --- from attempts.py ---
import argparse
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from self.adaptive.proposal import DEFAULT_CONFIG_SEARCH_SPACES, ConfigProposal
from self.adaptive.proposal import choose_default_program_pair
from self.adaptive.proposal import (
    PromptBundle,
    render_config_prompt,
    render_program_candidate_prompt,
)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AttemptPromptDeps:
    choose_default_program_pair: Callable[..., ConfigProposal]
    render_config_prompt: Callable[..., PromptBundle]
    render_program_candidate_prompt: Callable[..., PromptBundle]


@dataclass(frozen=True)
class AttemptPromptResult:
    prompt: PromptBundle
    default_program_pair: Optional[ConfigProposal]
    aggregate_metrics: JsonDict
    current_source: JsonDict
    allowed_target_frontier: JsonDict


DEFAULT_ATTEMPT_PROMPT_DEPS = AttemptPromptDeps(
    choose_default_program_pair=choose_default_program_pair,
    render_config_prompt=render_config_prompt,
    render_program_candidate_prompt=render_program_candidate_prompt,
)


def build_attempt_prompt(
    *,
    args: argparse.Namespace,
    current_checkpoint: str,
    current_final_accuracy: float,
    init_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    proposal_trace_buffer: Optional[Sequence[Any]] = None,
    extra_aggregate_metrics: Optional[Mapping[str, Any]] = None,
    deps: AttemptPromptDeps = DEFAULT_ATTEMPT_PROMPT_DEPS,
) -> AttemptPromptResult:
    frontier_min = args.frontier_min_size
    frontier_max = args.frontier_max_size
    aggregate_metrics: JsonDict = {
        "current_avg_accuracy": current_final_accuracy,
        "init_avg_accuracy": init_final_accuracy,
        "per_size_accuracy": {str(size): score for size, score in current_per_size_accuracy.items()},
        "source_sizes": sorted(source_sizes),
        "attempt_index": attempt_index,
        "selected_rounds_completed": selected_rounds,
        "consecutive_no_selection": consecutive_no_selection,
        "reward_formula": "candidate_avg_accuracy - current_avg_accuracy",
    }
    source_admission_threshold = getattr(args, "source_admission_target_accuracy_threshold", None)
    if source_admission_threshold is not None:
        aggregate_metrics["source_admission_target_accuracy_threshold"] = source_admission_threshold
    aggregate_metrics.update(_proposal_prompt_extra_metrics(args, proposal_trace_buffer or ()))
    if extra_aggregate_metrics:
        aggregate_metrics.update(dict(extra_aggregate_metrics))

    current_source_payload: JsonDict = {
        "sizes": sorted(source_sizes),
        "min": min(source_sizes),
        "max": max(source_sizes),
    }
    allowed_frontier_payload: JsonDict = {
        "min": frontier_min,
        "max": frontier_max,
    }
    default_program_pair: Optional[ConfigProposal] = None
    if args.condition == "program":
        default_program_pair = deps.choose_default_program_pair(
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
            allow_repeat_targets=args.allow_repeat_targets,
        )
        aggregate_metrics["driver_selected_program_pair"] = default_program_pair.to_json_dict()

    if args.condition == "config":
        prompt = deps.render_config_prompt(
            task_name=args.task,
            round_index=selected_round_for_prompt,
            current_source=current_source_payload,
            allowed_target_frontier=allowed_frontier_payload,
            aggregate_metrics=aggregate_metrics,
            guard_choices=DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"],
            model_name=current_checkpoint,
            proposal_output_schema=args.proposal_output_schema,
        )
    else:
        prompt = deps.render_program_candidate_prompt(
            args=args,
            round_index=selected_round_for_prompt,
            current_checkpoint=current_checkpoint,
            current_source=current_source_payload,
            allowed_target_frontier=allowed_frontier_payload,
            aggregate_metrics=aggregate_metrics,
            default_pair=default_program_pair,
        )

    return AttemptPromptResult(
        prompt=prompt,
        default_program_pair=default_program_pair,
        aggregate_metrics=aggregate_metrics,
        current_source=current_source_payload,
        allowed_target_frontier=allowed_frontier_payload,
    )


def _round_prompt_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, 4)


def _trace_payload(trace: Any) -> JsonDict:
    if isinstance(trace, Mapping):
        return dict(trace)
    to_json_dict = getattr(trace, "to_json_dict", None)
    if callable(to_json_dict):
        payload = to_json_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    metadata = getattr(trace, "metadata", {})
    return {
        "round": getattr(trace, "round_index", None),
        "reward": getattr(trace, "reward", None),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }


def _compact_selected_action_history(
    proposal_trace_buffer: Sequence[Any],
    *,
    max_items: int,
) -> List[JsonDict]:
    if max_items <= 0:
        return []
    rows: List[JsonDict] = []
    for trace in list(proposal_trace_buffer)[-max_items:]:
        payload = _trace_payload(trace)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        row: JsonDict = {}
        for output_key, source_key in (
            ("selected_round", "round"),
            ("left", "left"),
            ("right", "right"),
            ("target", "target"),
            ("guard", "guard"),
        ):
            source = payload if source_key == "round" else metadata
            value = source.get(source_key) if isinstance(source, Mapping) else None
            if value is not None:
                row[output_key] = value
        reward = _round_prompt_float(payload.get("reward"))
        if reward is not None:
            row["reward"] = reward
        for output_key, source_key in (
            ("avg_delta_from_current", "final_accuracy_delta_from_current"),
            ("target_accuracy", "target_accuracy"),
            ("target_delta", "target_delta"),
        ):
            value = _round_prompt_float(metadata.get(source_key))
            if value is not None:
                row[output_key] = value
        if {"left", "right", "target", "guard"} & set(row):
            rows.append(row)
    return rows


def _proposal_prompt_extra_metrics(args: argparse.Namespace, proposal_trace_buffer: Sequence[Any]) -> JsonDict:
    if not getattr(args, "proposal_prompt_action_history", False):
        return {}
    max_items = int(getattr(args, "proposal_prompt_action_history_max_items", 5))
    return {
        "recent_selected_actions": _compact_selected_action_history(
            proposal_trace_buffer,
            max_items=max_items,
        ),
        "recent_selected_actions_policy": (
            "Use this only to recognize exact repeats; repeat an action only when current diagnostics justify it."
        ),
    }


def _compact_candidate_action(metric: Any, *, attempt_index: int, selected_index: Optional[int]) -> JsonDict:
    proposal = metric.proposal
    action: JsonDict = {
        "attempt": int(attempt_index),
        "left": int(proposal.left),
        "right": int(proposal.right),
        "target": int(proposal.target),
        "guard": str(proposal.guard),
        "selected": selected_index is not None and int(metric.index) == int(selected_index),
        "valid": bool(metric.valid),
        "reward": _round_prompt_float(metric.reward),
        "frontier_delta": _round_prompt_float(metric.frontier_delta),
    }
    if not metric.valid and metric.failure_reason:
        action["failure"] = str(metric.failure_reason)[:80]
    return action


def _compact_attempt_actions(
    *,
    attempt_index: int,
    metrics: Sequence[Any],
    selected: Optional[Any],
) -> List[JsonDict]:
    selected_index = int(selected.index) if selected is not None else None

    def sort_key(metric: Any) -> tuple[int, float, int]:
        reward = _round_prompt_float(metric.reward)
        reward_value = reward if reward is not None else float("-inf")
        return (1 if selected_index is not None and int(metric.index) == selected_index else 0, reward_value, -int(metric.index))

    ordered = sorted(metrics, key=sort_key, reverse=True)
    return [
        _compact_candidate_action(metric, attempt_index=attempt_index, selected_index=selected_index)
        for metric in ordered
    ]


def _max_selected_rounds(args: argparse.Namespace) -> int:
    if hasattr(args, "max_selected_rounds"):
        return int(getattr(args, "max_selected_rounds") or 0)
    legacy_num_rounds = getattr(args, "num_rounds", 0)
    return int(legacy_num_rounds or 0)


def _source_admission_decision(args: argparse.Namespace, selected: Any) -> JsonDict:
    threshold = float(getattr(args, "source_admission_target_accuracy_threshold", 0.80))
    try:
        target_accuracy = float(selected.target_accuracy)
    except (TypeError, ValueError):
        target_accuracy = math.nan
    admitted = bool(math.isfinite(target_accuracy) and target_accuracy >= threshold)
    return {
        "admitted": admitted,
        "target": int(selected.proposal.target),
        "target_accuracy": target_accuracy,
        "threshold": threshold,
        "reason": (
            "target_accuracy_clears_threshold"
            if admitted
            else "target_accuracy_below_threshold"
        ),
    }


def _selected_cap_reached(args: argparse.Namespace, selected_rounds: int) -> bool:
    max_selected_rounds = _max_selected_rounds(args)
    return max_selected_rounds > 0 and selected_rounds >= max_selected_rounds


def _selected_progress_label(args: argparse.Namespace, selected_rounds: int) -> str:
    max_selected_rounds = _max_selected_rounds(args)
    if max_selected_rounds > 0:
        return f"{selected_rounds}/{max_selected_rounds}"
    return str(selected_rounds)


# --- from attempts.py ---
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from self.adaptive.proposal import ConfigProposal


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class DryRunAttemptDeps:
    load_fixture_proposals: Callable[[Path], list[Mapping[str, Any]]]
    rows_for_round: Callable[..., list[Mapping[str, Any]]]
    validate_proposal_rows: Callable[..., list[JsonDict]]
    build_candidate_work_items: Callable[..., list[Any]]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class DryRunAttemptResult:
    selected_rounds: int
    consecutive_no_selection: int
    should_stop: bool


def run_dry_attempt(
    *,
    args: argparse.Namespace,
    task: Any,
    round_dir: Path,
    output_dir: Path,
    selected_round_for_prompt: int,
    attempt_index: int,
    source_sizes: set[int],
    default_program_pair: Optional[ConfigProposal],
    source_examples: Sequence[Any],
    exclude_keys: set[Any],
    rng: random.Random,
    summary_records: list[JsonDict],
    selected_rounds: int,
    consecutive_no_selection: int,
    deps: DryRunAttemptDeps,
) -> DryRunAttemptResult:
    rows = (
        deps.rows_for_round(
            deps.load_fixture_proposals(args.proposal_fixture_jsonl),
            selected_round_for_prompt,
            attempt_index=attempt_index,
        )
        if args.proposal_fixture_jsonl
        else []
    )
    proposal_results = deps.validate_proposal_rows(
        rows=rows[: args.num_candidates],
        args=args,
        source_sizes=source_sizes,
        frontier_min=args.frontier_min_size,
        frontier_max=args.frontier_max_size,
        default_pair=default_program_pair,
    )
    deps.write_json(round_dir / "proposal_results.json", proposal_results)
    work_items = deps.build_candidate_work_items(
        args=args,
        task=task,
        round_dir=round_dir,
        proposal_results=proposal_results,
        source_examples=source_examples,
        exclude_keys=exclude_keys,
        rng=rng,
    )
    deps.write_json(
        round_dir / "dry_run_summary.json",
        {
            "work_items": len(work_items),
            "attempt_index": attempt_index,
            "selected_round": selected_round_for_prompt,
        },
    )
    summary_records.append(
        {
            "attempt": attempt_index,
            "selected_round": selected_round_for_prompt if work_items else None,
            "selected": None,
            "dry_run_data_only": True,
            "work_items": len(work_items),
            "source_sizes": sorted(source_sizes),
        }
    )
    deps.write_json(output_dir / "adaptive_candidate_training_results.json", summary_records)
    if work_items:
        return DryRunAttemptResult(
            selected_rounds=selected_rounds + 1,
            consecutive_no_selection=0,
            should_stop=False,
        )

    next_consecutive_no_selection = consecutive_no_selection + 1
    should_stop = next_consecutive_no_selection >= args.no_selection_patience
    if should_stop:
        print(
            f"[WARN] Reached no_selection_patience={args.no_selection_patience}; stopping dry run.",
            flush=True,
        )
    return DryRunAttemptResult(
        selected_rounds=selected_rounds,
        consecutive_no_selection=next_consecutive_no_selection,
        should_stop=should_stop,
    )


# --- from attempts.py ---
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Sequence

from self.core.models import CandidateMetrics

if TYPE_CHECKING:
    from self.adaptive.run import RoundModelDispatchDeps, RoundModelDispatchResult


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


@dataclass(frozen=True)
class CandidateAttemptDeps:
    run_round_model_dispatch: Callable[..., RoundModelDispatchResult]
    train_candidate_metrics: Callable[..., Sequence[CandidateMetrics]]
    select_candidate: Callable[[Sequence[CandidateMetrics], float], CandidateMetrics | None]
    write_round_trace: Callable[..., Sequence[Mapping[str, Any]]]
    handle_attempt_outcome: Callable[..., AttemptOutcomeResult]
    round_model_dispatch_deps: RoundModelDispatchDeps
    attempt_outcome_deps: AttemptOutcomeDeps


@dataclass(frozen=True)
class AttemptLoopDeps:
    ensure_dir: Callable[[Path], None]
    build_attempt_prompt: Callable[..., AttemptPromptResult]
    write_json: Callable[[Path, Any], None]
    run_dry_attempt: Callable[..., DryRunAttemptResult]
    run_round_model_dispatch: Callable[..., RoundModelDispatchResult]
    train_candidate_metrics: Callable[..., Sequence[CandidateMetrics]]
    select_candidate: Callable[[Sequence[CandidateMetrics], float], CandidateMetrics | None]
    write_round_trace: Callable[..., Sequence[Mapping[str, Any]]]
    handle_attempt_outcome: Callable[..., AttemptOutcomeResult]
    attempt_prompt_deps: AttemptPromptDeps
    dry_run_attempt_deps: DryRunAttemptDeps
    round_model_dispatch_deps: RoundModelDispatchDeps
    attempt_outcome_deps: AttemptOutcomeDeps


@dataclass(frozen=True)
class AttemptLoopResult:
    selected_rounds: int
    attempt_index: int
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    source_sizes: set[int]
    proposal_trace_buffer: list[Any]
    outcome_trace_buffer: list[Any]
    proposal_grpo_update_count: int


__all__ = [
    "AttemptLoopDeps",
    "AttemptLoopResult",
    "AttemptOutcomeDeps",
    "AttemptOutcomeResult",
    "CandidateAttemptDeps",
    "JsonDict",
]


# --- from attempt_no_selection_runtime.py ---
import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from self.core.models import CandidateMetrics
from self.adaptive.proposal import PromptBundle


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
        and not _selected_cap_reached(args, selected_rounds)
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
            proposal_trace_buffer=proposal_trace_buffer,
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
    attempt_actions = _compact_attempt_actions(
        attempt_index=attempt_index,
        metrics=metrics,
        selected=None,
    )
    failure_record = {
        "attempt": attempt_index,
        "selected_round": selected_round_for_prompt,
        "selected": None,
        "candidate_count": len(metrics),
        "attempt_actions": attempt_actions,
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


# --- from attempt_selected_runtime.py ---
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle


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
    proposal_trace_history_for_grpo = list(proposal_trace_buffer)
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
    source_admission = _source_admission_decision(args, selected)
    if source_admission["admitted"]:
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
    if not _selected_cap_reached(args, selected_rounds) and attempt_index < args.max_attempt_rounds:
        previous_checkpoint = current_checkpoint
        next_checkpoint, proposal_grpo_metrics = deps.apply_or_dispatch_proposal_grpo_update(
            args=args,
            source_checkpoint=current_checkpoint,
            output_dir=round_dir / "proposal_grpo",
            prompt=prompt,
            proposal_results=proposal_results,
            candidate_metrics=metrics,
            proposal_trace_buffer=proposal_trace_history_for_grpo,
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
            "attempt_actions": _compact_attempt_actions(
                attempt_index=attempt_index,
                metrics=metrics,
                selected=selected,
            ),
            "source_sizes_after": sorted(source_sizes),
            "source_example_count_after": len(source_examples),
            "source_admission": source_admission,
            "current_checkpoint": current_checkpoint,
            "trace_count": len(trace_rows),
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "outcome_trace_count": len(outcome_traces),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "proposal_grpo": proposal_grpo_metrics,
            "deleted_replaced_model_dirs": deleted_replaced_model_dirs,
        },
    )
    attempt_actions = _compact_attempt_actions(
        attempt_index=attempt_index,
        metrics=metrics,
        selected=selected,
    )
    summary_records.append(
        {
            "attempt": attempt_index,
            "selected_round": selected_rounds,
            "selected": selected_payload,
            "candidate_count": len(metrics),
            "attempt_actions": attempt_actions,
            "trace_count": len(trace_rows),
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "outcome_trace_count": len(outcome_traces),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "source_sizes": sorted(source_sizes),
            "source_example_count": len(source_examples),
            "source_admission": source_admission,
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


# --- from attempts.py ---
import argparse
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle


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
        return handle_no_selection_attempt(
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

    return handle_selected_attempt(
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


# --- from attempts.py ---
import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence


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
    attempt_start = time.monotonic()
    phase_timings: dict[str, float | int | None] = {
        "attempt": int(attempt_index),
        "selected_rounds_before": int(selected_rounds),
    }
    deleted_replaced_model_dirs: list[str] = []
    phase_start = time.monotonic()
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
        extra_aggregate_metrics=_proposal_prompt_extra_metrics(args, proposal_trace_buffer),
        deps=deps.round_model_dispatch_deps,
    )
    phase_timings["round_model_dispatch_seconds"] = time.monotonic() - phase_start
    current_final_accuracy = round_result.current_final_accuracy
    current_per_size_accuracy = round_result.current_per_size_accuracy
    prompt = round_result.prompt
    proposal_results = round_result.proposal_results
    work_items = round_result.work_items
    phase_timings["proposal_count"] = len(proposal_results)
    phase_timings["candidate_work_item_count"] = len(work_items)

    phase_start = time.monotonic()
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
    phase_timings["candidate_dispatch_seconds"] = time.monotonic() - phase_start
    phase_timings["candidate_metric_count"] = len(metrics)
    phase_start = time.monotonic()
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
    deleted_unselected_model_dirs = checkpoint_manager.cleanup_unselected_candidates(metrics=metrics, selected=selected)
    phase_timings["deleted_unselected_model_dir_count"] = len(deleted_unselected_model_dirs)
    if deleted_unselected_model_dirs:
        write_json = getattr(deps.attempt_outcome_deps, "write_json", None)
        if callable(write_json):
            write_json(round_dir / "deleted_unselected_model_dirs.json", deleted_unselected_model_dirs)
    phase_timings["selection_trace_cleanup_seconds"] = time.monotonic() - phase_start
    phase_timings["selected_candidate_index"] = int(selected.index) if selected is not None else None
    phase_start = time.monotonic()
    outcome_result = deps.handle_attempt_outcome(
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
    phase_timings["attempt_outcome_seconds"] = time.monotonic() - phase_start
    phase_timings["total_attempt_seconds"] = time.monotonic() - attempt_start
    write_json = getattr(deps.attempt_outcome_deps, "write_json", None)
    if callable(write_json):
        write_json(round_dir / "attempt_timing.json", phase_timings)
    return outcome_result


__all__ = ["CandidateAttemptDeps", "run_candidate_attempt"]


# --- from attempts.py ---
import argparse
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence


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

    while attempt_index < args.max_attempt_rounds and not _selected_cap_reached(args, selected_rounds):
        attempt_index += 1
        selected_round_for_prompt = selected_rounds + 1
        round_dir = output_dir / f"attempt_{attempt_index:04d}"
        deps.ensure_dir(round_dir)
        print(
            f"[INFO] Starting adaptive candidate attempt {attempt_index} "
            f"(selected_count={_selected_progress_label(args, selected_rounds)})",
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
            proposal_trace_buffer=proposal_trace_buffer,
            extra_aggregate_metrics={},
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

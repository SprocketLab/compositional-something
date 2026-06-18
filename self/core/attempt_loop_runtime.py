"""Adaptive selected-round attempt loop orchestration."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.core.attempt_outcome_runtime import AttemptOutcomeDeps, AttemptOutcomeResult
from self.core.attempt_prompt_runtime import AttemptPromptDeps, AttemptPromptResult
from self.core.dry_run_runtime import DryRunAttemptDeps, DryRunAttemptResult
from self.core.models import CandidateMetrics
from self.core.round_model_dispatch_runtime import RoundModelDispatchDeps, RoundModelDispatchResult
from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


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

    while selected_rounds < args.num_rounds and attempt_index < args.max_attempt_rounds:
        attempt_index += 1
        selected_round_for_prompt = selected_rounds + 1
        round_dir = output_dir / f"attempt_{attempt_index:04d}"
        deps.ensure_dir(round_dir)
        deleted_replaced_model_dirs: list[str] = []
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

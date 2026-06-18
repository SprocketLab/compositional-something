"""Adaptive attempt-loop dependency and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.attempts.attempt_outcome_models import AttemptOutcomeDeps, AttemptOutcomeResult
from self.adaptive.attempts.attempt_prompt_runtime import AttemptPromptDeps, AttemptPromptResult
from self.adaptive.attempts.dry_run_runtime import DryRunAttemptDeps, DryRunAttemptResult
from self.core.models import CandidateMetrics
from self.adaptive.run.round_model_dispatch_runtime import RoundModelDispatchDeps, RoundModelDispatchResult


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

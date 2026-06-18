"""Compatibility-aware candidate-dispatch wiring for adaptive runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.core import candidate_dispatch_runtime
from self.core.experience_trace_models import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposals import PromptBundle
from self.core.training import TrainingConfig


@dataclass(frozen=True)
class CandidateDispatchEntrypointDeps:
    train_and_score_candidate: Callable[..., CandidateMetrics]
    candidate_failure_metrics: Callable[..., CandidateMetrics]
    collect_candidate_array_metrics: Callable[..., List[CandidateMetrics]]
    train_candidates_serial: Callable[..., List[CandidateMetrics]]
    train_candidates_local_parallel: Callable[..., List[CandidateMetrics]]
    train_candidates_slurm_array: Callable[..., List[CandidateMetrics]]
    subprocess_module: Any


def candidate_failure_metrics(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    return candidate_dispatch_runtime.candidate_failure_metrics(
        item=item,
        reason=reason,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def train_candidates_serial(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return candidate_dispatch_runtime.train_candidates_serial(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        config=config,
        attempt_index=attempt_index,
        score_candidate_fn=deps.train_and_score_candidate,
    )


def collect_candidate_array_metrics(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return candidate_dispatch_runtime.collect_candidate_array_metrics(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=deps.candidate_failure_metrics,
    )


def train_candidates_slurm_array(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return candidate_dispatch_runtime.train_candidates_slurm_array(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
        collect_metrics_fn=deps.collect_candidate_array_metrics,
    )


def train_candidates_local_parallel(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return candidate_dispatch_runtime.train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
        collect_metrics_fn=deps.collect_candidate_array_metrics,
        subprocess_module=deps.subprocess_module,
    )


def train_candidate_metrics(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return candidate_dispatch_runtime.train_candidate_metrics(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        config=config,
        attempt_index=attempt_index,
        serial_fn=deps.train_candidates_serial,
        local_parallel_fn=deps.train_candidates_local_parallel,
        slurm_array_fn=deps.train_candidates_slurm_array,
    )

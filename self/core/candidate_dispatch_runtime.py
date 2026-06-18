"""Candidate training dispatch wrappers."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.core import candidate_workers
from self.core.candidate_metric_collection import (
    candidate_failure_metrics as _candidate_failure_metrics,
    collect_candidate_array_metrics as _collect_candidate_array_metrics,
)
from self.core.experience_trace_models import OutcomeTraceExample, ProposalTraceExample
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposal_prompts import PromptBundle
from self.core.training import TrainingConfig


def candidate_failure_metrics(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    return _candidate_failure_metrics(
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
    score_candidate_fn: Callable[..., CandidateMetrics],
) -> list[CandidateMetrics]:
    metrics: list[CandidateMetrics] = []
    model_bootstrap_cache = ModelBootstrapCache(
        cache_base_state=bool(getattr(args, "candidate_local_cache_base_state", False))
    )
    pass_model_bootstrap_cache = _score_candidate_accepts_model_bootstrap_cache(score_candidate_fn)
    for item in work_items:
        kwargs = dict(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            source_examples=source_examples,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_prompt=proposal_prompt,
            round_index=round_index,
            item=item,
            round_dir=round_dir,
            eval_examples=eval_examples,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            config=config,
            seed=args.seed + attempt_index * 1009 + item.index,
        )
        if pass_model_bootstrap_cache:
            kwargs["model_bootstrap_cache"] = model_bootstrap_cache
        metrics.append(
            score_candidate_fn(**kwargs)
        )
    return metrics


def _score_candidate_accepts_model_bootstrap_cache(score_candidate_fn: Callable[..., CandidateMetrics]) -> bool:
    try:
        signature = inspect.signature(score_candidate_fn)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "model_bootstrap_cache":
            return True
    return False


def collect_candidate_array_metrics(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    failure_metrics_fn: Callable[..., CandidateMetrics] | None = None,
) -> list[CandidateMetrics]:
    return _collect_candidate_array_metrics(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=failure_metrics_fn,
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
    collect_metrics_fn: Callable[..., list[CandidateMetrics]],
) -> list[CandidateMetrics]:
    return candidate_workers.train_candidates_slurm_array(
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
        collect_metrics_fn=collect_metrics_fn,
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
    collect_metrics_fn: Callable[..., list[CandidateMetrics]],
    subprocess_module: Any = None,
) -> list[CandidateMetrics]:
    if subprocess_module is not None:
        candidate_workers.subprocess = subprocess_module
    return candidate_workers.train_candidates_local_parallel(
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
        collect_metrics_fn=collect_metrics_fn,
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
    serial_fn: Callable[..., list[CandidateMetrics]],
    local_parallel_fn: Callable[..., list[CandidateMetrics]],
    slurm_array_fn: Callable[..., list[CandidateMetrics]],
) -> list[CandidateMetrics]:
    if args.candidate_execution_mode == "serial":
        return serial_fn(
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
        )
    if args.candidate_execution_mode == "local_parallel":
        return local_parallel_fn(
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
        )
    if args.candidate_execution_mode == "slurm_array":
        return slurm_array_fn(
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
        )
    raise ValueError(f"Unsupported candidate_execution_mode={args.candidate_execution_mode!r}.")

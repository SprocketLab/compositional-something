#!/usr/bin/env python3
"""Candidate execution and worker-result aggregation helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from self.core import candidate_workers
from self.core.candidate_metric_collection import (
    candidate_failure_metrics,
    collect_candidate_array_metrics,
)
from self.core.candidate_worker_payloads import (
    work_item_from_worker_payload,
    work_item_to_worker_payload,
)
from self.core.experience_traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposals import PromptBundle
from self.core.training import TrainingConfig


ScoreCandidateFn = Callable[..., CandidateMetrics]
CollectMetricsFn = Callable[..., List[CandidateMetrics]]


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
    score_candidate_fn: ScoreCandidateFn,
) -> List[CandidateMetrics]:
    metrics: List[CandidateMetrics] = []
    for item in work_items:
        metrics.append(
            score_candidate_fn(
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
        )
    return metrics


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
    collect_metrics_fn: CollectMetricsFn,
) -> List[CandidateMetrics]:
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
    collect_metrics_fn: CollectMetricsFn,
    subprocess_module: Optional[Any] = None,
) -> List[CandidateMetrics]:
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

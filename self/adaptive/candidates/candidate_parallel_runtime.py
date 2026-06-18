"""Local and Slurm candidate training dispatch wrappers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.candidates import candidate_workers
from self.adaptive.traces.experience_trace_models import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposals.proposal_prompts import PromptBundle


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

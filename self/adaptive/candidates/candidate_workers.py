#!/usr/bin/env python3
"""Candidate worker spec preparation and dispatch helpers."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.adaptive.candidates.candidate_worker_specs import (
    candidate_metric_path,
    candidate_worker_failure_path,
    prepare_candidate_worker_pack_specs,
    prepare_candidate_worker_specs,
)
from self.adaptive.candidates.candidate_local_workers import (
    train_candidates_local_parallel_from_specs,
    write_json,
    write_local_candidate_failure,
)
from self.adaptive.candidates.candidate_slurm_workers import (
    submit_candidate_array as _submit_candidate_array,
    train_candidates_slurm_array_from_specs,
    wait_for_candidate_array as _wait_for_candidate_array,
)
from self.adaptive.proposals.proposal_prompts import PromptBundle
from self.core.slurm import cancel_job, slurm_job_active, submit_sbatch


CollectMetricsFn = Callable[..., List[Any]]


def train_candidates_slurm_array(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    collect_metrics_fn: CollectMetricsFn,
) -> List[Any]:
    if not work_items:
        return []
    spec_paths = prepare_candidate_worker_specs(
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
    return train_candidates_slurm_array_from_specs(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        collect_metrics_fn=collect_metrics_fn,
        submit_candidate_array_fn=submit_candidate_array,
        wait_for_candidate_array_fn=wait_for_candidate_array,
    )


def train_candidates_local_parallel(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    collect_metrics_fn: CollectMetricsFn,
) -> List[Any]:
    if not work_items:
        return []
    spec_paths = prepare_candidate_worker_specs(
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
    return train_candidates_local_parallel_from_specs(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        collect_metrics_fn=collect_metrics_fn,
        prepare_pack_specs_fn=prepare_candidate_worker_pack_specs,
        subprocess_module=subprocess,
    )


def submit_candidate_array(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    spec_paths: Sequence[Path],
) -> str:
    return _submit_candidate_array(
        args=args,
        round_dir=round_dir,
        spec_paths=spec_paths,
        submit_sbatch_fn=submit_sbatch,
        executable=sys.executable,
        cwd_fn=Path.cwd,
    )


def wait_for_candidate_array(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    work_items: Sequence[Any],
    job_id: str,
) -> None:
    return _wait_for_candidate_array(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        job_id=job_id,
        cancel_job_fn=cancel_job,
        slurm_job_active_fn=slurm_job_active,
        monotonic_fn=time.monotonic,
        sleep_fn=time.sleep,
    )

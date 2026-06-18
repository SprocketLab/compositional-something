#!/usr/bin/env python3
"""Candidate execution and worker-result aggregation helpers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from self.core import candidate_workers, worker_io
from self.core.experience_traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import (
    CandidateMetrics,
    CandidateWorkItem,
    ExactPairDataset,
    candidate_metrics_from_json,
    proposal_from_payload,
)
from self.core.proposals import PromptBundle
from self.core.data_io import load_examples, sanitize_json_value
from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]
ScoreCandidateFn = Callable[..., CandidateMetrics]
CollectMetricsFn = Callable[..., List[CandidateMetrics]]


def work_item_to_worker_payload(
    *,
    item: CandidateWorkItem,
    round_dir: Path,
) -> JsonDict:
    candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
    return sanitize_json_value(
        {
            "index": item.index,
            "row_id": item.row_id,
            "proposal": item.proposal.to_json_dict(),
            "completion": item.completion,
            "raw_output": item.raw_output,
            "proposal_prediction": item.proposal_prediction,
            "pseudo_diagnostics": item.pseudo_diagnostics,
            "pseudo_examples_path": str(candidate_dir / "pseudo_examples.jsonl"),
            "pseudo_count": len(item.pseudo_examples),
            "composed_keys": [worker_io.json_ready_key(key) for key in sorted(item.composed.keys, key=repr)],
            "composed_count": len(item.composed.examples),
        }
    )


def work_item_from_worker_payload(
    *,
    payload: Mapping[str, Any],
    task: Any,
) -> CandidateWorkItem:
    pseudo_path = Path(str(payload["pseudo_examples_path"]))
    pseudo_examples = load_examples(pseudo_path, task.deserialize_example)
    composed_keys = {worker_io.key_from_json(key) for key in payload.get("composed_keys", [])}
    return CandidateWorkItem(
        index=int(payload["index"]),
        row_id=payload.get("row_id"),
        proposal=proposal_from_payload(dict(payload["proposal"])),
        completion=str(payload.get("completion", "")),
        raw_output=payload.get("raw_output"),
        composed=ExactPairDataset(examples=[], component_map={}, keys=composed_keys, diagnostics={}),
        pseudo_examples=pseudo_examples,
        pseudo_diagnostics=dict(payload.get("pseudo_diagnostics") or {}),
        proposal_prediction=dict(payload.get("proposal_prediction") or {}),
    )


def candidate_failure_metrics(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=False,
        reward=float("-inf"),
        frontier_delta=float("-inf"),
        target_accuracy=math.nan,
        current_target_accuracy=float(current_per_size_accuracy.get(item.proposal.target, 0.0)),
        final_accuracy=math.nan,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=math.nan,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=math.nan,
        per_size_accuracy={},
        pseudo_count=len(item.pseudo_examples),
        model_dir=None,
        failure_reason=reason,
        proposal_prediction=dict(item.proposal_prediction),
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


def collect_candidate_array_metrics(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    failure_metrics_fn: Optional[Callable[..., CandidateMetrics]] = None,
) -> List[CandidateMetrics]:
    metrics: List[CandidateMetrics] = []
    failures: List[JsonDict] = []
    failure_metrics_fn = failure_metrics_fn or candidate_failure_metrics
    for item in work_items:
        metrics_path = candidate_workers.candidate_metric_path(round_dir, item)
        if metrics_path.exists():
            metrics.append(candidate_metrics_from_json(worker_io.load_json(metrics_path)))
            continue
        failure_path = candidate_workers.candidate_worker_failure_path(round_dir, item)
        if failure_path.exists():
            failure_payload = worker_io.load_json(failure_path)
            reason = str(failure_payload.get("error") or "candidate worker failed")
        else:
            reason = "candidate worker finished without candidate_metrics.json"
        failure_metric = failure_metrics_fn(
            item=item,
            reason=reason,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
        )
        worker_io.write_json(metrics_path, failure_metric.to_json_dict())
        metrics.append(failure_metric)
        failures.append(
            {
                "candidate_index": item.index,
                "failure_reason": reason,
                "metrics_path": str(metrics_path),
                "worker_failure_path": str(failure_path),
            }
        )
    if failures:
        worker_io.write_json(round_dir / "candidate_jobs" / "gather_failures.json", failures)
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

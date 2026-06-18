#!/usr/bin/env python3
"""Candidate dispatch across serial, local-parallel, and Slurm modes."""

from __future__ import annotations

# --- from candidate_metric_collection.py ---
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from self.core import worker_io
from self.adaptive import candidate_workers as workers
from self.core.models import (
    CandidateMetrics,
    CandidateWorkItem,
    candidate_metrics_from_json,
)


JsonDict = Dict[str, Any]


def _candidate_failure_metrics_impl(
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
        current_target_accuracy=float(
            current_per_size_accuracy.get(item.proposal.target, 0.0)
        ),
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


def _collect_candidate_array_metrics_impl(
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
        metrics_path = workers.candidate_metric_path(round_dir, item)
        if metrics_path.exists():
            metrics.append(candidate_metrics_from_json(worker_io.load_json(metrics_path)))
            continue
        failure_path = workers.candidate_worker_failure_path(round_dir, item)
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


# --- from candidate_serial_runtime.py ---
import argparse
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal_prompts import PromptBundle
from self.core.training import TrainingConfig


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
    pass_model_bootstrap_cache = _score_candidate_accepts_model_bootstrap_cache(
        score_candidate_fn
    )
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
        metrics.append(score_candidate_fn(**kwargs))
    return metrics


def _score_candidate_accepts_model_bootstrap_cache(
    score_candidate_fn: Callable[..., CandidateMetrics],
) -> bool:
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


# --- from candidate_parallel_runtime.py ---
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive import candidate_workers as workers
from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal_prompts import PromptBundle


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
    return workers.train_candidates_slurm_array(
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
        workers.subprocess = subprocess_module
    return workers.train_candidates_local_parallel(
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


# --- from py ---
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal_prompts import PromptBundle
from self.core.training import TrainingConfig


def candidate_failure_metrics(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    return _candidate_failure_metrics_impl(
        item=item,
        reason=reason,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def collect_candidate_array_metrics(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    failure_metrics_fn: Callable[..., CandidateMetrics] | None = None,
) -> list[CandidateMetrics]:
    return _collect_candidate_array_metrics_impl(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=failure_metrics_fn,
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


# --- from candidate_dispatch_entrypoints.py ---
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal_prompts import PromptBundle
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


def build_candidate_dispatch_deps(bindings: Any) -> CandidateDispatchEntrypointDeps:
    return CandidateDispatchEntrypointDeps(
        train_and_score_candidate=bindings.train_and_score_candidate,
        candidate_failure_metrics=bindings._candidate_failure_metrics,
        collect_candidate_array_metrics=bindings._collect_candidate_array_metrics,
        train_candidates_serial=bindings.train_candidates_serial,
        train_candidates_local_parallel=bindings.train_candidates_local_parallel,
        train_candidates_slurm_array=bindings.train_candidates_slurm_array,
        subprocess_module=bindings.subprocess,
    )


def candidate_failure_metrics_with_deps(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    return candidate_failure_metrics(
        item=item,
        reason=reason,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def train_candidates_serial_with_deps(
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
    return train_candidates_serial(
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


def collect_candidate_array_metrics_with_deps(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return collect_candidate_array_metrics(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=deps.candidate_failure_metrics,
    )


def train_candidates_slurm_array_with_deps(
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
    return train_candidates_slurm_array(
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


def train_candidates_local_parallel_with_deps(
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
    return train_candidates_local_parallel(
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


def train_candidate_metrics_with_deps(
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
    return train_candidate_metrics(
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

"""Dependency models for candidate dispatch entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

from self.core.models import CandidateMetrics


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

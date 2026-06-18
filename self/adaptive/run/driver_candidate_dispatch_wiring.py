"""Driver-binding bridge for adaptive candidate dispatch."""

from __future__ import annotations

from typing import Any

from self.adaptive.candidates.candidate_dispatch_deps import (
    CandidateDispatchEntrypointDeps,
    build_candidate_dispatch_deps,
)
from self.adaptive.candidates.candidate_dispatch_entrypoints import (
    candidate_failure_metrics as _candidate_failure_metrics_entrypoint,
    collect_candidate_array_metrics as _collect_candidate_array_metrics_entrypoint,
    train_candidate_metrics as _train_candidate_metrics_entrypoint,
    train_candidates_local_parallel as _train_candidates_local_parallel_entrypoint,
    train_candidates_serial as _train_candidates_serial_entrypoint,
    train_candidates_slurm_array as _train_candidates_slurm_array_entrypoint,
)
from self.core.models import CandidateMetrics


def candidate_dispatch_deps(bindings: Any) -> CandidateDispatchEntrypointDeps:
    return build_candidate_dispatch_deps(bindings)


def candidate_failure_metrics(bindings: Any, **kwargs: Any) -> CandidateMetrics:
    return _candidate_failure_metrics_entrypoint(**kwargs)


def train_candidates_serial(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_serial_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def collect_candidate_array_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _collect_candidate_array_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidates_slurm_array(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_slurm_array_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidates_local_parallel(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_local_parallel_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidate_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidate_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))

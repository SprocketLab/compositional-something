#!/usr/bin/env python3
"""Compatibility reexports for candidate execution helpers."""

from __future__ import annotations

from self.adaptive.candidates.candidate_dispatch_runtime import (
    train_candidates_local_parallel,
    train_candidates_serial,
    train_candidates_slurm_array,
)
from self.adaptive.candidates.candidate_metric_collection import (
    candidate_failure_metrics,
    collect_candidate_array_metrics,
)
from self.adaptive.candidates.candidate_worker_payloads import (
    work_item_from_worker_payload,
    work_item_to_worker_payload,
)


__all__ = [
    "candidate_failure_metrics",
    "collect_candidate_array_metrics",
    "train_candidates_local_parallel",
    "train_candidates_serial",
    "train_candidates_slurm_array",
    "work_item_from_worker_payload",
    "work_item_to_worker_payload",
]

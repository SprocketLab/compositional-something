"""Candidate worker metric loading and failure-metric aggregation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from self.core import candidate_workers, worker_io
from self.core.models import (
    CandidateMetrics,
    CandidateWorkItem,
    candidate_metrics_from_json,
)


JsonDict = Dict[str, Any]


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

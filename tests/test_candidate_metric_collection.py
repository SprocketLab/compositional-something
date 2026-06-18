from __future__ import annotations

import math
from pathlib import Path

from self.core import candidate_execution, candidate_metric_collection, worker_io
from self.core.candidate_metric_collection import (
    candidate_failure_metrics,
    collect_candidate_array_metrics,
)
from self.core.models import CandidateMetrics, CandidateWorkItem, ExactPairDataset
from self.core.proposals import ConfigProposal


def _work_item(index: int = 0) -> CandidateWorkItem:
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    return CandidateWorkItem(
        index=index,
        row_id=f"row-{index}",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output=proposal.to_json_dict(),
        composed=ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=["pseudo-a", "pseudo-b"],
        pseudo_diagnostics={"retained_total": 2},
        proposal_prediction={"target": 5, "expected_frontier_delta": 0.1},
    )


def _metric(item: CandidateWorkItem) -> CandidateMetrics:
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=True,
        reward=0.3,
        frontier_delta=0.2,
        target_accuracy=0.7,
        current_target_accuracy=0.4,
        final_accuracy=0.6,
        current_final_accuracy=0.5,
        init_final_accuracy=0.25,
        final_accuracy_delta=0.35,
        final_accuracy_delta_from_current=0.1,
        per_size_accuracy={5: 0.7},
        pseudo_count=2,
        model_dir=Path("candidate-model"),
        proposal_prediction=item.proposal_prediction,
    )


def test_candidate_failure_metrics_preserves_candidate_context():
    item = _work_item(index=3)

    metric = candidate_failure_metrics(
        item=item,
        reason="worker crashed",
        current_final_accuracy=0.45,
        current_per_size_accuracy={5: 0.25},
        init_final_accuracy=0.2,
    )

    assert metric.index == 3
    assert metric.row_id == "row-3"
    assert metric.proposal == item.proposal
    assert not metric.valid
    assert metric.reward == float("-inf")
    assert metric.frontier_delta == float("-inf")
    assert math.isnan(metric.target_accuracy)
    assert metric.current_target_accuracy == 0.25
    assert metric.current_final_accuracy == 0.45
    assert metric.init_final_accuracy == 0.2
    assert metric.pseudo_count == 2
    assert metric.failure_reason == "worker crashed"
    assert metric.proposal_prediction == item.proposal_prediction


def test_collect_candidate_array_metrics_loads_existing_metric(tmp_path: Path):
    round_dir = tmp_path / "attempt_0001"
    item = _work_item(index=0)
    expected = _metric(item)
    worker_io.write_json(
        round_dir / "candidates" / "candidate_00" / "candidate_metrics.json",
        expected.to_json_dict(),
    )

    metrics = collect_candidate_array_metrics(
        round_dir=round_dir,
        work_items=[item],
        current_final_accuracy=0.5,
        current_per_size_accuracy={5: 0.4},
        init_final_accuracy=0.25,
    )

    assert len(metrics) == 1
    assert metrics[0].valid
    assert metrics[0].reward == 0.3
    assert metrics[0].per_size_accuracy == {5: 0.7}
    assert not (round_dir / "candidate_jobs" / "gather_failures.json").exists()


def test_collect_candidate_array_metrics_writes_failure_metric_and_manifest(tmp_path: Path):
    round_dir = tmp_path / "attempt_0001"
    item = _work_item(index=2)
    worker_io.write_json(
        round_dir / "candidates" / "candidate_02" / "worker_failure.json",
        {"error": "CUDA out of memory"},
    )

    metrics = collect_candidate_array_metrics(
        round_dir=round_dir,
        work_items=[item],
        current_final_accuracy=0.5,
        current_per_size_accuracy={5: 0.4},
        init_final_accuracy=0.25,
    )

    metric_path = round_dir / "candidates" / "candidate_02" / "candidate_metrics.json"
    gather_path = round_dir / "candidate_jobs" / "gather_failures.json"
    assert len(metrics) == 1
    assert metrics[0].failure_reason == "CUDA out of memory"
    assert metric_path.exists()
    assert worker_io.load_json(metric_path)["failure_reason"] == "CUDA out of memory"
    assert worker_io.load_json(gather_path) == [
        {
            "candidate_index": 2,
            "failure_reason": "CUDA out of memory",
            "metrics_path": str(metric_path),
            "worker_failure_path": str(
                round_dir / "candidates" / "candidate_02" / "worker_failure.json"
            ),
        }
    ]


def test_candidate_execution_reexports_metric_collection_helpers():
    assert candidate_execution.candidate_failure_metrics is candidate_metric_collection.candidate_failure_metrics
    assert (
        candidate_execution.collect_candidate_array_metrics
        is candidate_metric_collection.collect_candidate_array_metrics
    )

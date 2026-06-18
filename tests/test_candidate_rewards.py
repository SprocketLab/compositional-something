from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from self.adaptive.candidate_training import (
    build_no_pseudo_candidate_metrics,
    build_trained_candidate_metrics,
    mean_accuracy_for_sizes,
    static_frontier_sizes,
)
from self.core.models import CandidateWorkItem, ExactPairDataset
from self.adaptive.proposals import ConfigProposal


def _args(**overrides):
    values = dict(frontier_min_size=4, frontier_max_size=6, lambda_final=0.5)
    values.update(overrides)
    return SimpleNamespace(**values)


def _item(*, pseudo_examples=None, prediction=None):
    proposal = ConfigProposal(left=2, right=3, guard="none", target=5)
    return CandidateWorkItem(
        index=1,
        row_id="candidate-1",
        proposal=proposal,
        completion=proposal.to_completion(),
        raw_output={"left": 2, "right": 3, "guard": "none"},
        composed=ExactPairDataset(examples=[], component_map={}, keys=set(), diagnostics={}),
        pseudo_examples=list(pseudo_examples or []),
        pseudo_diagnostics={},
        proposal_prediction=dict(prediction or {"target": 5, "expected_frontier_delta": 0.2}),
    )


def test_static_frontier_accuracy_counts_missing_and_bad_values_as_zero():
    args = _args()

    assert static_frontier_sizes(args) == [4, 5, 6]
    assert mean_accuracy_for_sizes({4: 1.0, 5: "bad", 6: math.nan}, static_frontier_sizes(args)) == pytest.approx(
        1.0 / 3.0
    )
    assert math.isnan(mean_accuracy_for_sizes({}, []))


def test_no_pseudo_candidate_metrics_preserve_failure_contract():
    metrics = build_no_pseudo_candidate_metrics(
        args=_args(),
        item=_item(),
        current_final_accuracy=0.4,
        current_per_size_accuracy={4: 0.1, 5: 0.3, 6: 0.5},
        init_final_accuracy=0.2,
    )

    assert metrics.valid is False
    assert metrics.reward == float("-inf")
    assert metrics.frontier_delta == float("-inf")
    assert metrics.failure_reason == "no pseudo labels retained"
    assert metrics.pseudo_count == 0
    assert metrics.current_final_accuracy == 0.4
    assert metrics.current_target_accuracy == 0.3
    assert metrics.current_frontier_accuracy == pytest.approx(0.3)
    assert metrics.proposal_prediction["expected_frontier_delta"] == 0.2


def test_trained_candidate_metrics_use_static_frontier_and_final_delta_reward():
    item = _item(pseudo_examples=["p0", "p1"])
    metrics = build_trained_candidate_metrics(
        args=_args(lambda_final=0.5),
        item=item,
        final_accuracy=0.7,
        per_size_accuracy={4: 0.2, 5: 0.8, 6: 1.0, 7: 0.9},
        current_final_accuracy=0.4,
        current_per_size_accuracy={4: 0.1, 5: 0.3, 6: 0.4},
        init_final_accuracy=0.2,
        model_dir=Path("model"),
        proposal_trace_replay_count=3,
        candidate_proposal_trace_count=1,
        post_task_proposal_rehearsal_count=2,
        outcome_trace_replay_count=4,
    )

    assert metrics.valid is True
    assert metrics.target_accuracy == 0.8
    assert metrics.current_target_accuracy == 0.3
    assert metrics.target_delta == pytest.approx(0.5)
    assert metrics.frontier_accuracy == pytest.approx((0.2 + 0.8 + 1.0) / 3.0)
    assert metrics.current_frontier_accuracy == pytest.approx((0.1 + 0.3 + 0.4) / 3.0)
    assert metrics.frontier_delta == pytest.approx(0.4)
    assert metrics.final_accuracy_delta == pytest.approx(0.5)
    assert metrics.final_accuracy_delta_from_current == pytest.approx(0.3)
    assert metrics.reward == pytest.approx(0.4 + 0.5 * 0.5)
    assert metrics.pseudo_count == 2
    assert metrics.proposal_trace_replay_count == 3
    assert metrics.candidate_proposal_trace_count == 1
    assert metrics.post_task_proposal_rehearsal_count == 2
    assert metrics.outcome_trace_replay_count == 4
    assert metrics.per_size_accuracy == {4: 0.2, 5: 0.8, 6: 1.0, 7: 0.9}

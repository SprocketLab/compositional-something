from __future__ import annotations

from self.adaptive_frontier import proposal_quality_metrics, select_frontier


def test_select_frontier_prefers_low_accuracy_slice_size():
    diagnostics = {
        "per_size_accuracy": {"8": 0.91, "9": 0.74, "10": 0.80},
        "composed_eval_slices": {
            "boundary_carry": {
                "accuracy": 0.55,
                "count": 40,
                "per_size_accuracy": {"9": 0.60, "10": 0.40},
            },
            "no_boundary_carry": {
                "accuracy": 0.95,
                "count": 40,
                "per_size_accuracy": {"9": 0.94, "10": 0.96},
            },
        },
    }

    selection = select_frontier(
        diagnostics,
        task="addition",
        allowed_min=8,
        allowed_max=10,
        policy="weak_regime",
        max_accuracy=0.85,
    )

    assert selection.selected is not None
    assert selection.frontier_min() == 10
    assert selection.frontier_max() == 10
    assert selection.selected.slice_name == "boundary_carry"


def test_select_frontier_falls_back_when_no_weak_regime():
    selection = select_frontier(
        {"per_size_accuracy": {"8": 0.95, "9": 0.96}},
        task="addition",
        allowed_min=8,
        allowed_max=9,
        policy="weak_regime",
        max_accuracy=0.85,
    )

    assert selection.selected is None
    assert selection.frontier_min() == 8
    assert selection.frontier_max() == 9


def test_proposal_quality_metrics_tracks_validity_repairs_and_duplicates():
    metrics = proposal_quality_metrics(
        [
            {
                "id": "good",
                "valid": True,
                "reward": 0.2,
                "trace_include": True,
                "selection_eligible": True,
            },
            {
                "id": "repaired",
                "valid": True,
                "reward": 0.1,
                "trace_include": True,
                "selection_eligible": True,
                "repair_attempted": True,
                "repaired": True,
            },
            {
                "id": "duplicate",
                "valid": True,
                "reward": 0.3,
                "duplicate": True,
                "trace_include": False,
                "selection_eligible": False,
            },
            {
                "id": "bad",
                "valid": False,
                "validation_category": "forbidden_import",
            },
        ],
        selected_id="good",
    )

    assert metrics["valid_count"] == 3
    assert metrics["valid_rate"] == 0.75
    assert metrics["duplicate_count"] == 1
    assert metrics["repair_success_rate"] == 1.0
    assert metrics["invalid_by_category"] == {"forbidden_import": 1}

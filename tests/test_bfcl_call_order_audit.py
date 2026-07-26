from __future__ import annotations

import copy

from self.coding.bfcl_composition import build_hierarchical_cross_candidates
from self.experiments.bfcl_call_order_audit import (
    candidate_order_report,
    clause_units,
    prediction_order_report,
)
from tests.test_bfcl_composition import example


def _pool(count: int = 8):
    return [
        example(
            f"source-{index}",
            f"function-{index}",
            f"Please resolve request number {index}",
            f"argument-{index}",
            index,
            "integer",
        )
        for index in range(count)
    ]


def _questions(items):
    return {item.source_id: str(item.metadata["question"]) for item in items}


def _permuted(candidate, hidden):
    """Reproduce the defect: keep the request, reorder the component blocks."""

    broken_candidate = copy.deepcopy(dict(candidate))
    broken_candidate["component_specs"] = list(
        reversed(broken_candidate["component_specs"])
    )
    broken_oracle = copy.deepcopy(dict(hidden))
    width = len(broken_oracle["canonical_calls"]) // 2
    calls = broken_oracle["canonical_calls"]
    broken_oracle["canonical_calls"] = calls[width:] + calls[:width]
    return broken_candidate, broken_oracle


def test_report_accepts_aligned_candidates_and_flags_permuted_targets():
    items = _pool()
    questions = _questions(items)
    public, oracle = build_hierarchical_cross_candidates(
        items, component_count=4, count=4, seed=7
    )
    for candidate, hidden in zip(public, oracle):
        report = candidate_order_report(candidate, hidden, questions)
        assert report["target_in_clause_order"]
        assert report["displaced_call_fraction"] == 0.0
        assert report["call_count"] == 4

        broken_candidate, broken_oracle = _permuted(candidate, hidden)
        broken = candidate_order_report(broken_candidate, broken_oracle, questions)
        assert not broken["target_in_clause_order"]
        assert broken["displaced_call_fraction"] == 1.0


def test_prediction_order_report_tracks_the_emitted_call_order():
    items = _pool()
    questions = _questions(items)
    public, oracle = build_hierarchical_cross_candidates(
        items, component_count=2, count=3, seed=7
    )
    candidate, hidden = public[0], oracle[0]
    ordered = '[' + ', '.join(
        '{"name": "%s", "arguments": {}}' % call["name"]
        for call in hidden["canonical_calls"]
    ) + ']'
    reversed_calls = '[' + ', '.join(
        '{"name": "%s", "arguments": {}}' % call["name"]
        for call in reversed(hidden["canonical_calls"])
    ) + ']'
    assert prediction_order_report(candidate, hidden, ordered, questions) is True
    assert prediction_order_report(candidate, hidden, reversed_calls, questions) is False
    assert prediction_order_report(candidate, hidden, "not json", questions) is None


def test_clause_units_fall_back_to_persisted_leaf_questions():
    items = _pool(4)
    questions = _questions(items)
    public, _oracle = build_hierarchical_cross_candidates(
        items, component_count=4, count=1, seed=7
    )
    candidate = copy.deepcopy(public[0])
    for spec in candidate["component_specs"]:
        del spec["clause_questions"]
    assert clause_units(candidate, questions) == clause_units(public[0], questions)
    assert clause_units(candidate, {}) is None

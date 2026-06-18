from __future__ import annotations

from types import SimpleNamespace

import pytest

from self.nonadaptive.nonadaptive_schedule import build_nonadaptive_size_schedule, normalize_frontier_min_size


def test_legacy_size_schedule_without_frontier():
    args = SimpleNamespace(
        initial_max_size=7,
        expand_num_size=3,
        num_expand_rounds=4,
        frontier_min_size=None,
    )

    schedule = build_nonadaptive_size_schedule(args, normalize_frontier_min_size(args))

    assert schedule.final_max_size == 19
    assert schedule.composed_min_size == 8
    assert [schedule.round_max_size_for_index(index) for index in range(3)] == [7, 10, 13]
    assert [schedule.target_max_size_for_round(index) for index in range(3)] == [10, 13, 16]


def test_frontier_size_schedule_keeps_round_zero_at_initial_max():
    args = SimpleNamespace(
        initial_max_size=7,
        expand_num_size=3,
        num_expand_rounds=4,
        frontier_min_size=10,
    )

    schedule = build_nonadaptive_size_schedule(args, normalize_frontier_min_size(args))

    assert schedule.final_max_size == 21
    assert schedule.composed_min_size == 10
    assert [schedule.round_max_size_for_index(index) for index in range(3)] == [7, 12, 15]
    assert [schedule.target_max_size_for_round(index) for index in range(3)] == [12, 15, 18]


def test_frontier_zero_expand_rounds_keeps_final_max_at_initial_max():
    args = SimpleNamespace(
        initial_max_size=7,
        expand_num_size=3,
        num_expand_rounds=0,
        frontier_min_size=10,
    )

    schedule = build_nonadaptive_size_schedule(args, normalize_frontier_min_size(args))

    assert schedule.final_max_size == 7
    assert schedule.composed_min_size == 10
    assert schedule.round_max_size_for_index(0) == 7
    assert schedule.target_max_size_for_round(0) == 12


def test_frontier_min_size_must_exceed_initial_max():
    args = SimpleNamespace(initial_max_size=7, frontier_min_size=7)

    with pytest.raises(ValueError, match="frontier_min_size"):
        normalize_frontier_min_size(args)

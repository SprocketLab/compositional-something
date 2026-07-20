from __future__ import annotations

from self.experiments.coding_atomic_sweep import (
    INITIAL_SEED,
    LEARNING_RATES,
    REPLICATION_SEEDS,
    SweepCell,
    select_stage1_schedules,
    select_stage3_configs,
    stage1_cells,
    stage2_cells,
    stage3_cells,
)


def metrics(exact, format_accuracy=1.0):
    return {"validation": {"exact_accuracy": exact, "format_accuracy": format_accuracy}}


def test_staged_matrix_has_exactly_twenty_one_unique_cells_per_task():
    for task in ("bfcl", "commitpack"):
        first = stage1_cells(task)
        assert len(first) == 9
        assert {cell.learning_rate for cell in first} == set(LEARNING_RATES)
        results = [(cell, metrics(index / 100)) for index, cell in enumerate(first)]
        schedules = select_stage1_schedules(results)
        second = stage2_cells(task, schedules)
        seed7 = first + second
        selected = select_stage3_configs(
            [(cell, metrics(index / 100)) for index, cell in enumerate(seed7)]
        )
        third = stage3_cells(selected)
        all_cells = first + second + third
        assert len(all_cells) == 21
        assert len({(cell.slug, cell.task) for cell in all_cells}) == 21
        assert {cell.seed for cell in first + second} == {INITIAL_SEED}
        assert {cell.seed for cell in third} == set(REPLICATION_SEEDS)


def test_stage1_ties_prefer_fewer_steps_then_lower_learning_rate():
    cells = stage1_cells("bfcl")
    schedules = select_stage1_schedules([(cell, metrics(0.8)) for cell in cells])
    assert schedules == [(10, 1e-5), (10, 5e-5)]


def test_stage3_ranking_uses_validation_only_and_cost_tiebreaks():
    cells = [
        SweepCell("bfcl", 240, 10, 1e-5, 7, 1),
        SweepCell("bfcl", 240, 30, 1e-5, 7, 1),
        SweepCell("bfcl", 240, 100, 1e-5, 7, 1),
    ]
    cells.extend(
        SweepCell("bfcl", size, 10, 1e-5, 7, 2)
        for size in (30, 60, 120, 30, 60, 120, 30, 60, 120, 30, 60, 120)
    )
    selected = select_stage3_configs([(cell, metrics(0.9)) for cell in cells])
    assert [(cell.data_size, cell.max_steps) for cell in selected] == [(30, 10), (30, 10), (30, 10)]

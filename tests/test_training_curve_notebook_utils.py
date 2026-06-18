from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self.analysis import training_curve_results
from self.training_curve_notebook_utils import (
    per_size_accuracy_frame_from_results,
    plot_per_size_accuracy_heatmap_from_results,
)


def test_per_size_accuracy_frame_supports_addition_schema(tmp_path: Path):
    assert (
        per_size_accuracy_frame_from_results
        is training_curve_results.per_size_accuracy_frame_from_results
    )

    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.98, "7": 0.91},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.99, "12": 0.73},
                },
            ]
        ),
        encoding="utf-8",
    )

    frame = per_size_accuracy_frame_from_results(results_path, run_label="filtered")

    observed = [
        (int(row.size), int(row.max_size), float(row.accuracy))
        for row in frame.itertuples(index=False)
    ]
    assert observed == [
        (3, 7, 1.0),
        (3, 12, 1.0),
        (4, 7, 0.98),
        (4, 12, 0.99),
        (7, 7, 0.91),
        (12, 12, 0.73),
    ]


def test_plot_per_size_accuracy_heatmap_supports_addition_schema(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.98, "7": 0.91},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {"3": 1.0, "4": 0.99, "12": 0.73},
                },
            ]
        ),
        encoding="utf-8",
    )

    figure = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="addition",
        mode="with_carry_filtered",
        title="Addition Test",
    )

    assert figure.axes[0].get_ylabel() == "Digits in addends"


def test_plot_per_size_accuracy_heatmap_sparse_annotations_reduce_text_count(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_bits": 16,
                    "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 17)},
                },
                {
                    "round": 1,
                    "max_bits": 24,
                    "per_bit_accuracy": {str(bits): 0.80 for bits in range(8, 25)},
                },
                {
                    "round": 2,
                    "max_bits": 32,
                    "per_bit_accuracy": {str(bits): 0.60 for bits in range(8, 33)},
                },
            ]
        ),
        encoding="utf-8",
    )

    dense = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="run_length",
        mode="compose",
        annotate_mode="all",
    )
    sparse = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="run_length",
        mode="compose",
        annotate_mode="sparse",
        y_tick_stride=4,
        dense_y_ticks_through=16,
    )

    assert len(sparse.axes[0].texts) < len(dense.axes[0].texts)


def test_plot_per_size_accuracy_heatmap_supports_paper_layout_controls(tmp_path: Path):
    results_path = tmp_path / "self_improvement_results.json"
    results_path.write_text(
        json.dumps(
            [
                {
                    "round": 0,
                    "max_digits": 7,
                    "per_digit_accuracy": {str(digits): 0.99 for digits in range(3, 8)},
                },
                {
                    "round": 1,
                    "max_digits": 12,
                    "per_digit_accuracy": {str(digits): 0.75 for digits in range(3, 13)},
                },
            ]
        ),
        encoding="utf-8",
    )

    figure = plot_per_size_accuracy_heatmap_from_results(
        results_path,
        task="addition",
        mode="with_carry_filtered",
        title="Should Be Hidden",
        annotate_mode="sparse",
        show_title=False,
        y_tick_stride=2,
        dense_y_ticks_through=12,
        font_scale=1.4,
        fixed_canvas_size=(7.4, 9.0),
    )

    assert tuple(round(value, 1) for value in figure.get_size_inches()) == (7.4, 9.0)
    assert figure.axes[0].get_title() == ""
    assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == [str(value) for value in range(3, 13)]

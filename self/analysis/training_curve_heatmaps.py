"""Heatmap plotting helpers for self-improvement training-curve analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from self.analysis.training_curve_bundle import CurveBundle, per_size_accuracy_frame
from self.analysis.training_curve_results import per_size_accuracy_frame_from_results
from self.analysis.training_curve_style import configure_plot_style, mode_label


def _visible_tick_indices(
    sizes: List[int],
    *,
    y_tick_stride: Optional[int],
    dense_y_ticks_through: Optional[int],
) -> List[int]:
    if y_tick_stride is None or y_tick_stride <= 1:
        return list(range(len(sizes)))

    visible: List[int] = []
    for index, size in enumerate(sizes):
        if dense_y_ticks_through is not None and size <= dense_y_ticks_through:
            visible.append(index)
            continue
        if index == 0 or index == len(sizes) - 1 or size % y_tick_stride == 0:
            visible.append(index)
    return sorted(set(visible))


def _should_annotate_sparse_cell(
    *,
    row_index: int,
    column_index: int,
    row_count: int,
    column_count: int,
    y_tick_stride: Optional[int],
    dense_y_ticks_through: Optional[int],
    sizes: List[int],
) -> bool:
    if column_index == 0 or column_index == column_count - 1:
        return True
    if row_index == 0 or row_index == row_count - 1:
        return True
    if y_tick_stride is None or y_tick_stride <= 1:
        return False
    size = sizes[row_index]
    if dense_y_ticks_through is not None and size <= dense_y_ticks_through:
        return True
    return size % y_tick_stride == 0


def plot_per_size_accuracy_heatmap(
    bundle: CurveBundle,
    task: str,
    mode: str,
    budget: str,
    *,
    annotate: bool = True,
) -> plt.Figure:
    """Plot a round-by-size accuracy heatmap with max-size annotations in the column labels."""
    configure_plot_style()

    frame = per_size_accuracy_frame(bundle, task, mode, budget)
    if frame.empty:
        raise ValueError(f"No per-size accuracy found for task={task!r}, mode={mode!r}, budget={budget!r}.")

    size_axis_labels = {
        "run_length": "String length",
        "multiplication": "Digits per operand",
        "addition": "Digits in addends",
    }

    sizes = sorted(frame["size"].unique())
    rounds = sorted(frame["round"].unique())
    max_size_by_round = {
        int(round_index): int(frame.loc[frame["round"] == round_index, "max_size"].iloc[0]) for round_index in rounds
    }

    heatmap = (
        frame.pivot(index="size", columns="round", values="accuracy")
        .reindex(index=sizes, columns=rounds)
        .astype(float)
        .to_numpy()
    )

    fig_height = max(4.8, 0.42 * len(sizes) + 2.0)
    fig, ax = plt.subplots(figsize=(2.6 * len(rounds) + 2.4, fig_height), constrained_layout=True)
    image = ax.imshow(heatmap, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)

    x_tick_labels = [f"{round_index}\n(max {max_size_by_round[round_index]})" for round_index in rounds]
    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels([str(size) for size in sizes])
    ax.set_xlabel("Self-Improvement Round")
    ax.set_ylabel(size_axis_labels.get(task, "Size"))
    ax.set_title(f"{task.title()} per-size accuracy heatmap | {mode_label(mode)} | {budget.title()} budget")

    if annotate:
        for row_index, _size in enumerate(sizes):
            for column_index, _round_index in enumerate(rounds):
                value = heatmap[row_index, column_index]
                if np.isnan(value):
                    text = "n/a"
                    color = "white"
                else:
                    text = f"{value:.2f}"
                    color = "white" if value < 0.55 else "#202020"
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=color,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy")
    return fig


def plot_per_size_accuracy_heatmap_from_results(
    results_path: str | Path,
    *,
    task: str,
    mode: str,
    title: Optional[str] = None,
    annotate: bool = True,
    annotate_mode: str = "all",
    show_title: bool = True,
    y_tick_stride: Optional[int] = None,
    dense_y_ticks_through: Optional[int] = None,
    font_scale: float = 1.0,
    fixed_canvas_size: Optional[tuple[float, float]] = None,
    show_round_max_labels: bool = True,
) -> plt.Figure:
    """Plot a round-by-size heatmap directly from a self-improvement results file."""
    configure_plot_style()

    frame = per_size_accuracy_frame_from_results(results_path, run_label=mode)
    if frame.empty:
        raise ValueError(f"No per-size accuracy found in {results_path}.")

    size_axis_labels = {
        "run_length": "String length",
        "multiplication": "Digits per operand",
        "addition": "Digits in addends",
    }

    sizes = sorted(frame["size"].unique())
    rounds = sorted(frame["round"].unique())
    max_size_by_round = {
        int(round_index): int(frame.loc[frame["round"] == round_index, "max_size"].iloc[0]) for round_index in rounds
    }
    if annotate_mode not in {"all", "sparse", "none"}:
        raise ValueError(f"Unsupported annotate_mode={annotate_mode!r}.")
    if not annotate:
        annotate_mode = "none"
    if font_scale <= 0:
        raise ValueError("font_scale must be positive.")

    heatmap = (
        frame.pivot(index="size", columns="round", values="accuracy")
        .reindex(index=sizes, columns=rounds)
        .astype(float)
        .to_numpy()
    )

    if fixed_canvas_size is None:
        fig_height = max(4.6, 0.22 * len(sizes) + 1.7)
        figure_size = (3.0 * len(rounds) + 3.2, fig_height)
    else:
        figure_size = fixed_canvas_size
    fig, ax = plt.subplots(figsize=figure_size, constrained_layout=True)
    image = ax.imshow(heatmap, aspect="auto", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(rounds)))
    ax.set_xticklabels([str(round_index) for round_index in rounds], fontsize=12 * font_scale)
    visible_y_tick_indices = _visible_tick_indices(
        sizes,
        y_tick_stride=y_tick_stride,
        dense_y_ticks_through=dense_y_ticks_through,
    )
    ax.set_yticks(np.array(visible_y_tick_indices, dtype=float))
    ax.set_yticklabels([str(sizes[index]) for index in visible_y_tick_indices], fontsize=12 * font_scale)
    ax.set_xlabel("Self-Improvement Round", fontsize=14 * font_scale)
    ax.set_ylabel(size_axis_labels.get(task, "Size"), fontsize=14 * font_scale)
    if show_title:
        ax.set_title(title or f"{task.title()} per-size accuracy heatmap — {mode_label(mode)}", fontsize=16 * font_scale)
    else:
        ax.set_title("")
    ax.tick_params(axis="x", labelsize=12 * font_scale)
    ax.tick_params(axis="y", labelsize=12 * font_scale)

    if show_round_max_labels:
        for column_index, round_index in enumerate(rounds):
            max_size = max_size_by_round[round_index]
            ax.text(
                column_index,
                -1.15,
                f"max {max_size}",
                ha="center",
                va="bottom",
                fontsize=11 * font_scale,
                color="#303030",
                clip_on=False,
            )

    if annotate_mode != "none":
        for row_index, _size in enumerate(sizes):
            for column_index, _round_index in enumerate(rounds):
                if annotate_mode == "sparse" and not _should_annotate_sparse_cell(
                    row_index=row_index,
                    column_index=column_index,
                    row_count=len(sizes),
                    column_count=len(rounds),
                    y_tick_stride=y_tick_stride,
                    dense_y_ticks_through=dense_y_ticks_through,
                    sizes=sizes,
                ):
                    continue
                value = heatmap[row_index, column_index]
                if np.isnan(value):
                    text = "n/a"
                    color = "white"
                else:
                    text = f"{value:.2f}"
                    color = "white" if value < 0.55 else "#202020"
                ax.text(
                    column_index,
                    row_index,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9.2 * font_scale,
                    color=color,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Accuracy", fontsize=13 * font_scale)
    colorbar.ax.tick_params(labelsize=11 * font_scale)
    return fig


__all__ = [
    "_should_annotate_sparse_cell",
    "_visible_tick_indices",
    "plot_per_size_accuracy_heatmap",
    "plot_per_size_accuracy_heatmap_from_results",
]

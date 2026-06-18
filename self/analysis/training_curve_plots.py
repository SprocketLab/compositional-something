"""Plotting helpers for self-improvement training-curve analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from self.analysis.training_curve_bundle import CurveBundle
from self.analysis.training_curve_heatmaps import (
    _should_annotate_sparse_cell,
    _visible_tick_indices,
    plot_per_size_accuracy_heatmap,
    plot_per_size_accuracy_heatmap_from_results,
)
from self.analysis.training_curve_results import (
    round_summary_frame,
)
from self.analysis.training_curve_style import (
    BASELINE_COLORS,
    BUDGET_ORDER,
    MODE_ORDER,
    configure_plot_style,
    mode_label,
)


def _subset_modes(frame: pd.DataFrame) -> List[str]:
    present = [mode for mode in MODE_ORDER if mode in set(frame["mode"].astype(str))]
    return present


def plot_task_curves(
    bundle: CurveBundle,
    task: str,
    *,
    include_composed_eval: bool = False,
) -> plt.Figure:
    """Plot training loss, validation loss, and test accuracy for one task."""
    configure_plot_style()

    round_subset = bundle.round_metrics[bundle.round_metrics["task"] == task].copy()
    train_subset = bundle.train_logs[bundle.train_logs["task"] == task].copy()
    if bundle.validation_logs.empty:
        validation_subset = pd.DataFrame(
            columns=[
                "task",
                "mode",
                "budget",
                "round",
                "step_in_round",
                "round_progress",
                "validation_loss",
                "line_number",
            ]
        )
    else:
        validation_subset = bundle.validation_logs[bundle.validation_logs["task"] == task].copy()
    if round_subset.empty:
        raise ValueError(f"No round metrics found for task={task!r}.")

    budgets = [budget for budget in BUDGET_ORDER if budget in set(round_subset["budget"].astype(str))]
    modes = _subset_modes(round_subset)
    max_round = int(round_subset["round"].max())

    fig, axes = plt.subplots(
        3,
        len(budgets),
        figsize=(5.4 * len(budgets), 10.8),
        constrained_layout=True,
        squeeze=False,
    )

    for column, budget in enumerate(budgets):
        budget_rounds = round_subset[round_subset["budget"].astype(str) == budget]
        budget_train = train_subset[train_subset["budget"].astype(str) == budget]
        budget_validation = validation_subset[validation_subset["budget"].astype(str) == budget]

        train_ax = axes[0, column]
        val_ax = axes[1, column]
        test_ax = axes[2, column]

        train_ax.set_title(f"{task.title()} | {budget.title()} Budget")
        for boundary in range(1, max_round + 1):
            train_ax.axvline(boundary, color="#666666", linestyle="--", linewidth=0.9, alpha=0.25)
            val_ax.axvline(boundary, color="#666666", linestyle="--", linewidth=0.9, alpha=0.25)

        for mode in modes:
            color = BASELINE_COLORS.get(mode, "#444444")
            mode_train = budget_train[budget_train["mode"].astype(str) == mode].sort_values(
                ["round", "step_in_round", "line_number"]
            )
            if not mode_train.empty:
                train_ax.plot(
                    mode_train["round_progress"],
                    mode_train["loss"],
                    marker="o",
                    color=color,
                    label=mode_label(mode),
                )

            mode_validation = budget_validation[budget_validation["mode"].astype(str) == mode].sort_values(
                ["round", "step_in_round", "line_number"]
            )
            if not mode_validation.empty:
                val_ax.plot(
                    mode_validation["round_progress"],
                    mode_validation["validation_loss"],
                    marker="o",
                    color=color,
                    label=mode_label(mode),
                )

            mode_rounds = budget_rounds[budget_rounds["mode"].astype(str) == mode].sort_values("round")
            if not mode_rounds.empty:
                test_ax.plot(
                    mode_rounds["round"],
                    mode_rounds["eval_accuracy"],
                    marker="o",
                    color=color,
                    label=mode_label(mode),
                )
                if include_composed_eval and mode_rounds["composed_eval_accuracy"].notna().any():
                    test_ax.plot(
                        mode_rounds["round"],
                        mode_rounds["composed_eval_accuracy"],
                        marker="s",
                        linestyle="--",
                        color=color,
                        alpha=0.75,
                    )

        train_ax.set_ylabel("Training Loss")
        train_ax.set_xlabel("Round + Fractional Epoch")
        train_ax.set_xlim(0.0, max_round + 1.0)

        val_ax.set_ylabel("Validation Loss")
        val_ax.set_xlabel("Round + Fractional Epoch")
        val_ax.set_xlim(0.0, max_round + 1.0)
        if budget_validation.empty:
            val_ax.text(
                0.5,
                0.5,
                "No validation loss logged\n(eval_dataset=None)",
                ha="center",
                va="center",
                transform=val_ax.transAxes,
                fontsize=12,
                color="#555555",
            )

        test_ax.set_ylabel("Test Accuracy")
        test_ax.set_xlabel("Round")
        test_ax.set_xticks(range(0, max_round + 1))
        test_ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.02))

    if include_composed_eval:
        fig.suptitle(
            f"{task.title()} Training Curves (solid test = ordinary eval, dashed test = composed eval)",
            y=1.06,
        )
    else:
        fig.suptitle(f"{task.title()} Training Curves", y=1.04)
    return fig


def plot_self_improvement_comparison_curve(
    runs: Dict[str, str | Path],
    *,
    x_key: str,
    metric_key: str = "eval_accuracy",
    title: str,
    x_label: str,
    y_label: str = "Accuracy",
    ylim: tuple[float, float] = (0.0, 1.0),
) -> plt.Figure:
    """Plot a multi-baseline self-improvement comparison from arbitrary results files."""
    configure_plot_style()

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for label, raw_path in runs.items():
        frame = round_summary_frame(raw_path, run_label=label)
        if frame.empty:
            continue
        if x_key not in frame.columns:
            raise KeyError(f"{x_key!r} not found in {raw_path}.")
        if metric_key not in frame.columns:
            raise KeyError(f"{metric_key!r} not found in {raw_path}.")
        x_values = frame[x_key].astype(int).to_numpy()
        y_values = frame[metric_key].astype(float).to_numpy()
        ax.plot(
            x_values,
            y_values,
            marker="o",
            color=BASELINE_COLORS.get(label, "#444444"),
            label=mode_label(label),
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    return fig


def save_figure_bundle(fig: plt.Figure, output_stem: str | Path) -> List[Path]:
    """Save a figure as both PDF and PNG and return the written paths."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths

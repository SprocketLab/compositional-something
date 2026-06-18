"""Plotting helpers for seed-fit training-curve analysis."""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from self.analysis.seed_fit_bundle import SeedFitCurveBundle


TASK_COLORS: Dict[str, str] = {
    "addition": "#1d3557",
    "run_length": "#bc6c25",
}


def configure_plot_style() -> None:
    """Set consistent, paper-readable plotting defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.8,
            "lines.linewidth": 2.1,
            "lines.markersize": 4.5,
        }
    )


def _task_color(task: str) -> str:
    return TASK_COLORS.get(task, "#4c566a")


def _budget_palette(count: int, base_color: str) -> List[Any]:
    cmap = plt.get_cmap("viridis")
    if count <= 1:
        return [base_color]
    return [cmap(x) for x in np.linspace(0.2, 0.9, count)]


def plot_task_loss_curves(bundle: SeedFitCurveBundle, task: str) -> plt.Figure:
    """Plot train and validation loss over optimization steps for a single task."""
    task_runs = bundle.runs[bundle.runs["task"] == task].sort_values("total_train_examples")
    if task_runs.empty:
        raise ValueError(f"No seed-fit runs found for task={task!r}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    colors = _budget_palette(len(task_runs), _task_color(task))

    for color, (_, run) in zip(colors, task_runs.iterrows()):
        label = run["budget_label"]
        train_df = bundle.train_logs[bundle.train_logs["results_path"] == run["results_path"]]
        val_df = bundle.validation_logs[bundle.validation_logs["results_path"] == run["results_path"]]
        if not train_df.empty:
            axes[0].plot(train_df["step"], train_df["loss"], label=label, color=color)
        if not val_df.empty:
            axes[1].plot(val_df["step"], val_df["eval_loss"], label=label, color=color)

    axes[0].set_title(f"{task} train loss")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Train loss")
    axes[1].set_title(f"{task} validation loss")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Validation loss")

    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        axes[1].legend(handles, labels, title="Budget", frameon=False, loc="upper right")

    return fig


def plot_task_budget_curve(bundle: SeedFitCurveBundle, task: str, threshold: float = 0.95) -> plt.Figure:
    """Plot final held-out accuracy as a function of total seed-train budget."""
    summary = bundle.runs[bundle.runs["task"] == task].sort_values("total_train_examples")
    if summary.empty:
        raise ValueError(f"No seed-fit runs found for task={task!r}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    base_color = _task_color(task)

    ax.plot(
        summary["total_train_examples"],
        summary["validation_min_per_size_accuracy"],
        marker="o",
        color=base_color,
        label="Validation min-per-size",
    )
    ax.plot(
        summary["total_train_examples"],
        summary["test_min_per_size_accuracy"],
        marker="s",
        color="#c1121f",
        label="Test min-per-size",
    )
    ax.axhline(threshold, linestyle="--", color="#6c757d", linewidth=1.5, label=f"Target = {threshold:.2f}")

    ax.set_title(f"{task} held-out seed accuracy vs budget")
    ax.set_xlabel("Seed train examples")
    ax.set_ylabel("Exact-match accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, loc="lower right")

    for _, row in summary.iterrows():
        ax.annotate(
            row["budget_label"],
            (row["total_train_examples"], row["test_min_per_size_accuracy"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    return fig

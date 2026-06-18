"""Utilities for plotting self-improvement training curves from Slurm runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from self.analysis.training_curve_logs import (
    ROUND_PATTERN,
    _to_float,
    load_round_metrics,
    parse_training_log,
)
from self.analysis.training_curve_results import (
    load_round_payload,
    per_size_accuracy_frame_from_results,
    resolve_results_path,
    round_summary_frame,
)
from self.analysis.training_curve_style import (
    BASELINE_COLORS,
    BUDGET_ORDER,
    MODE_ORDER,
    configure_plot_style,
    mode_label,
)


@dataclass
class CurveBundle:
    jobs: pd.DataFrame
    train_logs: pd.DataFrame
    validation_logs: pd.DataFrame
    round_metrics: pd.DataFrame


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


def load_submission_jobs(run_root: Path, logs_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load the Slurm submission table and attach derived paths."""
    run_root = Path(run_root)
    if logs_dir is None:
        logs_dir = run_root.parents[2] / "logs"

    jobs = pd.read_csv(run_root / "submission_jobs.tsv", sep="\t")
    jobs["job_id"] = jobs["job_id"].astype(str)
    jobs["run_root"] = str(run_root)
    jobs["results_path"] = jobs["out_dir"].map(lambda item: str(Path(item) / "self_improvement_results.json"))
    jobs["log_path"] = jobs["job_id"].map(lambda job_id: str(logs_dir / f"selfimp-grid-{job_id}.out"))
    jobs["budget"] = pd.Categorical(jobs["budget"], categories=BUDGET_ORDER, ordered=True)
    jobs["mode"] = pd.Categorical(jobs["mode"], categories=MODE_ORDER, ordered=True)
    jobs["task"] = jobs["task"].astype(str)
    return jobs


def load_curve_bundle(run_root: str | Path, logs_dir: Optional[str | Path] = None) -> CurveBundle:
    """Load all curves for a budget-grid run root."""
    run_root_path = Path(run_root)
    log_root = Path(logs_dir) if logs_dir is not None else None
    jobs = load_submission_jobs(run_root_path, logs_dir=log_root)

    train_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    round_rows: List[Dict[str, Any]] = []

    for job in jobs.to_dict(orient="records"):
        train_df, validation_df = parse_training_log(Path(job["log_path"]))
        round_df = load_round_metrics(Path(job["results_path"]))

        for frame in (train_df, validation_df, round_df):
            if frame.empty:
                continue
            frame["job_id"] = job["job_id"]
            frame["task"] = job["task"]
            frame["mode"] = job["mode"]
            frame["budget"] = job["budget"]
            frame["out_dir"] = job["out_dir"]

        if not train_df.empty:
            train_rows.extend(train_df.to_dict(orient="records"))
        if not validation_df.empty:
            validation_rows.extend(validation_df.to_dict(orient="records"))
        if not round_df.empty:
            round_rows.extend(round_df.to_dict(orient="records"))

    train_logs = pd.DataFrame(train_rows)
    validation_logs = pd.DataFrame(validation_rows)
    round_metrics = pd.DataFrame(round_rows)

    if not train_logs.empty:
        train_logs["budget"] = pd.Categorical(train_logs["budget"], categories=BUDGET_ORDER, ordered=True)
        train_logs["mode"] = pd.Categorical(train_logs["mode"], categories=MODE_ORDER, ordered=True)
    if not validation_logs.empty:
        validation_logs["budget"] = pd.Categorical(validation_logs["budget"], categories=BUDGET_ORDER, ordered=True)
        validation_logs["mode"] = pd.Categorical(validation_logs["mode"], categories=MODE_ORDER, ordered=True)
    if not round_metrics.empty:
        round_metrics["budget"] = pd.Categorical(round_metrics["budget"], categories=BUDGET_ORDER, ordered=True)
        round_metrics["mode"] = pd.Categorical(round_metrics["mode"], categories=MODE_ORDER, ordered=True)

    return CurveBundle(
        jobs=jobs,
        train_logs=train_logs,
        validation_logs=validation_logs,
        round_metrics=round_metrics,
    )


def get_job_record(bundle: CurveBundle, task: str, mode: str, budget: str) -> Dict[str, Any]:
    """Return the submission-table row for a specific task/mode/budget run."""
    matches = bundle.jobs[
        (bundle.jobs["task"] == task)
        & (bundle.jobs["mode"].astype(str) == mode)
        & (bundle.jobs["budget"].astype(str) == budget)
    ]
    if matches.empty:
        raise ValueError(f"No run found for task={task!r}, mode={mode!r}, budget={budget!r}.")
    return matches.iloc[0].to_dict()


def per_size_accuracy_frame(bundle: CurveBundle, task: str, mode: str, budget: str) -> pd.DataFrame:
    """Expand per-size accuracy from round summaries into a tidy frame."""
    job = get_job_record(bundle, task, mode, budget)
    payload = load_round_payload(Path(job["results_path"]))
    rows: List[Dict[str, Any]] = []
    for entry in payload:
        round_index = int(entry["round"])
        max_size = int(entry["max_size"])
        per_size_accuracy = entry.get("per_size_accuracy", {})
        if not isinstance(per_size_accuracy, dict):
            continue
        for size, accuracy in per_size_accuracy.items():
            rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "budget": budget,
                    "round": round_index,
                    "max_size": max_size,
                    "size": int(size),
                    "accuracy": _to_float(accuracy),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["size", "round"]).reset_index(drop=True)
    return frame


def build_run_summary(bundle: CurveBundle) -> pd.DataFrame:
    """Create a compact summary table for notebook display."""
    rows: List[Dict[str, Any]] = []
    for job in bundle.jobs.to_dict(orient="records"):
        if bundle.round_metrics.empty:
            round_subset = pd.DataFrame()
        else:
            round_subset = bundle.round_metrics[bundle.round_metrics["job_id"] == job["job_id"]]

        if bundle.train_logs.empty:
            train_subset = pd.DataFrame()
        else:
            train_subset = bundle.train_logs[bundle.train_logs["job_id"] == job["job_id"]]

        if bundle.validation_logs.empty:
            validation_subset = pd.DataFrame()
        else:
            validation_subset = bundle.validation_logs[bundle.validation_logs["job_id"] == job["job_id"]]

        final_eval = None
        final_composed = None
        final_train_examples = None
        if not round_subset.empty:
            final_row = round_subset.sort_values("round").iloc[-1]
            final_eval = final_row["eval_accuracy"]
            final_composed = final_row["composed_eval_accuracy"]
            final_train_examples = int(final_row["train_examples"])

        rows.append(
            {
                "task": job["task"],
                "mode": job["mode"],
                "budget": job["budget"],
                "job_id": job["job_id"],
                "train_log_points": len(train_subset),
                "validation_log_points": len(validation_subset),
                "rounds": int(round_subset["round"].nunique()) if not round_subset.empty else 0,
                "final_train_examples": final_train_examples,
                "final_eval_accuracy": final_eval,
                "final_composed_eval_accuracy": final_composed,
            }
        )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["budget"] = pd.Categorical(summary["budget"], categories=BUDGET_ORDER, ordered=True)
        summary["mode"] = pd.Categorical(summary["mode"], categories=MODE_ORDER, ordered=True)
        summary = summary.sort_values(["task", "budget", "mode"]).reset_index(drop=True)
    return summary


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
        for row_index, size in enumerate(sizes):
            for column_index, round_index in enumerate(rounds):
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

#!/usr/bin/env python3
"""Seed-fit artifact loading, summaries, and plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, natural_sort_key, read_json


SEED_FIT_RESULTS_FILE = "seed_fit_results.json"


def is_seed_fit_run_dir(path: Path | str) -> bool:
    resolved = Path(path)
    return resolved.is_dir() and (resolved / SEED_FIT_RESULTS_FILE).exists()


def resolve_seed_fit_results_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / SEED_FIT_RESULTS_FILE
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find seed-fit results at {resolved}.")
    if resolved.name != SEED_FIT_RESULTS_FILE:
        raise ValueError(f"Expected {SEED_FIT_RESULTS_FILE}, got {resolved.name}.")
    return resolved


def discover_seed_fit_results(root: Path | str) -> list[Path]:
    resolved = Path(root)
    if resolved.is_file():
        return [resolve_seed_fit_results_path(resolved)]
    if is_seed_fit_run_dir(resolved):
        return [resolved / SEED_FIT_RESULTS_FILE]
    if not resolved.exists():
        return []
    return sorted(resolved.rglob(SEED_FIT_RESULTS_FILE), key=natural_sort_key)


def load_seed_fit_result(path: Path | str) -> JsonDict:
    results_path = resolve_seed_fit_results_path(path)
    payload = read_json(results_path, {})
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object payload in {results_path}.")
    return dict(payload)


def load_seed_fit_results(root: Path | str) -> list[tuple[Path, JsonDict]]:
    return [(path, load_seed_fit_result(path)) for path in discover_seed_fit_results(root)]


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd



@dataclass
class SeedFitCurveBundle:
    runs: pd.DataFrame
    train_logs: pd.DataFrame
    validation_logs: pd.DataFrame


def _load_json(path: Path) -> Dict[str, Any]:
    """Compatibility alias for older notebook helpers."""

    return load_seed_fit_result(path)


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _split_accuracy(results: Dict[str, Any], split: str) -> Optional[float]:
    split_payload = results.get(split)
    if not isinstance(split_payload, dict):
        return None
    return _as_float(split_payload.get("accuracy"))


def _split_min_per_size_accuracy(results: Dict[str, Any], split: str) -> Optional[float]:
    split_payload = results.get(split)
    if not isinstance(split_payload, dict):
        return None
    return _as_float(split_payload.get("min_per_size_accuracy"))


def _final_metric(log_history: List[Dict[str, Any]], key: str) -> Optional[float]:
    for record in reversed(log_history):
        if key in record:
            return _as_float(record.get(key))
    return None


def _budget_label(row: pd.Series) -> str:
    requested = row.get("initial_train_per_size_requested")
    total = row.get("total_train_examples")
    if pd.notna(requested):
        requested_int = int(requested)
        return f"{requested_int:,}/size"
    return f"{int(total):,} total"


def load_seed_fit_bundle(run_roots: Iterable[Path]) -> SeedFitCurveBundle:
    """Load seed-fit result files and flatten their training logs."""
    runs: List[Dict[str, Any]] = []
    train_logs: List[Dict[str, Any]] = []
    validation_logs: List[Dict[str, Any]] = []

    for run_root in run_roots:
        root = Path(run_root)
        if not root.exists():
            continue
        for results_path in discover_seed_fit_results(root):
            payload = _load_json(results_path)
            training = payload.get("training", {})
            results = payload.get("results", {})
            log_history = training.get("log_history", []) or []

            run_record = {
                "task": payload.get("task"),
                "run_root": str(root),
                "root_name": root.name,
                "run_name": results_path.parent.name,
                "results_path": str(results_path),
                "output_dir": payload.get("output_dir"),
                "model_name": payload.get("model_name"),
                "seed": payload.get("seed"),
                "initial_min_size": payload.get("initial_min_size"),
                "initial_max_size": payload.get("initial_max_size"),
                "num_sizes": (
                    int(payload["initial_max_size"]) - int(payload["initial_min_size"]) + 1
                    if payload.get("initial_min_size") is not None and payload.get("initial_max_size") is not None
                    else None
                ),
                "initial_train_per_size_requested": payload.get("initial_train_per_size"),
                "initial_eval_per_size": payload.get("initial_eval_per_size"),
                "total_train_examples": payload.get("train_examples"),
                "total_validation_examples": payload.get("validation_examples"),
                "total_test_examples": payload.get("test_examples"),
                "learning_rate": training.get("learning_rate"),
                "effective_batch_size": training.get("effective_batch_size"),
                "num_epochs": training.get("num_epochs"),
                "max_steps": training.get("max_steps"),
                "final_epoch": training.get("final_epoch"),
                "approx_effective_epochs_from_steps": training.get("approx_effective_epochs_from_steps"),
                "train_accuracy": _split_accuracy(results, "train"),
                "validation_accuracy": _split_accuracy(results, "validation"),
                "test_accuracy": _split_accuracy(results, "test"),
                "validation_min_per_size_accuracy": _split_min_per_size_accuracy(results, "validation"),
                "test_min_per_size_accuracy": _split_min_per_size_accuracy(results, "test"),
                "final_train_loss": _final_metric(log_history, "loss"),
                "final_validation_loss": _final_metric(log_history, "eval_loss"),
                "meets_threshold": payload.get("meets_threshold"),
            }
            runs.append(run_record)

            for record in log_history:
                step = record.get("step")
                epoch = record.get("epoch")
                if "loss" in record:
                    train_logs.append(
                        {
                            "task": run_record["task"],
                            "run_name": run_record["run_name"],
                            "results_path": run_record["results_path"],
                            "total_train_examples": run_record["total_train_examples"],
                            "initial_train_per_size_requested": run_record["initial_train_per_size_requested"],
                            "step": step,
                            "epoch": epoch,
                            "loss": _as_float(record.get("loss")),
                        }
                    )
                if "eval_loss" in record:
                    validation_logs.append(
                        {
                            "task": run_record["task"],
                            "run_name": run_record["run_name"],
                            "results_path": run_record["results_path"],
                            "total_train_examples": run_record["total_train_examples"],
                            "initial_train_per_size_requested": run_record["initial_train_per_size_requested"],
                            "step": step,
                            "epoch": epoch,
                            "eval_loss": _as_float(record.get("eval_loss")),
                        }
                    )

    runs_df = pd.DataFrame(runs)
    if runs_df.empty:
        empty = pd.DataFrame()
        return SeedFitCurveBundle(runs=empty, train_logs=empty, validation_logs=empty)

    runs_df["budget_label"] = runs_df.apply(_budget_label, axis=1)
    runs_df = runs_df.sort_values(["task", "total_train_examples", "run_name"]).reset_index(drop=True)

    train_logs_df = pd.DataFrame(train_logs)
    if not train_logs_df.empty:
        train_logs_df = train_logs_df.sort_values(["task", "total_train_examples", "step"]).reset_index(drop=True)

    validation_logs_df = pd.DataFrame(validation_logs)
    if not validation_logs_df.empty:
        validation_logs_df = validation_logs_df.sort_values(["task", "total_train_examples", "step"]).reset_index(
            drop=True
        )

    return SeedFitCurveBundle(
        runs=runs_df,
        train_logs=train_logs_df,
        validation_logs=validation_logs_df,
    )


def summarize_task(bundle: SeedFitCurveBundle, task: str) -> pd.DataFrame:
    """Return a compact summary table for one task."""
    summary = bundle.runs[bundle.runs["task"] == task].copy()
    if summary.empty:
        return summary
    columns = [
        "run_name",
        "budget_label",
        "total_train_examples",
        "validation_accuracy",
        "validation_min_per_size_accuracy",
        "test_accuracy",
        "test_min_per_size_accuracy",
        "final_validation_loss",
        "approx_effective_epochs_from_steps",
        "meets_threshold",
    ]
    return summary[columns].sort_values("total_train_examples").reset_index(drop=True)


def find_threshold_budget(bundle: SeedFitCurveBundle, task: str, threshold: float = 0.95) -> Optional[pd.Series]:
    """Return the smallest budget whose held-out minimum per-size test accuracy clears the threshold."""
    task_runs = bundle.runs[bundle.runs["task"] == task].sort_values("total_train_examples")
    if task_runs.empty:
        return None
    eligible = task_runs[task_runs["test_min_per_size_accuracy"] >= threshold]
    if eligible.empty:
        return None
    return eligible.iloc[0]


from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np



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

"""Bundle loaders and summary tables for seed-fit analysis notebooks."""

from __future__ import annotations

import json
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
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
        for results_path in sorted(root.glob("**/seed_fit_results.json")):
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

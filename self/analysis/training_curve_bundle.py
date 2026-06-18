"""Bundle loaders and summary tables for self-improvement curve notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from self.analysis.training_curve_logs import _to_float, load_round_metrics, parse_training_log
from self.analysis.training_curve_results import load_round_payload
from self.analysis.training_curve_style import BUDGET_ORDER, MODE_ORDER


@dataclass
class CurveBundle:
    jobs: pd.DataFrame
    train_logs: pd.DataFrame
    validation_logs: pd.DataFrame
    round_metrics: pd.DataFrame


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

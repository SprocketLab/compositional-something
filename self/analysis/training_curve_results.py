"""Result-file frame builders for training-curve notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_round_payload(results_path: Path) -> List[Dict[str, Any]]:
    """Load the raw round summaries from a results JSON file."""
    with Path(results_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of round summaries in {results_path}.")
    return [entry for entry in payload if isinstance(entry, dict)]


def resolve_results_path(path: str | Path) -> Path:
    """Resolve a run directory or direct JSON path to a results file."""
    path = Path(path)
    if path.is_dir():
        path = path / "self_improvement_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find self-improvement results at {path}.")
    return path


def round_summary_frame(
    results_path: str | Path,
    *,
    run_label: Optional[str] = None,
) -> pd.DataFrame:
    """Load round summaries from an arbitrary self_improvement_results.json file."""
    resolved_path = resolve_results_path(results_path)
    payload = load_round_payload(resolved_path)
    rows: List[Dict[str, Any]] = []
    for entry in payload:
        row = dict(entry)
        row["results_path"] = str(resolved_path)
        row["run_label"] = run_label if run_label is not None else resolved_path.parent.name
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and "round" in frame.columns:
        frame["round"] = frame["round"].astype(int)
        frame = frame.sort_values("round").reset_index(drop=True)
    return frame


def per_size_accuracy_frame_from_results(
    results_path: str | Path,
    *,
    run_label: Optional[str] = None,
) -> pd.DataFrame:
    """Expand per-size accuracy from an arbitrary self-improvement results file.

    Newer addition runs store size metadata as ``max_digits`` /
    ``per_digit_accuracy`` rather than the older generic ``max_size`` /
    ``per_size_accuracy`` keys, so both schemas are normalized here.
    """
    resolved_path = resolve_results_path(results_path)
    payload = load_round_payload(resolved_path)
    rows: List[Dict[str, Any]] = []
    for entry in payload:
        round_index = int(entry["round"])
        max_size_value = None
        for key in ("max_size", "max_digits", "max_bits", "max_ops"):
            candidate = entry.get(key)
            if candidate is not None:
                max_size_value = candidate
                break
        if max_size_value is None:
            raise KeyError(f"Could not find a size field in round entry from {resolved_path}.")

        per_size_accuracy = None
        for key in (
            "per_size_accuracy",
            "per_digit_accuracy",
            "per_bit_accuracy",
            "per_op_accuracy",
        ):
            candidate = entry.get(key)
            if isinstance(candidate, dict):
                per_size_accuracy = candidate
                break
        if not isinstance(per_size_accuracy, dict):
            continue
        for size, accuracy in per_size_accuracy.items():
            rows.append(
                {
                    "run_label": run_label if run_label is not None else resolved_path.parent.name,
                    "results_path": str(resolved_path),
                    "round": round_index,
                    "max_size": int(max_size_value),
                    "size": int(size),
                    "accuracy": _to_float(accuracy),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["size", "round"]).reset_index(drop=True)
    return frame

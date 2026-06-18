"""Training-log and round-metric parsers for self-improvement curves."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from self.analysis.artifact_io import read_round_summaries


ROUND_PATTERN = re.compile(r"\[ROUND\s+(\d+)\].*?eval_acc=([0-9.]+)")


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_training_log(log_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover per-log-step training loss and optional validation loss from Slurm stdout."""
    pending_train: List[Dict[str, Any]] = []
    pending_validation: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    train_summary: Optional[float] = None

    with Path(log_path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = ast.literal_eval(line)
                except (SyntaxError, ValueError):
                    payload = None
                if isinstance(payload, dict):
                    if "loss" in payload:
                        pending_train.append(
                            {
                                "epoch": _to_float(payload.get("epoch")),
                                "loss": _to_float(payload.get("loss")),
                                "grad_norm": _to_float(payload.get("grad_norm")),
                                "learning_rate": _to_float(payload.get("learning_rate")),
                                "line_number": line_number,
                            }
                        )
                        continue
                    if "eval_loss" in payload:
                        pending_validation.append(
                            {
                                "epoch": _to_float(payload.get("epoch")),
                                "validation_loss": _to_float(payload.get("eval_loss")),
                                "line_number": line_number,
                            }
                        )
                        continue
                    if "train_loss" in payload:
                        train_summary = _to_float(payload.get("train_loss"))
                        continue

            round_match = ROUND_PATTERN.search(line)
            if not round_match:
                continue

            round_index = int(round_match.group(1))
            for step_index, row in enumerate(pending_train, start=1):
                epoch = row.get("epoch")
                round_progress = round_index + (epoch if epoch is not None else 0.0)
                train_rows.append(
                    {
                        **row,
                        "round": round_index,
                        "step_in_round": step_index,
                        "round_progress": round_progress,
                        "train_loss_summary": train_summary,
                    }
                )
            for step_index, row in enumerate(pending_validation, start=1):
                epoch = row.get("epoch")
                round_progress = round_index + (epoch if epoch is not None else 0.0)
                validation_rows.append(
                    {
                        **row,
                        "round": round_index,
                        "step_in_round": step_index,
                        "round_progress": round_progress,
                    }
                )

            pending_train = []
            pending_validation = []
            train_summary = None

    return pd.DataFrame(train_rows), pd.DataFrame(validation_rows)


def load_round_metrics(results_path: Path) -> pd.DataFrame:
    """Load round-level accuracy metrics."""
    rows: List[Dict[str, Any]] = []
    for entry in read_round_summaries(results_path):
        rows.append(
            {
                "round": int(entry["round"]),
                "max_size": int(entry["max_size"]),
                "train_examples": int(entry["train_examples"]),
                "pseudo_examples": int(entry["pseudo_examples"]),
                "eval_accuracy": _to_float(entry.get("eval_accuracy")),
                "composed_eval_accuracy": _to_float(entry.get("composed_eval_accuracy")),
                "pseudo_retention_rate": _to_float(entry.get("pseudo_retention_rate")),
                "max_solved_size_at_90_accuracy": entry.get("max_solved_size_at_90_accuracy"),
            }
        )
    return pd.DataFrame(rows)

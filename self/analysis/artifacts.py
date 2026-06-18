"""Stable loaders for self-improvement artifacts.

Notebook code should use these helpers instead of hard-coding raw JSON paths.
The helpers intentionally return plain Python records so pandas remains
optional. Adaptive-run helpers are implemented in
``self.analysis.adaptive_artifacts`` and reexported here for compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from self.analysis.adaptive_artifacts import (
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    adaptive_attempt_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_proposal_grpo_records,
    adaptive_proposal_records,
    adaptive_selected_per_size_timeline_records,
    adaptive_trace_rows,
    discover_adaptive_runs,
    is_adaptive_run_dir,
    iter_attempt_dirs,
    load_adaptive_attempt,
    load_adaptive_run,
    load_adaptive_runs,
)
from self.analysis.artifact_io import (
    ADAPTIVE_RESULTS_FILE,
    SELF_IMPROVEMENT_RESULTS_FILE,
    JsonDict,
    read_json,
    read_jsonl,
)


def load_self_improvement_rounds(path_or_dir: Path | str) -> list[JsonDict]:
    resolved = Path(path_or_dir)
    path = resolved / SELF_IMPROVEMENT_RESULTS_FILE if resolved.is_dir() else resolved
    rows = read_json(path, []) or []
    if not isinstance(rows, list):
        raise ValueError(f"Expected list of round records in {path}.")
    return rows


def per_size_accuracy_records(
    rounds: Sequence[Mapping[str, Any]],
    *,
    run_name: str | None = None,
    metric_key: str = "per_size_accuracy",
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for round_record in rounds:
        round_index = round_record.get("round")
        accuracy_map = round_record.get(metric_key) or {}
        if not isinstance(accuracy_map, Mapping):
            continue
        for size, accuracy in accuracy_map.items():
            row: JsonDict = {
                "round": round_index,
                "size": int(size),
                "accuracy": accuracy,
            }
            if run_name is not None:
                row["run_name"] = run_name
            rows.append(row)
    return rows


def records_to_dataframe(records: Iterable[Mapping[str, Any]]) -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("records_to_dataframe requires pandas.") from exc
    return pd.DataFrame(list(records))

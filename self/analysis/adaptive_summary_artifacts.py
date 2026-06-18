"""Notebook-facing adaptive run summary tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from self.analysis.adaptive_artifact_common import AdaptiveRunArtifacts, _run_context
from self.analysis.adaptive_artifacts import (
    adaptive_attempt_records,
    adaptive_validity_summary_records,
    load_adaptive_run,
    load_adaptive_runs,
)
from self.analysis.adaptive_candidate_artifacts import (
    adaptive_candidate_artifact_records,
    adaptive_candidate_records,
    adaptive_local_dispatch_records,
)
from self.analysis.artifact_io import JsonDict

AdaptiveRunSource = AdaptiveRunArtifacts | Path | str | Iterable[AdaptiveRunArtifacts | Path | str]


def _coerce_adaptive_runs(source: AdaptiveRunSource) -> list[AdaptiveRunArtifacts]:
    if isinstance(source, AdaptiveRunArtifacts):
        return [source]
    if isinstance(source, (str, Path)):
        return load_adaptive_runs(source)

    runs: list[AdaptiveRunArtifacts] = []
    for item in source:
        if isinstance(item, AdaptiveRunArtifacts):
            runs.append(item)
        else:
            runs.append(load_adaptive_run(item))
    return runs


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _last_numeric(values: Iterable[Any]) -> Any:
    last = None
    for value in values:
        if value is not None:
            last = value
    return last


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _final_accuracy(run: AdaptiveRunArtifacts, selected_candidate_rows: list[JsonDict]) -> Any:
    if run.summary.get("final_accuracy") is not None:
        return run.summary.get("final_accuracy")
    selected_final = _last_numeric(row.get("final_accuracy") for row in selected_candidate_rows)
    if selected_final is not None:
        return selected_final
    if run.results:
        final_row = run.results[-1]
        if isinstance(final_row, dict):
            return final_row.get("final_accuracy", final_row.get("eval_accuracy"))
    return None


def adaptive_run_overview_records(source: AdaptiveRunSource) -> list[JsonDict]:
    """Return one compact summary row per adaptive run.

    The summary aggregates the common notebook counts without exposing raw
    ``attempt_*`` path conventions to notebooks.
    """

    rows: list[JsonDict] = []
    for run in _coerce_adaptive_runs(source):
        context = _run_context(run)
        attempt_rows = adaptive_attempt_records(run)
        validity_rows = adaptive_validity_summary_records(run)
        candidate_rows = adaptive_candidate_records(run)
        selected_candidate_rows = [
            row for row in candidate_rows if bool(row.get("selected_candidate"))
        ]
        candidate_artifact_rows = adaptive_candidate_artifact_records(run)
        local_dispatch_rows = adaptive_local_dispatch_records(run)

        proposal_count = sum(int(row.get("proposal_count") or 0) for row in validity_rows)
        valid_proposal_count = sum(
            int(row.get("valid_proposal_count") or 0) for row in validity_rows
        )
        invalid_proposal_count = sum(
            int(row.get("invalid_proposal_count") or 0) for row in validity_rows
        )
        candidate_count = len(candidate_rows)
        valid_candidate_count = sum(1 for row in candidate_rows if bool(row.get("valid")))
        final_accuracy = _final_accuracy(run, selected_candidate_rows)
        init_final_accuracy = context.get("init_final_accuracy")
        init_float = _float_or_none(init_final_accuracy)
        final_float = _float_or_none(final_accuracy)

        rows.append(
            {
                **context,
                "attempt_records": len(attempt_rows),
                "last_attempt": _last_numeric(row.get("attempt") for row in attempt_rows),
                "selected_attempts": sum(
                    1 for row in attempt_rows if row.get("selected_id") is not None
                ),
                "no_selection_attempts": sum(
                    1 for row in attempt_rows if bool(row.get("no_selection"))
                ),
                "proposal_count": proposal_count,
                "valid_proposal_count": valid_proposal_count,
                "invalid_proposal_count": invalid_proposal_count,
                "valid_proposal_rate": _rate(valid_proposal_count, proposal_count),
                "candidate_count": candidate_count,
                "valid_candidate_count": valid_candidate_count,
                "valid_candidate_rate": _rate(valid_candidate_count, candidate_count),
                "selected_candidate_count": len(selected_candidate_rows),
                "worker_failure_count": sum(
                    1 for row in candidate_artifact_rows if bool(row.get("has_worker_failure"))
                ),
                "missing_candidate_metrics_count": sum(
                    1 for row in candidate_artifact_rows if not bool(row.get("has_metrics"))
                ),
                "local_dispatch_attempts": sum(
                    1 for row in local_dispatch_rows if bool(row.get("has_local_dispatch"))
                ),
                "packed_local_dispatch_attempts": sum(
                    1 for row in local_dispatch_rows if bool(row.get("packed_workers"))
                ),
                "final_accuracy": final_accuracy,
                "final_accuracy_delta_from_init": (
                    final_float - init_float if final_float is not None and init_float is not None else None
                ),
            }
        )
    return rows


def adaptive_validity_summary_records_for_runs(
    source: AdaptiveRunSource,
    *,
    max_attempt: int | None = None,
) -> list[JsonDict]:
    """Return proposal-validity rows for every adaptive run under ``source``."""

    rows: list[JsonDict] = []
    for run in _coerce_adaptive_runs(source):
        for row in adaptive_validity_summary_records(run):
            attempt = row.get("attempt")
            if max_attempt is not None and attempt is not None and int(attempt) > max_attempt:
                continue
            rows.append(row)
    return rows


__all__ = [
    "adaptive_run_overview_records",
    "adaptive_validity_summary_records_for_runs",
]

"""Stable loaders for self-improvement and adaptive-run artifacts.

Notebook code should use these helpers instead of hard-coding raw JSON paths.
The helpers intentionally return plain Python records so pandas remains
optional.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JsonDict = dict[str, Any]

ADAPTIVE_RESULTS_FILE = "adaptive_candidate_training_results.json"
SELF_IMPROVEMENT_RESULTS_FILE = "self_improvement_results.json"


def read_json(path: Path | str, default: Any = None) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        return default
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path | str) -> list[JsonDict]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    rows: list[JsonDict] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for chunk in re.split(r"(\d+)", str(path)):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk)
    return tuple(parts)


def _attempt_index(path: Path, summary: Mapping[str, Any] | None = None) -> int | None:
    if summary and summary.get("attempt") is not None:
        return int(summary["attempt"])
    match = re.search(r"attempt_(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return None


def _run_context(run: "AdaptiveRunArtifacts") -> JsonDict:
    summary = run.summary
    return {
        "run_dir": str(run.path),
        "run_name": run.path.name,
        "task": summary.get("task"),
        "condition": summary.get("condition"),
        "selected_rounds_completed": summary.get("selected_rounds_completed"),
        "attempts_completed": summary.get("attempts_completed"),
        "init_final_accuracy": summary.get("init_final_accuracy"),
    }


def _proposal_fields(proposal: Any) -> JsonDict:
    if not isinstance(proposal, Mapping):
        return {}
    parsed = proposal.get("parsed_proposal")
    fields: JsonDict = {}
    if isinstance(parsed, Mapping):
        for key in ("left", "right", "target", "guard"):
            if key in parsed:
                fields[f"proposal_{key}"] = parsed[key]
    return fields


def _selected_id(selected: Any) -> Any:
    if isinstance(selected, Mapping):
        return selected.get("id")
    return selected


@dataclass(frozen=True)
class AdaptiveAttemptArtifacts:
    path: Path
    attempt_summary: JsonDict
    round_summary: JsonDict
    proposal_results: list[JsonDict]
    candidate_metrics: list[JsonDict]
    selected_candidate: JsonDict | None

    @property
    def attempt(self) -> int | None:
        return _attempt_index(self.path, self.attempt_summary)


@dataclass(frozen=True)
class AdaptiveRunArtifacts:
    path: Path
    summary: JsonDict
    results: list[JsonDict]
    seed_metrics: JsonDict
    attempts: tuple[AdaptiveAttemptArtifacts, ...]


def is_adaptive_run_dir(path: Path | str) -> bool:
    resolved = Path(path)
    if not resolved.is_dir():
        return False
    if (resolved / ADAPTIVE_RESULTS_FILE).exists():
        return True
    return any(resolved.glob("attempt_*/attempt_summary.json"))


def discover_adaptive_runs(root: Path | str) -> list[Path]:
    resolved = Path(root)
    if is_adaptive_run_dir(resolved):
        return [resolved]
    if not resolved.exists():
        return []
    candidates = {
        path.parent
        for path in resolved.rglob("summary.json")
        if is_adaptive_run_dir(path.parent)
    }
    candidates.update(
        path.parent
        for path in resolved.rglob(ADAPTIVE_RESULTS_FILE)
        if is_adaptive_run_dir(path.parent)
    )
    return sorted(candidates, key=_natural_sort_key)


def iter_attempt_dirs(run_dir: Path | str) -> list[Path]:
    resolved = Path(run_dir)
    return sorted(
        [path for path in resolved.glob("attempt_*") if path.is_dir()],
        key=_natural_sort_key,
    )


def load_adaptive_attempt(attempt_dir: Path | str) -> AdaptiveAttemptArtifacts:
    resolved = Path(attempt_dir)
    selected = read_json(resolved / "selected_candidate.json", None)
    if selected is not None and not isinstance(selected, dict):
        selected = {"value": selected}
    return AdaptiveAttemptArtifacts(
        path=resolved,
        attempt_summary=read_json(resolved / "attempt_summary.json", {}) or {},
        round_summary=read_json(resolved / "round_summary.json", {}) or {},
        proposal_results=read_json(resolved / "proposal_results.json", []) or [],
        candidate_metrics=read_json(resolved / "candidate_metrics.json", []) or [],
        selected_candidate=selected,
    )


def load_adaptive_run(run_dir: Path | str) -> AdaptiveRunArtifacts:
    resolved = Path(run_dir)
    attempts = tuple(load_adaptive_attempt(path) for path in iter_attempt_dirs(resolved))
    return AdaptiveRunArtifacts(
        path=resolved,
        summary=read_json(resolved / "summary.json", {}) or {},
        results=read_json(resolved / ADAPTIVE_RESULTS_FILE, []) or [],
        seed_metrics=read_json(resolved / "round_00" / "metrics.json", {}) or {},
        attempts=attempts,
    )


def load_adaptive_runs(root: Path | str) -> list[AdaptiveRunArtifacts]:
    return [load_adaptive_run(path) for path in discover_adaptive_runs(root)]


def adaptive_attempt_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        summary = attempt.attempt_summary
        selected = summary.get("selected")
        selected_id = _selected_id(selected)
        proposal_count = len(attempt.proposal_results)
        valid_proposal_count = sum(1 for proposal in attempt.proposal_results if bool(proposal.get("valid")))
        row: JsonDict = {
            **context,
            "attempt_dir": str(attempt.path),
            "attempt": attempt.attempt,
            "selected_round": summary.get("selected_round"),
            "no_selection": bool(summary.get("no_selection", False)),
            "candidate_count": summary.get("candidate_count", len(attempt.candidate_metrics)),
            "proposal_count": proposal_count,
            "valid_proposal_count": valid_proposal_count,
            "selected_id": selected_id,
            "trace_count": summary.get("trace_count"),
            "outcome_trace_count": summary.get("outcome_trace_count"),
            "proposal_trace_buffer_size": summary.get("proposal_trace_buffer_size"),
            "outcome_trace_buffer_size": summary.get("outcome_trace_buffer_size"),
        }
        rows.append(row)
    return rows


def adaptive_proposal_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        attempt_summary = attempt.attempt_summary
        for proposal in attempt.proposal_results:
            row: JsonDict = {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt_summary.get("selected_round"),
                "proposal_id": proposal.get("id"),
                "proposal_index": proposal.get("proposal_index"),
                "valid": proposal.get("valid"),
                "validation_category": proposal.get("validation_category"),
                "validation_message": proposal.get("validation_message"),
                "duplicate": proposal.get("duplicate"),
                "repeat_target": proposal.get("repeat_target"),
                "parsed_proposal": proposal.get("parsed_proposal"),
                "completion": proposal.get("completion"),
                "raw_output": proposal.get("raw_output"),
            }
            row.update(_proposal_fields(proposal))
            rows.append(row)
    return rows


def adaptive_candidate_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        selected_id = _selected_id(attempt.attempt_summary.get("selected"))
        for candidate in attempt.candidate_metrics:
            row: JsonDict = {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "candidate_id": candidate.get("id"),
                "candidate_index": candidate.get("index"),
                "selected_candidate": candidate.get("id") == selected_id if selected_id is not None else None,
                "valid": candidate.get("valid"),
                "reward": candidate.get("reward"),
                "final_accuracy": candidate.get("final_accuracy"),
                "final_accuracy_delta": candidate.get("final_accuracy_delta"),
                "final_accuracy_delta_from_current": candidate.get("final_accuracy_delta_from_current"),
                "frontier_delta": candidate.get("frontier_delta"),
                "target_accuracy": candidate.get("target_accuracy"),
                "failure_reason": candidate.get("failure_reason"),
                "parsed_proposal": candidate.get("parsed_proposal"),
                "per_size_accuracy": candidate.get("per_size_accuracy"),
            }
            row.update(_proposal_fields(candidate))
            rows.append(row)
    return rows


def adaptive_proposal_grpo_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        metrics_path = attempt.path / "proposal_grpo" / "proposal_grpo_metrics.json"
        metrics = read_json(metrics_path, None)
        if not isinstance(metrics, Mapping):
            continue
        row: JsonDict = {
            **context,
            "attempt_dir": str(attempt.path),
            "attempt": attempt.attempt,
            "selected_round": attempt.attempt_summary.get("selected_round"),
            "proposal_grpo_metrics_path": str(metrics_path),
        }
        row.update(metrics)
        rows.append(row)
    return rows


def adaptive_trace_rows(
    attempt: AdaptiveAttemptArtifacts | Path | str,
    *,
    name: str = "trace_examples.jsonl",
) -> list[JsonDict]:
    attempt_dir = attempt.path if isinstance(attempt, AdaptiveAttemptArtifacts) else Path(attempt)
    return read_jsonl(attempt_dir / name)


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

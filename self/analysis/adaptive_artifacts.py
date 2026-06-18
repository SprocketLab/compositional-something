#!/usr/bin/env python3
"""Adaptive run artifact discovery and table-building helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, read_json


DEFAULT_ADAPTIVE_TRACE_FILES = (
    "trace_examples.jsonl",
    "selected_proposal_trace.jsonl",
    "outcome_trace_examples.jsonl",
    "proposal_grpo/proposal_grpo_traces.jsonl",
)

ADAPTIVE_CANDIDATE_METRICS_FILE = "candidate_metrics.json"
ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE = "train_mix_summary.json"
ADAPTIVE_CANDIDATE_FAILURE_FILE = "worker_failure.json"
ADAPTIVE_LOCAL_DISPATCH_FILE = "candidate_jobs/local_dispatch.json"


def _attempt_index(path: Path, summary: Mapping[str, Any] | None = None) -> int | None:
    if summary and summary.get("attempt") is not None:
        return int(summary["attempt"])
    match = re.search(r"attempt_(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return None


def _candidate_index(path: Path, payload: Mapping[str, Any] | None = None) -> int | None:
    if payload and payload.get("index") is not None:
        return int(payload["index"])
    match = re.search(r"candidate_(\d+)$", path.name)
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


def _attempt_selected_payload(attempt: "AdaptiveAttemptArtifacts") -> JsonDict | None:
    for selected in (
        attempt.selected_candidate,
        attempt.round_summary.get("selected"),
        attempt.attempt_summary.get("selected"),
    ):
        if isinstance(selected, Mapping):
            return dict(selected)
    return None


def _candidate_metric_for_dir(
    attempt: "AdaptiveAttemptArtifacts",
    candidate_dir: Path,
) -> JsonDict:
    metric = read_json(candidate_dir / ADAPTIVE_CANDIDATE_METRICS_FILE, None)
    if isinstance(metric, Mapping):
        return dict(metric)
    candidate_index = _candidate_index(candidate_dir)
    for item in attempt.candidate_metrics:
        if not isinstance(item, Mapping):
            continue
        if candidate_index is not None and item.get("index") == candidate_index:
            return dict(item)
        if item.get("id") == candidate_dir.name:
            return dict(item)
    return {}


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


from collections import Counter
from pathlib import Path
import re

from self.analysis.artifact_io import (
    ADAPTIVE_RESULTS_FILE,
    JsonDict,
    natural_sort_key,
    read_json,
)


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
    return sorted(candidates, key=natural_sort_key)


def iter_attempt_dirs(run_dir: Path | str) -> list[Path]:
    resolved = Path(run_dir)
    return sorted(
        [path for path in resolved.glob("attempt_*") if path.is_dir()],
        key=natural_sort_key,
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
        valid_proposal_count = sum(
            1 for proposal in attempt.proposal_results if bool(proposal.get("valid"))
        )
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


def _validation_count_key(category: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", category.strip().lower()).strip("_")
    return f"validation_{normalized or 'unknown'}_count"


def adaptive_validity_summary_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    """Return per-attempt proposal validity summaries for analysis notebooks."""

    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        proposal_count = len(attempt.proposal_results)
        valid_count = sum(
            1 for proposal in attempt.proposal_results if bool(proposal.get("valid"))
        )
        invalid_count = proposal_count - valid_count
        categories = Counter(
            str(
                proposal.get("validation_category")
                or ("valid" if bool(proposal.get("valid")) else "unknown")
            )
            for proposal in attempt.proposal_results
        )
        row: JsonDict = {
            **context,
            "attempt_dir": str(attempt.path),
            "attempt": attempt.attempt,
            "selected_round": attempt.attempt_summary.get("selected_round"),
            "no_selection": bool(attempt.attempt_summary.get("no_selection", False)),
            "proposal_count": proposal_count,
            "valid_proposal_count": valid_count,
            "invalid_proposal_count": invalid_count,
            "valid_rate": valid_count / proposal_count if proposal_count else None,
            "selected_id": _selected_id(attempt.attempt_summary.get("selected")),
            "validation_category_counts": dict(sorted(categories.items())),
        }
        for category, count in sorted(categories.items()):
            row[_validation_count_key(category)] = count
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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, natural_sort_key, read_json


@dataclass(frozen=True)
class AdaptiveCandidateArtifacts:
    path: Path
    metrics: JsonDict
    train_mix_summary: JsonDict
    worker_failure: JsonDict | None

    @property
    def candidate_index(self) -> int | None:
        return _candidate_index(self.path, self.metrics)

    @property
    def candidate_id(self) -> Any:
        return self.metrics.get("id", self.path.name)

    @property
    def metrics_path(self) -> Path:
        return self.path / ADAPTIVE_CANDIDATE_METRICS_FILE

    @property
    def train_mix_summary_path(self) -> Path:
        return self.path / ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE

    @property
    def worker_failure_path(self) -> Path:
        return self.path / ADAPTIVE_CANDIDATE_FAILURE_FILE


def iter_candidate_dirs(attempt: AdaptiveAttemptArtifacts | Path | str) -> list[Path]:
    attempt_dir = attempt.path if isinstance(attempt, AdaptiveAttemptArtifacts) else Path(attempt)
    return sorted(
        [path for path in (attempt_dir / "candidates").glob("candidate_*") if path.is_dir()],
        key=natural_sort_key,
    )


def load_adaptive_candidate(
    candidate_dir: Path | str,
    *,
    attempt: AdaptiveAttemptArtifacts | None = None,
) -> AdaptiveCandidateArtifacts:
    resolved = Path(candidate_dir)
    worker_failure = read_json(resolved / ADAPTIVE_CANDIDATE_FAILURE_FILE, None)
    if worker_failure is not None and not isinstance(worker_failure, Mapping):
        worker_failure = {"value": worker_failure}
    return AdaptiveCandidateArtifacts(
        path=resolved,
        metrics=(
            _candidate_metric_for_dir(attempt, resolved)
            if attempt is not None
            else read_json(resolved / ADAPTIVE_CANDIDATE_METRICS_FILE, {}) or {}
        ),
        train_mix_summary=read_json(resolved / ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE, {}) or {},
        worker_failure=worker_failure,
    )


def load_adaptive_candidates(
    attempt: AdaptiveAttemptArtifacts | Path | str,
) -> tuple[AdaptiveCandidateArtifacts, ...]:
    artifacts = load_adaptive_attempt(attempt) if not isinstance(attempt, AdaptiveAttemptArtifacts) else attempt
    return tuple(
        load_adaptive_candidate(candidate_dir, attempt=artifacts)
        for candidate_dir in iter_candidate_dirs(artifacts)
    )


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
                "selected_candidate": (
                    candidate.get("id") == selected_id if selected_id is not None else None
                ),
                "valid": candidate.get("valid"),
                "reward": candidate.get("reward"),
                "final_accuracy": candidate.get("final_accuracy"),
                "final_accuracy_delta": candidate.get("final_accuracy_delta"),
                "final_accuracy_delta_from_current": candidate.get(
                    "final_accuracy_delta_from_current"
                ),
                "frontier_delta": candidate.get("frontier_delta"),
                "target_accuracy": candidate.get("target_accuracy"),
                "failure_reason": candidate.get("failure_reason"),
                "parsed_proposal": candidate.get("parsed_proposal"),
                "per_size_accuracy": candidate.get("per_size_accuracy"),
            }
            row.update(_proposal_fields(candidate))
            rows.append(row)
    return rows


def adaptive_candidate_artifact_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        selected_id = _selected_id(attempt.attempt_summary.get("selected"))
        for candidate in load_adaptive_candidates(attempt):
            candidate_id = candidate.candidate_id
            metric = candidate.metrics
            row: JsonDict = {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "candidate_dir": str(candidate.path),
                "candidate_index": candidate.candidate_index,
                "candidate_id": candidate_id,
                "selected_candidate": (
                    candidate_id == selected_id if selected_id is not None else None
                ),
                "valid": metric.get("valid"),
                "reward": metric.get("reward"),
                "final_accuracy": metric.get("final_accuracy"),
                "frontier_delta": metric.get("frontier_delta"),
                "metrics_path": str(candidate.metrics_path),
                "train_mix_summary_path": str(candidate.train_mix_summary_path),
                "worker_failure_path": str(candidate.worker_failure_path),
                "has_metrics": bool(metric),
                "has_train_mix_summary": bool(candidate.train_mix_summary),
                "has_worker_failure": candidate.worker_failure is not None,
                "worker_failure": candidate.worker_failure,
            }
            row.update(_proposal_fields(metric))
            rows.append(row)
    return rows


def adaptive_candidate_train_mix_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        selected_id = _selected_id(attempt.attempt_summary.get("selected"))
        for candidate in load_adaptive_candidates(attempt):
            train_mix = candidate.train_mix_summary
            if not train_mix:
                continue
            metric = candidate.metrics
            candidate_id = candidate.candidate_id
            row: JsonDict = {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "candidate_dir": str(candidate.path),
                "candidate_index": candidate.candidate_index,
                "candidate_id": candidate_id,
                "selected_candidate": (
                    candidate_id == selected_id if selected_id is not None else None
                ),
                "valid": metric.get("valid"),
                "reward": metric.get("reward"),
                "final_accuracy": metric.get("final_accuracy"),
                "frontier_delta": metric.get("frontier_delta"),
                "train_mix_summary_path": str(candidate.train_mix_summary_path),
            }
            row.update(_proposal_fields(metric))
            row.update(dict(train_mix))
            rows.append(row)
    return rows


def load_adaptive_local_dispatch(attempt: AdaptiveAttemptArtifacts | Path | str) -> JsonDict:
    artifacts = load_adaptive_attempt(attempt) if not isinstance(attempt, AdaptiveAttemptArtifacts) else attempt
    payload = read_json(artifacts.path / ADAPTIVE_LOCAL_DISPATCH_FILE, {}) or {}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"value": payload}


def adaptive_local_dispatch_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        dispatch = load_adaptive_local_dispatch(attempt)
        cache_plan = dispatch.get("cache_plan") if isinstance(dispatch.get("cache_plan"), Mapping) else {}
        planned_units = dispatch.get("planned_units") if isinstance(dispatch.get("planned_units"), list) else []
        launched = dispatch.get("launched") if isinstance(dispatch.get("launched"), list) else []
        active_pids = dispatch.get("active_pids") if isinstance(dispatch.get("active_pids"), list) else []
        rows.append(
            {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "local_dispatch_path": str(attempt.path / ADAPTIVE_LOCAL_DISPATCH_FILE),
                "has_local_dispatch": bool(dispatch),
                "candidate_count": dispatch.get("candidate_count"),
                "planned_processes": dispatch.get("planned_processes"),
                "max_parallel": dispatch.get("max_parallel"),
                "pack_size": dispatch.get("pack_size"),
                "packed_workers": dispatch.get("packed_workers"),
                "pending": dispatch.get("pending"),
                "launched_processes": len(launched),
                "active_processes": len(active_pids),
                "cache_shared_input": cache_plan.get("shared_input_cache"),
                "cache_tokenizer_bootstrap": cache_plan.get("tokenizer_bootstrap_cache"),
                "cache_base_state": cache_plan.get("base_state_cache"),
                "planned_candidate_groups": [
                    unit.get("candidate_indices")
                    for unit in planned_units
                    if isinstance(unit, Mapping)
                ],
                "planned_units": planned_units,
            }
        )
    return rows


def adaptive_candidate_per_size_records(
    run: AdaptiveRunArtifacts | Path | str,
    *,
    metric_key: str = "per_size_accuracy",
) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        selected_id = _selected_id(attempt.attempt_summary.get("selected"))
        for candidate in attempt.candidate_metrics:
            accuracy_map = candidate.get(metric_key) or {}
            if not isinstance(accuracy_map, Mapping):
                continue
            selected_candidate = (
                candidate.get("id") == selected_id if selected_id is not None else None
            )
            base: JsonDict = {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "candidate_id": candidate.get("id"),
                "candidate_index": candidate.get("index"),
                "selected_candidate": selected_candidate,
                "valid": candidate.get("valid"),
                "reward": candidate.get("reward"),
                "metric_key": metric_key,
            }
            base.update(_proposal_fields(candidate))
            for size, accuracy in accuracy_map.items():
                rows.append(
                    {
                        **base,
                        "size": int(size),
                        "accuracy": accuracy,
                    }
                )
    return rows


from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, read_json, read_jsonl


def _as_adaptive_run(run: AdaptiveRunArtifacts | Path | str) -> AdaptiveRunArtifacts:
    if isinstance(run, AdaptiveRunArtifacts):
        return run
    from self.analysis.adaptive_artifacts import load_adaptive_run

    return load_adaptive_run(run)


def adaptive_prompt_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = _as_adaptive_run(run)
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        prompt_path = attempt.path / "proposal_prompt.json"
        prompt = read_json(prompt_path, None)
        if not isinstance(prompt, Mapping):
            continue
        system = prompt.get("system")
        user = prompt.get("user")
        rows.append(
            {
                **context,
                "attempt_dir": str(attempt.path),
                "attempt": attempt.attempt,
                "selected_round": attempt.attempt_summary.get("selected_round"),
                "proposal_prompt_path": str(prompt_path),
                "system": system,
                "user": user,
                "system_chars": len(system) if isinstance(system, str) else None,
                "user_chars": len(user) if isinstance(user, str) else None,
            }
        )
    return rows


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def adaptive_selected_per_size_timeline_records(
    run: AdaptiveRunArtifacts | Path | str,
    *,
    metric_key: str = "per_size_accuracy",
) -> list[JsonDict]:
    artifacts = _as_adaptive_run(run)
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    current_accuracy = artifacts.seed_metrics.get(metric_key) or {}
    if not isinstance(current_accuracy, Mapping):
        current_accuracy = {}
    current_final_accuracy = artifacts.seed_metrics.get("eval_accuracy")
    selected_round = 0

    def append_rows(
        *,
        attempt_index: int | None,
        attempt_dir: str | None,
        selected_this_attempt: bool,
        checkpoint_source: str,
        selected: Mapping[str, Any] | None,
    ) -> None:
        base: JsonDict = {
            **context,
            "attempt_dir": attempt_dir,
            "attempt": attempt_index,
            "selected_round": selected_round,
            "selected_this_attempt": selected_this_attempt,
            "checkpoint_source": checkpoint_source,
            "checkpoint_final_accuracy": current_final_accuracy,
            "selected_id": selected.get("id") if selected is not None else None,
            "metric_key": metric_key,
        }
        if selected is not None:
            base.update(_proposal_fields(selected))
        target_size = _coerce_int(base.get("proposal_target"))
        base["target_size"] = target_size
        for size, accuracy in sorted(current_accuracy.items(), key=lambda item: int(item[0])):
            size_value = int(size)
            rows.append(
                {
                    **base,
                    "size": size_value,
                    "accuracy": accuracy,
                    "is_target_size": (
                        selected_this_attempt
                        and target_size is not None
                        and size_value == target_size
                    ),
                }
            )

    append_rows(
        attempt_index=0,
        attempt_dir=None,
        selected_this_attempt=False,
        checkpoint_source="seed",
        selected=None,
    )
    for attempt in artifacts.attempts:
        selected = _attempt_selected_payload(attempt)
        selected_accuracy = selected.get(metric_key) if selected is not None else None
        selected_this_attempt = isinstance(selected_accuracy, Mapping)
        checkpoint_source = "carried_forward"
        if selected_this_attempt:
            current_accuracy = selected_accuracy
            selected_round = int(
                attempt.attempt_summary.get("selected_round")
                or attempt.round_summary.get("selected_round")
                or selected_round + 1
            )
            current_final_accuracy = selected.get(
                "final_accuracy",
                selected.get("eval_accuracy", current_final_accuracy),
            )
            checkpoint_source = "selected_candidate"
        append_rows(
            attempt_index=attempt.attempt,
            attempt_dir=str(attempt.path),
            selected_this_attempt=selected_this_attempt,
            checkpoint_source=checkpoint_source,
            selected=selected if selected_this_attempt else None,
        )
    return rows


def adaptive_proposal_grpo_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = _as_adaptive_run(run)
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


def adaptive_trace_records(
    run: AdaptiveRunArtifacts | Path | str,
    *,
    names: tuple[str, ...] = DEFAULT_ADAPTIVE_TRACE_FILES,
) -> list[JsonDict]:
    artifacts = _as_adaptive_run(run)
    context = _run_context(artifacts)
    rows: list[JsonDict] = []
    for attempt in artifacts.attempts:
        for name in names:
            trace_path = attempt.path / name
            for trace_index, trace in enumerate(read_jsonl(trace_path)):
                rows.append(
                    {
                        **context,
                        "attempt_dir": str(attempt.path),
                        "attempt": attempt.attempt,
                        "selected_round": attempt.attempt_summary.get("selected_round"),
                        "trace_name": name,
                        "trace_path": str(trace_path),
                        "trace_index": trace_index,
                        "trace": trace,
                    }
                )
    return rows


from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, natural_sort_key, read_json


SUBMISSION_MANIFEST_FILE = "submission_manifest.json"


def resolve_submission_manifest_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_file():
        return resolved
    return resolved / SUBMISSION_MANIFEST_FILE


def load_submission_manifest(path: Path | str, default: Any = None) -> JsonDict:
    payload = read_json(resolve_submission_manifest_path(path), default if default is not None else {})
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object payload in submission manifest: {path}")
    return dict(payload)


def discover_submission_manifests(root: Path | str) -> list[Path]:
    resolved = Path(root)
    if resolved.is_file():
        return [resolved] if resolved.name == SUBMISSION_MANIFEST_FILE else []
    manifest = resolved / SUBMISSION_MANIFEST_FILE
    if manifest.exists():
        return [manifest]
    if not resolved.exists():
        return []
    return sorted(resolved.rglob(SUBMISSION_MANIFEST_FILE), key=natural_sort_key)


def adaptive_submission_job_records(root: Path | str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for manifest_path in discover_submission_manifests(root):
        payload = load_submission_manifest(manifest_path)
        jobs = payload.get("jobs") or {}
        if not isinstance(jobs, Mapping):
            continue
        slurm = payload.get("slurm") or {}
        if not isinstance(slurm, Mapping):
            slurm = {}
        for job_key, job in jobs.items():
            if not isinstance(job, Mapping):
                job = {"value": job}
            output_dir = job.get("output_dir", job.get("output_root"))
            row: JsonDict = {
                "manifest_path": str(manifest_path),
                "manifest_dir": str(manifest_path.parent),
                "manifest_name": manifest_path.parent.name,
                "out_root": payload.get("out_root"),
                "job_key": job_key,
                "job_id": job.get("job_id"),
                "status": job.get("status"),
                "output_dir": output_dir,
                "task": job.get("task"),
                "condition": job.get("condition"),
                "outcome_trace_target_mode": job.get("outcome_trace_target_mode"),
                "proposal_grpo_zero_variance": job.get("proposal_grpo_zero_variance"),
                "num_candidates": job.get("num_candidates"),
                "target_mode": job.get("target_mode"),
                "composition_path": job.get("composition_path"),
            }
            for key, value in slurm.items():
                row[f"slurm_{key}"] = value
            for key, value in job.items():
                row.setdefault(key, value)
            rows.append(row)
    return rows


__all__ = [
    "SUBMISSION_MANIFEST_FILE",
    "adaptive_submission_job_records",
    "discover_submission_manifests",
    "load_submission_manifest",
    "resolve_submission_manifest_path",
]


from collections.abc import Iterable
from pathlib import Path
from typing import Any

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

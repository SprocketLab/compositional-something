"""Candidate-level artifact helpers for adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from self.analysis.adaptive_artifact_common import (
    ADAPTIVE_CANDIDATE_FAILURE_FILE,
    ADAPTIVE_CANDIDATE_METRICS_FILE,
    ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE,
    ADAPTIVE_LOCAL_DISPATCH_FILE,
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    _candidate_index,
    _candidate_metric_for_dir,
    _proposal_fields,
    _run_context,
    _selected_id,
)
from self.analysis.adaptive_artifacts import (
    load_adaptive_attempt,
    load_adaptive_run,
)
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

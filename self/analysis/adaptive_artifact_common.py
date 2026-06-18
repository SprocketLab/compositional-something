"""Shared models and helpers for adaptive artifact loaders."""

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

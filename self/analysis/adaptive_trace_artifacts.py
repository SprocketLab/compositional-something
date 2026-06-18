"""Adaptive prompt, trace, and selected-checkpoint timeline artifact rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self.analysis.adaptive_artifact_common import (
    DEFAULT_ADAPTIVE_TRACE_FILES,
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    _attempt_selected_payload,
    _proposal_fields,
    _run_context,
)
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
        for size, accuracy in sorted(current_accuracy.items(), key=lambda item: int(item[0])):
            rows.append(
                {
                    **base,
                    "size": int(size),
                    "accuracy": accuracy,
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

"""Adaptive self-improvement artifact loaders and row flatteners."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from self.analysis.adaptive_artifact_common import (
    ADAPTIVE_CANDIDATE_FAILURE_FILE,
    ADAPTIVE_CANDIDATE_METRICS_FILE,
    ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE,
    DEFAULT_ADAPTIVE_TRACE_FILES,
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    _attempt_selected_payload,
    _proposal_fields,
    _run_context,
    _selected_id,
)
from self.analysis.artifact_io import (
    ADAPTIVE_RESULTS_FILE,
    JsonDict,
    natural_sort_key,
    read_json,
    read_jsonl,
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


def adaptive_prompt_records(run: AdaptiveRunArtifacts | Path | str) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
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
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
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


def adaptive_trace_records(
    run: AdaptiveRunArtifacts | Path | str,
    *,
    names: tuple[str, ...] = DEFAULT_ADAPTIVE_TRACE_FILES,
) -> list[JsonDict]:
    artifacts = load_adaptive_run(run) if not isinstance(run, AdaptiveRunArtifacts) else run
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


from self.analysis.adaptive_candidate_artifacts import (  # noqa: E402
    AdaptiveCandidateArtifacts,
    adaptive_candidate_artifact_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_candidate_train_mix_records,
    iter_candidate_dirs,
    load_adaptive_candidate,
    load_adaptive_candidates,
)

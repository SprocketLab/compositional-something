"""Adaptive self-improvement artifact loaders and row flatteners."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from self.analysis.adaptive_artifact_common import (
    ADAPTIVE_CANDIDATE_FAILURE_FILE,
    ADAPTIVE_CANDIDATE_METRICS_FILE,
    ADAPTIVE_CANDIDATE_TRAIN_MIX_FILE,
    DEFAULT_ADAPTIVE_TRACE_FILES,
    AdaptiveAttemptArtifacts,
    AdaptiveRunArtifacts,
    _proposal_fields,
    _run_context,
    _selected_id,
)
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


from self.analysis.adaptive_candidate_artifacts import (  # noqa: E402
    AdaptiveCandidateArtifacts,
    adaptive_candidate_artifact_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_candidate_train_mix_records,
    adaptive_local_dispatch_records,
    iter_candidate_dirs,
    load_adaptive_candidate,
    load_adaptive_candidates,
    load_adaptive_local_dispatch,
)
from self.analysis.adaptive_trace_artifacts import (  # noqa: E402
    adaptive_prompt_records,
    adaptive_proposal_grpo_records,
    adaptive_selected_per_size_timeline_records,
    adaptive_trace_records,
    adaptive_trace_rows,
)

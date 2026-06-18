"""Candidate-worker failure artifact helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping

from self.core import worker_io


JsonDict = Dict[str, Any]


def candidate_worker_failure_payload(spec_path: Path, exc: Exception) -> JsonDict:
    return {
        "spec_path": str(spec_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def candidate_worker_failure_path_from_payload(payload: Mapping[str, Any]) -> Path:
    candidate_index = int(payload["candidate_index"])
    round_dir = Path(str(payload["round_dir"]))
    return worker_io.candidate_worker_failure_path(round_dir, candidate_index)


def write_candidate_worker_failure(
    *,
    spec_path: Path,
    spec_payload: Mapping[str, Any],
    exc: Exception,
    write_json_fn: Callable[[Path, Any], None],
) -> JsonDict:
    failure_payload = candidate_worker_failure_payload(spec_path, exc)
    write_json_fn(candidate_worker_failure_path_from_payload(spec_payload), failure_payload)
    return failure_payload


def write_candidate_worker_failure_from_spec(
    spec_path: Path,
    exc: Exception,
    *,
    load_json_fn: Callable[[Path], Any],
    write_json_fn: Callable[[Path, Any], None],
) -> JsonDict:
    spec_payload = load_json_fn(spec_path)
    return write_candidate_worker_failure(
        spec_path=spec_path,
        spec_payload=spec_payload,
        exc=exc,
        write_json_fn=write_json_fn,
    )

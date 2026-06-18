from __future__ import annotations

from pathlib import Path

from self.adaptive.candidates.workers import (
    candidate_worker_failure_path_from_payload,
    candidate_worker_failure_payload,
    write_candidate_worker_failure_from_spec,
)


def test_candidate_worker_failure_payload_and_path_use_existing_layout(tmp_path: Path):
    spec_path = tmp_path / "candidate_jobs" / "specs" / "candidate_0.json"
    exc = RuntimeError("CUDA out of memory")
    payload = {
        "round_dir": str(tmp_path / "attempt_0001"),
        "candidate_index": 7,
    }

    assert candidate_worker_failure_payload(spec_path, exc) == {
        "spec_path": str(spec_path),
        "error_type": "RuntimeError",
        "error": "CUDA out of memory",
    }
    assert candidate_worker_failure_path_from_payload(payload) == (
        tmp_path / "attempt_0001" / "candidates" / "candidate_07" / "worker_failure.json"
    )


def test_write_candidate_worker_failure_from_spec_uses_injected_io(tmp_path: Path):
    spec_path = tmp_path / "candidate_jobs" / "specs" / "candidate_0.json"
    spec_payload = {
        "round_dir": str(tmp_path / "attempt_0001"),
        "candidate_index": 2,
    }
    writes = []

    failure = write_candidate_worker_failure_from_spec(
        spec_path,
        ValueError("bad candidate"),
        load_json_fn=lambda path: spec_payload,
        write_json_fn=lambda path, payload: writes.append((path, payload)),
    )

    expected = {
        "spec_path": str(spec_path),
        "error_type": "ValueError",
        "error": "bad candidate",
    }
    assert failure == expected
    assert writes == [
        (
            tmp_path / "attempt_0001" / "candidates" / "candidate_02" / "worker_failure.json",
            expected,
        )
    ]

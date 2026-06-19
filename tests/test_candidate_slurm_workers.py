from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from self.adaptive import candidate as workers
from self.adaptive.candidate import train_candidates_slurm_array_from_specs, wait_for_candidate_array


def _args(**overrides):
    args = dict(
        candidate_array_sbatch_script=Path("worker.sbatch"),
        candidate_array_max_parallel=3,
        candidate_array_time_limit="02:00:00",
        candidate_array_timeout_seconds=0.0,
        candidate_array_poll_seconds=0.1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_candidate_workers_submit_wrapper_uses_module_level_submit_binding(tmp_path, monkeypatch):
    calls = []

    def fake_submit_sbatch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            job_id="12345",
            command=["sbatch", "--array", kwargs["array_spec"]],
            stdout="12345",
            stderr="",
        )

    monkeypatch.setattr(workers, "submit_sbatch", fake_submit_sbatch)
    monkeypatch.setattr(workers.sys, "executable", "/tmp/python")

    job_id = workers.submit_candidate_array(
        args=_args(),
        round_dir=tmp_path / "attempt_0001",
        spec_paths=[tmp_path / "spec0.json", tmp_path / "spec1.json"],
    )

    assert job_id == "12345"
    assert calls[0]["array_spec"] == "0-1%3"
    assert calls[0]["job_name"] == "adaptive-cand-attempt_0001"
    assert calls[0]["time_limit"] == "02:00:00"
    assert calls[0]["exports"]["PYTHON_BIN"] == "/tmp/python"
    assert calls[0]["exports"]["CANDIDATE_SPEC_DIR"] == tmp_path / "attempt_0001" / "candidate_jobs" / "specs"
    dispatch = json.loads(
        (tmp_path / "attempt_0001" / "candidate_jobs" / "slurm_dispatch.json").read_text(encoding="utf-8")
    )
    assert dispatch["job_id"] == "12345"
    assert dispatch["array_spec"] == "0-1%3"
    assert dispatch["command"] == ["sbatch", "--array", "0-1%3"]


def test_wait_for_candidate_array_timeout_cancels_and_writes_timeout(tmp_path):
    cancelled = []
    work_items = [SimpleNamespace(index=0)]

    wait_for_candidate_array(
        args=_args(candidate_array_timeout_seconds=1.0),
        round_dir=tmp_path / "attempt_0001",
        work_items=work_items,
        job_id="job-7",
        cancel_job_fn=lambda job_id: cancelled.append(job_id),
        slurm_job_active_fn=lambda job_id: True,
        monotonic_fn=iter([0.0, 2.5]).__next__,
        sleep_fn=lambda seconds: None,
    )

    assert cancelled == ["job-7"]
    timeout = json.loads(
        (tmp_path / "attempt_0001" / "candidate_jobs" / "slurm_timeout.json").read_text(encoding="utf-8")
    )
    assert timeout == {
        "elapsed_seconds": 2.5,
        "job_id": "job-7",
        "timeout_seconds": 1.0,
    }


def test_train_candidates_slurm_array_from_specs_submits_waits_and_collects(tmp_path):
    calls = []
    collected = []
    work_items = [SimpleNamespace(index=0), SimpleNamespace(index=1)]
    spec_paths = [tmp_path / "spec0.json", tmp_path / "spec1.json"]

    def submit_candidate_array_fn(**kwargs):
        calls.append(("submit", kwargs))
        return "job-9"

    def wait_for_candidate_array_fn(**kwargs):
        calls.append(("wait", kwargs))

    def collect_metrics_fn(**kwargs):
        collected.append(kwargs)
        return ["metric"]

    result = train_candidates_slurm_array_from_specs(
        args=_args(candidate_array_max_parallel=0),
        round_dir=tmp_path / "attempt_0001",
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.2},
        init_final_accuracy=0.1,
        collect_metrics_fn=collect_metrics_fn,
        submit_candidate_array_fn=submit_candidate_array_fn,
        wait_for_candidate_array_fn=wait_for_candidate_array_fn,
    )

    assert result == ["metric"]
    assert calls[0] == (
        "submit",
        {"args": _args(candidate_array_max_parallel=0), "round_dir": tmp_path / "attempt_0001", "spec_paths": spec_paths},
    )
    assert calls[1] == (
        "wait",
        {
            "args": _args(candidate_array_max_parallel=0),
            "round_dir": tmp_path / "attempt_0001",
            "work_items": work_items,
            "job_id": "job-9",
        },
    )
    assert collected == [
        {
            "round_dir": tmp_path / "attempt_0001",
            "work_items": work_items,
            "current_final_accuracy": 0.4,
            "current_per_size_accuracy": {5: 0.2},
            "init_final_accuracy": 0.1,
        }
    ]

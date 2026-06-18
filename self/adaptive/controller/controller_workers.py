"""Generic controller-worker spec and Slurm dispatch helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from self.core import worker_io
from self.adaptive.controller.controller_phases import PHASE_PROPOSAL_GRPO, PHASE_ROUND_MODEL, PHASE_SEED
from self.core.data_io import ensure_dir
from self.core.slurm import submit_sbatch, wait_for_files_or_job_exit


JsonDict = Dict[str, Any]
RunControllerSpecFn = Callable[[Path], JsonDict]


def controller_worker_failure_path(worker_dir: Path) -> Path:
    return worker_io.controller_worker_failure_path(worker_dir)


def controller_worker_output_path(worker_dir: Path) -> Path:
    return worker_io.controller_worker_output_path(worker_dir)


def controller_worker_time_limit_for_phase(*, args: argparse.Namespace, phase: str) -> str:
    if phase == PHASE_SEED:
        return str(getattr(args, "controller_seed_worker_time_limit", None) or args.controller_worker_time_limit)
    if phase == PHASE_ROUND_MODEL:
        return str(getattr(args, "controller_round_worker_time_limit", None) or args.controller_worker_time_limit)
    if phase == PHASE_PROPOSAL_GRPO:
        return str(getattr(args, "controller_grpo_worker_time_limit", None) or args.controller_worker_time_limit)
    return str(args.controller_worker_time_limit)


def submit_controller_worker(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    spec_path: Path,
    phase: str,
) -> str:
    script = args.controller_worker_sbatch_script
    logs_dir = worker_dir / "logs"
    ensure_dir(logs_dir)
    time_limit = controller_worker_time_limit_for_phase(args=args, phase=phase)
    submission = submit_sbatch(
        script=script,
        job_name=f"adaptive-cand-controller-{phase}",
        output_path=logs_dir / "controller-%j.out",
        error_path=logs_dir / "controller-%j.err",
        time_limit=str(time_limit),
        exports={
            "CONTROLLER_WORKER_SPEC": spec_path,
            "PYTHON_BIN": sys.executable,
            "ROOT_DIR": Path.cwd(),
        },
    )
    worker_io.write_json(
        worker_dir / "slurm_dispatch.json",
        {
            "job_id": submission.job_id,
            "phase": phase,
            "command": list(submission.command),
            "stdout": submission.stdout,
            "stderr": submission.stderr,
        },
    )
    return submission.job_id


def wait_for_controller_worker(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    job_id: str,
    phase: str,
) -> None:
    wait_for_files_or_job_exit(
        job_id=job_id,
        done_paths=[
            controller_worker_output_path(worker_dir),
            controller_worker_failure_path(worker_dir),
        ],
        poll_seconds=float(args.controller_worker_poll_seconds),
        on_first_poll_message=f"[INFO] Controller GPU worker {job_id} ({phase}) is running/pending.",
    )


def run_controller_worker_slurm(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    phase: str,
    payload: JsonDict,
) -> JsonDict:
    ensure_dir(worker_dir)
    spec_path = worker_dir / "worker_spec.json"
    output_path = controller_worker_output_path(worker_dir)
    failure_path = controller_worker_failure_path(worker_dir)
    if output_path.exists():
        return worker_io.load_json(output_path)
    if failure_path.exists():
        raise RuntimeError(f"Controller worker {phase} has existing failure: {worker_io.load_json(failure_path)}")
    args_payload = worker_io.clear_worker_entry_flags(worker_io.json_ready_args(args))
    worker_io.write_json(
        spec_path,
        {
            "phase": phase,
            "args": args_payload,
            "worker_dir": str(worker_dir),
            **payload,
        },
    )
    job_id = submit_controller_worker(args=args, worker_dir=worker_dir, spec_path=spec_path, phase=phase)
    print(f"[INFO] Submitted controller GPU worker {job_id} for phase={phase}.", flush=True)
    wait_for_controller_worker(args=args, worker_dir=worker_dir, job_id=job_id, phase=phase)
    if failure_path.exists():
        raise RuntimeError(f"Controller worker {phase} failed: {worker_io.load_json(failure_path)}")
    if not output_path.exists():
        raise RuntimeError(f"Controller worker {phase} finished without worker_output.json")
    return worker_io.load_json(output_path)


def run_controller_worker(
    *,
    spec_path: Path,
    run_from_spec_fn: RunControllerSpecFn,
) -> JsonDict:
    payload: JsonDict | None = None
    try:
        payload = worker_io.load_json(spec_path)
        worker_dir = Path(payload["worker_dir"])
        output = run_from_spec_fn(spec_path)
        worker_io.write_json(controller_worker_output_path(worker_dir), output)
        return output
    except Exception as exc:
        if payload is not None:
            try:
                worker_dir = Path(payload["worker_dir"])
                worker_io.write_json(
                    controller_worker_failure_path(worker_dir),
                    {
                        "spec_path": str(spec_path),
                        "phase": payload.get("phase"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        raise

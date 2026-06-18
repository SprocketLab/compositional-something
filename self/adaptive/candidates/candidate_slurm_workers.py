"""Slurm-array candidate-worker dispatch."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from self.core import worker_io
from self.core.data_io import ensure_dir
from self.adaptive.candidates.candidate_local_workers import write_json
from self.core.slurm import cancel_job, slurm_job_active, submit_sbatch


CollectMetricsFn = Callable[..., List[Any]]


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


def train_candidates_slurm_array_from_specs(
    *,
    args: Any,
    round_dir: Path,
    work_items: Sequence[Any],
    spec_paths: Sequence[Path],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    collect_metrics_fn: CollectMetricsFn,
    submit_candidate_array_fn: Optional[Callable[..., str]] = None,
    wait_for_candidate_array_fn: Optional[Callable[..., None]] = None,
) -> List[Any]:
    submit_candidate_array_fn = submit_candidate_array_fn or submit_candidate_array
    wait_for_candidate_array_fn = wait_for_candidate_array_fn or wait_for_candidate_array
    job_id = submit_candidate_array_fn(args=args, round_dir=round_dir, spec_paths=spec_paths)
    print(
        f"[INFO] Submitted candidate worker array {job_id} with {len(spec_paths)} tasks "
        f"(max_parallel={args.candidate_array_max_parallel or 'unlimited'}).",
        flush=True,
    )
    wait_for_candidate_array_fn(args=args, round_dir=round_dir, work_items=work_items, job_id=job_id)
    return collect_metrics_fn(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def submit_candidate_array(
    *,
    args: Any,
    round_dir: Path,
    spec_paths: Sequence[Path],
    submit_sbatch_fn: Callable[..., Any] = submit_sbatch,
    executable: str | None = None,
    cwd_fn: Callable[[], Path] = Path.cwd,
) -> str:
    script = args.candidate_array_sbatch_script
    job_dir = round_dir / "candidate_jobs"
    logs_dir = job_dir / "logs"
    ensure_dir(logs_dir)
    throttle = f"%{args.candidate_array_max_parallel}" if args.candidate_array_max_parallel > 0 else ""
    array_spec = f"0-{len(spec_paths) - 1}{throttle}"
    submission = submit_sbatch_fn(
        script=script,
        job_name=f"adaptive-cand-{round_dir.name}",
        output_path=logs_dir / "candidate-%A_%a.out",
        error_path=logs_dir / "candidate-%A_%a.err",
        time_limit=str(args.candidate_array_time_limit),
        exports={
            "CANDIDATE_SPEC_DIR": round_dir / "candidate_jobs" / "specs",
            "PYTHON_BIN": executable or sys.executable,
            "ROOT_DIR": cwd_fn(),
        },
        array_spec=array_spec,
    )
    write_json(
        job_dir / "slurm_dispatch.json",
        {
            "job_id": submission.job_id,
            "array_spec": array_spec,
            "command": list(submission.command),
            "stdout": submission.stdout,
            "stderr": submission.stderr,
        },
    )
    return submission.job_id


def wait_for_candidate_array(
    *,
    args: Any,
    round_dir: Path,
    work_items: Sequence[Any],
    job_id: str,
    cancel_job_fn: Callable[[str], None] = cancel_job,
    slurm_job_active_fn: Callable[[str], bool] = slurm_job_active,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    start = monotonic_fn()
    last_reported_done = -1
    while True:
        done_count = 0
        for item in work_items:
            if candidate_metric_path(round_dir, item).exists() or candidate_worker_failure_path(round_dir, item).exists():
                done_count += 1
        if done_count == len(work_items):
            return
        if done_count != last_reported_done:
            print(
                f"[INFO] Candidate worker array {job_id}: {done_count}/{len(work_items)} workers finished.",
                flush=True,
            )
            last_reported_done = done_count
        if args.candidate_array_timeout_seconds > 0.0:
            elapsed = monotonic_fn() - start
            if elapsed >= args.candidate_array_timeout_seconds:
                cancel_job_fn(job_id)
                write_json(
                    round_dir / "candidate_jobs" / "slurm_timeout.json",
                    {
                        "job_id": job_id,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": args.candidate_array_timeout_seconds,
                    },
                )
                return
        if not slurm_job_active_fn(job_id):
            return
        sleep_fn(float(args.candidate_array_poll_seconds))

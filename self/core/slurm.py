#!/usr/bin/env python3
"""Small SLURM helpers for adaptive self-improvement controllers."""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class SlurmSubmission:
    job_id: str
    command: Sequence[str]
    stdout: str
    stderr: str


def require_sbatch(context: str) -> None:
    if shutil.which("sbatch") is None:
        raise RuntimeError(f"{context} requires sbatch in PATH.")


def slurm_job_active(job_id: str) -> bool:
    result = subprocess.run(
        ["squeue", "-h", "-j", job_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def submit_sbatch(
    *,
    script: Path,
    job_name: str,
    output_path: Path,
    error_path: Path,
    time_limit: str,
    exports: Mapping[str, object],
    array_spec: Optional[str] = None,
) -> SlurmSubmission:
    require_sbatch(str(script))
    if not script.exists():
        raise FileNotFoundError(f"SBATCH script not found: {script}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    export_value = "ALL"
    if exports:
        export_value += "," + ",".join(f"{key}={value}" for key, value in exports.items())
    cmd = [
        "sbatch",
        "--parsable",
    ]
    if array_spec is not None:
        cmd.extend(["--array", array_spec])
    cmd.extend(
        [
            "--job-name",
            job_name,
            "--output",
            str(output_path),
            "--error",
            str(error_path),
            "--time",
            str(time_limit),
            "--export",
            export_value,
            str(script),
        ]
    )
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return SlurmSubmission(
        job_id=result.stdout.strip().split(";", 1)[0],
        command=cmd,
        stdout=result.stdout.strip(),
        stderr=result.stderr.strip(),
    )


def wait_for_files_or_job_exit(
    *,
    job_id: str,
    done_paths: Iterable[Path],
    poll_seconds: float,
    on_first_poll_message: Optional[str] = None,
) -> None:
    reported = False
    while True:
        if any(path.exists() for path in done_paths):
            return
        if on_first_poll_message and not reported:
            print(on_first_poll_message, flush=True)
            reported = True
        if not slurm_job_active(job_id):
            return
        time.sleep(float(poll_seconds))


def cancel_job(job_id: str) -> None:
    subprocess.run(["scancel", job_id], check=False)

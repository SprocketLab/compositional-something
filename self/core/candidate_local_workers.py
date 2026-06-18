"""Local candidate-worker process scheduling."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core import worker_io
from self.core.data_io import ensure_dir, sanitize_json_value


JsonDict = Dict[str, Any]
CollectMetricsFn = Callable[..., List[Any]]
PreparePackSpecsFn = Callable[..., List[Tuple[int, List[Any], Path]]]


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


def train_candidates_local_parallel_from_specs(
    *,
    args: Any,
    round_dir: Path,
    work_items: Sequence[Any],
    spec_paths: Sequence[Path],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    collect_metrics_fn: CollectMetricsFn,
    prepare_pack_specs_fn: PreparePackSpecsFn,
    subprocess_module: Any = subprocess,
    executable: str | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[Any]:
    job_dir = round_dir / "candidate_jobs"
    logs_dir = job_dir / "logs"
    ensure_dir(logs_dir)
    max_parallel = max(1, int(args.candidate_local_parallelism))
    pack_size = max(1, int(getattr(args, "candidate_local_pack_size", 1)))
    if pack_size == 1:
        pending = [
            {
                "label": f"candidate-{item.index:02d}",
                "items": [item],
                "spec_path": spec_path,
                "is_pack": False,
            }
            for item, spec_path in zip(work_items, spec_paths)
        ]
    else:
        pending = [
            {
                "label": f"pack-{pack_index:02d}",
                "items": chunk_items,
                "spec_path": pack_path,
                "is_pack": True,
            }
            for pack_index, chunk_items, pack_path in prepare_pack_specs_fn(
                round_dir=round_dir,
                work_items=work_items,
                spec_paths=spec_paths,
                pack_size=pack_size,
            )
        ]
    dispatch_plan = local_candidate_dispatch_plan(
        args=args,
        candidate_count=len(spec_paths),
        process_units=pending,
        max_parallel=max_parallel,
        pack_size=pack_size,
    )
    active: List[Tuple[JsonDict, subprocess.Popen[Any], Any, Any]] = []
    launched: List[JsonDict] = []
    start = monotonic_fn()
    executable = executable or sys.executable
    print(
        f"[INFO] Running {len(spec_paths)} local candidate workers as {len(pending)} process(es) "
        f"(max_parallel={max_parallel}, pack_size={pack_size}).",
        flush=True,
    )
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                unit = pending.pop(0)
                stdout_path = logs_dir / f"candidate-local-{unit['label']}.out"
                stderr_path = logs_dir / f"candidate-local-{unit['label']}.err"
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                stderr_handle = stderr_path.open("w", encoding="utf-8")
                if unit["is_pack"]:
                    command = [
                        executable,
                        "-m",
                        "self.core.driver",
                        "--run-candidate-pack-worker",
                        "--candidate-worker-pack-spec",
                        str(unit["spec_path"]),
                    ]
                else:
                    command = [
                        executable,
                        "-m",
                        "self.core.driver",
                        "--run-candidate-worker",
                        "--candidate-worker-spec",
                        str(unit["spec_path"]),
                    ]
                process = subprocess_module.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
                active.append((unit, process, stdout_handle, stderr_handle))
                launched.append(
                    {
                        "label": unit["label"],
                        "candidate_indices": [item.index for item in unit["items"]],
                        "spec_path": str(unit["spec_path"]),
                        "is_pack": bool(unit["is_pack"]),
                        "pid": process.pid,
                        "command": command,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    }
                )
            write_json(
                job_dir / "local_dispatch.json",
                {
                    **dispatch_plan,
                    "max_parallel": max_parallel,
                    "pack_size": pack_size,
                    "launched": launched,
                    "active_pids": [process.pid for _, process, _, _ in active],
                    "pending": len(pending),
                },
            )
            next_active: List[Tuple[JsonDict, subprocess.Popen[Any], Any, Any]] = []
            for unit, process, stdout_handle, stderr_handle in active:
                returncode = process.poll()
                if returncode is None:
                    next_active.append((unit, process, stdout_handle, stderr_handle))
                    continue
                stdout_handle.close()
                stderr_handle.close()
                if returncode != 0:
                    for item in unit["items"]:
                        write_local_candidate_failure(
                            round_dir=round_dir,
                            item=item,
                            spec_path=Path(unit["spec_path"]),
                            returncode=returncode,
                            reason=f"candidate local worker {unit['label']} exited with code {returncode}",
                        )
            active = next_active
            done_count = sum(
                1
                for item in work_items
                if candidate_metric_path(round_dir, item).exists()
                or candidate_worker_failure_path(round_dir, item).exists()
            )
            print(
                f"[INFO] Local candidate workers: {done_count}/{len(work_items)} finished.",
                flush=True,
            )
            if args.candidate_array_timeout_seconds > 0.0:
                elapsed = monotonic_fn() - start
                if elapsed >= args.candidate_array_timeout_seconds:
                    for unit, process, stdout_handle, stderr_handle in active:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        stdout_handle.close()
                        stderr_handle.close()
                        for item in unit["items"]:
                            write_local_candidate_failure(
                                round_dir=round_dir,
                                item=item,
                                spec_path=Path(unit["spec_path"]),
                                returncode=process.returncode,
                                reason=(
                                    f"candidate local worker {unit['label']} timed out after "
                                    f"{args.candidate_array_timeout_seconds} seconds"
                                ),
                            )
                    write_json(
                        job_dir / "local_timeout.json",
                        {
                            "elapsed_seconds": elapsed,
                            "timeout_seconds": args.candidate_array_timeout_seconds,
                            "max_parallel": max_parallel,
                            "pack_size": pack_size,
                        },
                    )
                    active = []
                    pending = []
                    break
            if pending or active:
                sleep_fn(float(args.candidate_array_poll_seconds))
    finally:
        for unit, process, stdout_handle, stderr_handle in active:
            if process.poll() is None:
                process.terminate()
            stdout_handle.close()
            stderr_handle.close()
    return collect_metrics_fn(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def local_candidate_dispatch_plan(
    *,
    args: Any,
    candidate_count: int,
    process_units: Sequence[JsonDict],
    max_parallel: int,
    pack_size: int,
) -> JsonDict:
    packed_workers = any(bool(unit.get("is_pack")) for unit in process_units)
    candidate_local_cache_base_state = bool(getattr(args, "candidate_local_cache_base_state", False))
    return {
        "candidate_count": int(candidate_count),
        "planned_processes": len(process_units),
        "max_parallel": int(max_parallel),
        "pack_size": int(pack_size),
        "packed_workers": packed_workers,
        "cache_plan": {
            "shared_input_cache": packed_workers,
            "tokenizer_bootstrap_cache": packed_workers or candidate_local_cache_base_state,
            "base_state_cache": candidate_local_cache_base_state,
        },
        "planned_units": [
            {
                "label": str(unit["label"]),
                "candidate_indices": [item.index for item in unit["items"]],
                "spec_path": str(unit["spec_path"]),
                "is_pack": bool(unit["is_pack"]),
            }
            for unit in process_units
        ],
    }


def write_local_candidate_failure(
    *,
    round_dir: Path,
    item: Any,
    spec_path: Path,
    returncode: Optional[int],
    reason: str,
) -> None:
    failure_path = candidate_worker_failure_path(round_dir, item)
    if failure_path.exists() or candidate_metric_path(round_dir, item).exists():
        return
    write_json(
        failure_path,
        {
            "spec_path": str(spec_path),
            "error_type": "LocalCandidateWorkerError",
            "error": reason,
            "returncode": returncode,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")

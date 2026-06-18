#!/usr/bin/env python3
"""Candidate worker spec preparation and dispatch helpers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core import worker_io
from self.core.data_io import ensure_dir, sanitize_json_value, save_examples
from self.core.proposals import PromptBundle, write_trace_jsonl
from self.core.slurm import cancel_job, slurm_job_active, submit_sbatch


JsonDict = Dict[str, Any]
CollectMetricsFn = Callable[..., List[Any]]


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


def prepare_candidate_worker_specs(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
) -> List[Path]:
    job_dir = round_dir / "candidate_jobs"
    input_dir = job_dir / "inputs"
    spec_dir = job_dir / "specs"
    ensure_dir(input_dir)
    ensure_dir(spec_dir)
    source_examples_path = input_dir / "source_examples.jsonl"
    eval_examples_path = input_dir / "eval_examples.jsonl"
    proposal_trace_path = input_dir / "proposal_trace_buffer.jsonl"
    outcome_trace_path = input_dir / "outcome_trace_buffer.jsonl"
    prompt_path = input_dir / "proposal_prompt.json"
    save_examples(source_examples_path, source_examples, task.serialize_example)
    save_examples(eval_examples_path, eval_examples, task.serialize_example)
    write_trace_jsonl(proposal_trace_path, [trace.to_json_dict() for trace in proposal_trace_buffer])
    write_trace_jsonl(outcome_trace_path, [trace.to_json_dict() for trace in outcome_trace_buffer])
    write_json(prompt_path, {"system": proposal_prompt.system, "user": proposal_prompt.user})

    spec_paths: List[Path] = []
    manifest: List[JsonDict] = []
    args_payload = worker_io.clear_worker_entry_flags(worker_io.json_ready_args(args))
    for array_index, item in enumerate(work_items):
        candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
        pseudo_examples_path = candidate_dir / "pseudo_examples.jsonl"
        spec_path = spec_dir / f"candidate_{array_index}.json"
        payload: JsonDict = {
            "args": args_payload,
            "array_index": array_index,
            "candidate_index": item.index,
            "round_index": round_index,
            "attempt_index": attempt_index,
            "current_checkpoint": current_checkpoint,
            "round_dir": str(round_dir),
            "source_examples_path": str(source_examples_path),
            "eval_examples_path": str(eval_examples_path),
            "proposal_trace_buffer_path": str(proposal_trace_path),
            "outcome_trace_buffer_path": str(outcome_trace_path),
            "proposal_prompt_path": str(prompt_path),
            "pseudo_examples_path": str(pseudo_examples_path),
            "current_final_accuracy": current_final_accuracy,
            "current_per_size_accuracy": {str(size): score for size, score in current_per_size_accuracy.items()},
            "init_final_accuracy": init_final_accuracy,
            "seed": args.seed + attempt_index * 1009 + item.index,
            "candidate": {
                "index": item.index,
                "row_id": item.row_id,
                "proposal": item.proposal.to_json_dict(),
                "completion": item.completion,
                "raw_output": item.raw_output,
                "proposal_prediction": item.proposal_prediction,
                "pseudo_diagnostics": item.pseudo_diagnostics,
            },
        }
        write_json(spec_path, payload)
        spec_paths.append(spec_path)
        manifest.append(
            {
                "array_index": array_index,
                "candidate_index": item.index,
                "spec_path": str(spec_path),
                "metrics_path": str(candidate_metric_path(round_dir, item)),
                "worker_failure_path": str(candidate_worker_failure_path(round_dir, item)),
            }
        )
    write_json(job_dir / "manifest.json", manifest)
    return spec_paths


def prepare_candidate_worker_pack_specs(
    *,
    round_dir: Path,
    work_items: Sequence[Any],
    spec_paths: Sequence[Path],
    pack_size: int,
) -> List[Tuple[int, List[Any], Path]]:
    if pack_size < 1:
        raise ValueError("pack_size must be positive.")
    pack_dir = round_dir / "candidate_jobs" / "pack_specs"
    ensure_dir(pack_dir)
    packs: List[Tuple[int, List[Any], Path]] = []
    manifest: List[JsonDict] = []
    pairs = list(zip(work_items, spec_paths))
    for pack_index, start in enumerate(range(0, len(pairs), pack_size)):
        chunk = pairs[start : start + pack_size]
        chunk_items = [item for item, _ in chunk]
        chunk_spec_paths = [spec_path for _, spec_path in chunk]
        pack_path = pack_dir / f"pack_{pack_index}.json"
        payload = {
            "pack_index": pack_index,
            "spec_paths": [str(spec_path) for spec_path in chunk_spec_paths],
            "candidates": [
                {
                    "candidate_index": item.index,
                    "spec_path": str(spec_path),
                    "metrics_path": str(candidate_metric_path(round_dir, item)),
                    "worker_failure_path": str(candidate_worker_failure_path(round_dir, item)),
                }
                for item, spec_path in chunk
            ],
        }
        write_json(pack_path, payload)
        packs.append((pack_index, chunk_items, pack_path))
        manifest.append(
            {
                "pack_index": pack_index,
                "pack_spec_path": str(pack_path),
                "candidate_indices": [item.index for item in chunk_items],
                "spec_paths": [str(spec_path) for spec_path in chunk_spec_paths],
            }
        )
    write_json(round_dir / "candidate_jobs" / "pack_manifest.json", manifest)
    return packs


def train_candidates_slurm_array(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    collect_metrics_fn: CollectMetricsFn,
) -> List[Any]:
    if not work_items:
        return []
    spec_paths = prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
    )
    job_id = submit_candidate_array(args=args, round_dir=round_dir, spec_paths=spec_paths)
    print(
        f"[INFO] Submitted candidate worker array {job_id} with {len(spec_paths)} tasks "
        f"(max_parallel={args.candidate_array_max_parallel or 'unlimited'}).",
        flush=True,
    )
    wait_for_candidate_array(args=args, round_dir=round_dir, work_items=work_items, job_id=job_id)
    return collect_metrics_fn(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def train_candidates_local_parallel(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    collect_metrics_fn: CollectMetricsFn,
) -> List[Any]:
    if not work_items:
        return []
    spec_paths = prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
    )
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
            for pack_index, chunk_items, pack_path in prepare_candidate_worker_pack_specs(
                round_dir=round_dir,
                work_items=work_items,
                spec_paths=spec_paths,
                pack_size=pack_size,
            )
        ]
    active: List[Tuple[JsonDict, subprocess.Popen[Any], Any, Any]] = []
    launched: List[JsonDict] = []
    start = time.monotonic()
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
                        sys.executable,
                        "-m",
                        "self.core.driver",
                        "--run-candidate-pack-worker",
                        "--candidate-worker-pack-spec",
                        str(unit["spec_path"]),
                    ]
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "self.core.driver",
                        "--run-candidate-worker",
                        "--candidate-worker-spec",
                        str(unit["spec_path"]),
                    ]
                process = subprocess.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
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
                elapsed = time.monotonic() - start
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
                time.sleep(float(args.candidate_array_poll_seconds))
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


def submit_candidate_array(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    spec_paths: Sequence[Path],
) -> str:
    script = args.candidate_array_sbatch_script
    job_dir = round_dir / "candidate_jobs"
    logs_dir = job_dir / "logs"
    ensure_dir(logs_dir)
    throttle = f"%{args.candidate_array_max_parallel}" if args.candidate_array_max_parallel > 0 else ""
    array_spec = f"0-{len(spec_paths) - 1}{throttle}"
    submission = submit_sbatch(
        script=script,
        job_name=f"adaptive-cand-{round_dir.name}",
        output_path=logs_dir / "candidate-%A_%a.out",
        error_path=logs_dir / "candidate-%A_%a.err",
        time_limit=str(args.candidate_array_time_limit),
        exports={
            "CANDIDATE_SPEC_DIR": round_dir / "candidate_jobs" / "specs",
            "PYTHON_BIN": sys.executable,
            "ROOT_DIR": Path.cwd(),
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
    args: argparse.Namespace,
    round_dir: Path,
    work_items: Sequence[Any],
    job_id: str,
) -> None:
    start = time.monotonic()
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
            elapsed = time.monotonic() - start
            if elapsed >= args.candidate_array_timeout_seconds:
                cancel_job(job_id)
                write_json(
                    round_dir / "candidate_jobs" / "slurm_timeout.json",
                    {
                        "job_id": job_id,
                        "elapsed_seconds": elapsed,
                        "timeout_seconds": args.candidate_array_timeout_seconds,
                    },
                )
                return
        if not slurm_job_active(job_id):
            return
        time.sleep(float(args.candidate_array_poll_seconds))


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

#!/usr/bin/env python3
"""Candidate worker payloads, specs, and local/Slurm execution."""

from __future__ import annotations

# --- from candidate_worker_payloads.py ---
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.core import worker_io
from self.core.data_io import load_examples, sanitize_json_value
from self.core.models import CandidateWorkItem, ExactPairDataset, proposal_from_payload


JsonDict = Dict[str, Any]


def candidate_payload_from_work_item(item: CandidateWorkItem) -> JsonDict:
    """Return the candidate block embedded in worker specs."""
    return sanitize_json_value(
        {
            "index": item.index,
            "row_id": item.row_id,
            "proposal": item.proposal.to_json_dict(),
            "completion": item.completion,
            "raw_output": item.raw_output,
            "proposal_prediction": item.proposal_prediction,
            "pseudo_diagnostics": item.pseudo_diagnostics,
        }
    )


def candidate_payload_to_work_item(
    *,
    payload: Mapping[str, Any],
    pseudo_examples: Sequence[Any],
    composed_keys: Sequence[Any] | set[Any] | None = None,
) -> CandidateWorkItem:
    """Rebuild a candidate work item from a serialized candidate block."""
    return CandidateWorkItem(
        index=int(payload["index"]),
        row_id=payload.get("row_id"),
        proposal=proposal_from_payload(dict(payload["proposal"])),
        completion=str(payload.get("completion", "")),
        raw_output=payload.get("raw_output"),
        composed=ExactPairDataset(
            examples=[],
            component_map={},
            keys=set(composed_keys or ()),
            diagnostics={},
        ),
        pseudo_examples=list(pseudo_examples),
        pseudo_diagnostics=dict(payload.get("pseudo_diagnostics") or {}),
        proposal_prediction=dict(payload.get("proposal_prediction") or {}),
    )


def work_item_to_worker_payload(
    *,
    item: CandidateWorkItem,
    round_dir: Path,
) -> JsonDict:
    candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
    payload = candidate_payload_from_work_item(item)
    payload.update(
        {
            "pseudo_examples_path": str(candidate_dir / "pseudo_examples.jsonl"),
            "pseudo_count": len(item.pseudo_examples),
            "composed_keys": [
                worker_io.json_ready_key(key)
                for key in sorted(item.composed.keys, key=repr)
            ],
            "composed_count": len(item.composed.examples),
        }
    )
    return sanitize_json_value(payload)


def work_item_from_worker_payload(
    *,
    payload: Mapping[str, Any],
    task: Any,
) -> CandidateWorkItem:
    pseudo_path = Path(str(payload["pseudo_examples_path"]))
    pseudo_examples = load_examples(pseudo_path, task.deserialize_example)
    composed_keys = {worker_io.key_from_json(key) for key in payload.get("composed_keys", [])}
    return candidate_payload_to_work_item(
        payload=payload,
        pseudo_examples=pseudo_examples,
        composed_keys=composed_keys,
    )


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


# --- from candidate_worker_inputs.py ---
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

import torch

from self.core.data_io import load_examples
from self.adaptive.traces.traces import outcome_trace_from_json, proposal_trace_from_json
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposals.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CandidateWorkerRuntimeDeps:
    load_json: Callable[[Path], Any]
    namespace_from_json_args: Callable[[Any], argparse.Namespace]
    normalize_args: Callable[[argparse.Namespace], argparse.Namespace]
    task_for_name: Callable[[str], Any]
    make_config: Callable[[argparse.Namespace], Any]
    load_trace_jsonl: Callable[[Path, Any], list[Any]]
    train_and_score_candidate: Callable[..., CandidateMetrics]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class CandidateWorkerSharedInputs:
    args: argparse.Namespace
    task: Any
    config: Any
    source_examples: Sequence[Any]
    eval_examples: Sequence[Any]
    proposal_trace_buffer: Sequence[Any]
    outcome_trace_buffer: Sequence[Any]
    proposal_prompt: PromptBundle
    model_bootstrap_cache: Optional[ModelBootstrapCache]


SharedInputCache = MutableMapping[str, CandidateWorkerSharedInputs]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _shared_input_cache_key(payload: JsonDict) -> str:
    return _stable_json(
        {
            "args": payload.get("args"),
            "source_examples_path": payload.get("source_examples_path"),
            "eval_examples_path": payload.get("eval_examples_path"),
            "proposal_trace_buffer_path": payload.get("proposal_trace_buffer_path"),
            "outcome_trace_buffer_path": payload.get("outcome_trace_buffer_path"),
            "proposal_prompt_path": payload.get("proposal_prompt_path"),
        }
    )


def load_candidate_worker_shared_inputs(
    payload: JsonDict,
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    shared_cache: Optional[SharedInputCache],
) -> CandidateWorkerSharedInputs:
    cache_key = _shared_input_cache_key(payload)
    if shared_cache is not None and cache_key in shared_cache:
        return shared_cache[cache_key]

    args = deps.namespace_from_json_args(payload["args"])
    args.run_candidate_worker = True
    args.candidate_worker_spec = spec_path
    args = deps.normalize_args(args)
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] Worker defaulting to bf16 on CUDA.", flush=True)
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    config = deps.make_config(args)
    source_examples = load_examples(Path(payload["source_examples_path"]), task.deserialize_example)
    eval_examples = load_examples(Path(payload["eval_examples_path"]), task.deserialize_example)
    proposal_trace_buffer = deps.load_trace_jsonl(
        Path(payload["proposal_trace_buffer_path"]),
        proposal_trace_from_json,
    )
    outcome_trace_buffer = deps.load_trace_jsonl(
        Path(payload["outcome_trace_buffer_path"]),
        outcome_trace_from_json,
    )
    prompt_payload = deps.load_json(Path(payload["proposal_prompt_path"]))
    shared = CandidateWorkerSharedInputs(
        args=args,
        task=task,
        config=config,
        source_examples=source_examples,
        eval_examples=eval_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=PromptBundle(
            system=str(prompt_payload.get("system", "")),
            user=str(prompt_payload.get("user", "")),
        ),
        model_bootstrap_cache=_make_model_bootstrap_cache(args, shared_cache=shared_cache),
    )
    if shared_cache is not None:
        shared_cache[cache_key] = shared
    return shared


def _make_model_bootstrap_cache(
    args: argparse.Namespace,
    *,
    shared_cache: Optional[SharedInputCache],
) -> Optional[ModelBootstrapCache]:
    cache_base_state = bool(getattr(args, "candidate_local_cache_base_state", False))
    if shared_cache is not None or cache_base_state:
        return ModelBootstrapCache(cache_base_state=cache_base_state)
    return None


def candidate_item_from_payload(payload: JsonDict, pseudo_examples: Sequence[Any]) -> CandidateWorkItem:
    candidate_payload = dict(payload["candidate"])
    return candidate_payload_to_work_item(
        payload=candidate_payload,
        pseudo_examples=pseudo_examples,
    )


# --- from workers.py ---
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from self.core.models import CandidateMetrics


JsonDict = Dict[str, Any]


def run_candidate_worker_pack_from_spec(
    pack_spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    run_from_spec_fn: Callable[..., CandidateMetrics],
) -> JsonDict:
    payload = deps.load_json(pack_spec_path)
    spec_paths: Sequence[Path] = [Path(str(path)) for path in payload.get("spec_paths", [])]
    results: list[JsonDict] = []
    failed = 0
    shared_cache: SharedInputCache = {}
    runner_accepts_cache = _run_from_spec_accepts_shared_cache(run_from_spec_fn)
    for spec_path in spec_paths:
        try:
            if runner_accepts_cache:
                metrics = run_from_spec_fn(spec_path, shared_cache=shared_cache)
            else:
                metrics = run_from_spec_fn(spec_path)
            results.append(
                {
                    "spec_path": str(spec_path),
                    "status": "ok",
                    "candidate_index": metrics.index,
                }
            )
        except Exception as exc:
            failed += 1
            try:
                failure_payload = write_candidate_worker_failure_from_spec(
                    spec_path,
                    exc,
                    load_json_fn=deps.load_json,
                    write_json_fn=deps.write_json,
                )
            except Exception:
                failure_payload = {
                    "spec_path": str(spec_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            results.append(
                {
                    "spec_path": str(spec_path),
                    "status": "failed",
                    "failure": failure_payload,
                }
            )
            print(f"[ERROR] Packed candidate worker failed for {spec_path}: {exc}", flush=True)
    return {
        "pack_spec_path": str(pack_spec_path),
        "total": len(spec_paths),
        "succeeded": len(spec_paths) - failed,
        "failed": failed,
        "results": results,
        "shared_input_cache_entries": len(shared_cache),
        "model_bootstrap_cache": [
            shared.model_bootstrap_cache.stats()
            for shared in shared_cache.values()
            if shared.model_bootstrap_cache is not None
        ],
        "model_bootstrap_cache_details": [
            shared.model_bootstrap_cache.detailed_stats()
            for shared in shared_cache.values()
            if shared.model_bootstrap_cache is not None
        ],
    }


def _run_from_spec_accepts_shared_cache(run_from_spec_fn: Callable[..., CandidateMetrics]) -> bool:
    try:
        signature = inspect.signature(run_from_spec_fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "shared_cache":
            return True
    return False


# --- from workers.py ---
import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import set_seed

from self.core.data_io import load_examples
from self.core.models import CandidateMetrics, float_or_nan


JsonDict = Dict[str, Any]


def run_candidate_worker_from_spec(
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    shared_cache: Optional[SharedInputCache] = None,
) -> CandidateMetrics:
    payload = deps.load_json(spec_path)
    shared = load_candidate_worker_shared_inputs(
        payload,
        spec_path,
        deps=deps,
        shared_cache=shared_cache,
    )
    args = copy.copy(shared.args)
    args.candidate_worker_spec = spec_path
    seed = int(payload["seed"])
    set_seed(seed)
    pseudo_examples = load_examples(Path(payload["pseudo_examples_path"]), shared.task.deserialize_example)
    item = candidate_item_from_payload(payload, pseudo_examples)
    current_per_size_accuracy = {
        int(size): float(score)
        for size, score in dict(payload.get("current_per_size_accuracy", {})).items()
        if score is not None
    }
    return deps.train_and_score_candidate(
        args=args,
        task=shared.task,
        current_checkpoint=str(payload["current_checkpoint"]),
        source_examples=shared.source_examples,
        proposal_trace_buffer=shared.proposal_trace_buffer,
        outcome_trace_buffer=shared.outcome_trace_buffer,
        proposal_prompt=shared.proposal_prompt,
        round_index=int(payload["round_index"]),
        item=item,
        round_dir=Path(payload["round_dir"]),
        eval_examples=shared.eval_examples,
        current_final_accuracy=float_or_nan(payload.get("current_final_accuracy")),
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=float_or_nan(payload.get("init_final_accuracy")),
        config=shared.config,
        seed=seed,
        model_bootstrap_cache=shared.model_bootstrap_cache,
    )


def run_candidate_worker(
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    run_from_spec_fn: Callable[[Path], CandidateMetrics],
) -> JsonDict:
    try:
        metrics = run_from_spec_fn(spec_path)
        return metrics.to_json_dict()
    except Exception as exc:
        try:
            write_candidate_worker_failure_from_spec(
                spec_path,
                exc,
                load_json_fn=deps.load_json,
                write_json_fn=deps.write_json,
            )
        except Exception:
            print(f"[ERROR] Candidate worker failed before failure artifact could be written: {exc}", flush=True)
        raise


# --- from candidate_local_workers.py ---
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
                        "self.adaptive.run.driver",
                        "--run-candidate-pack-worker",
                        "--candidate-worker-pack-spec",
                        str(unit["spec_path"]),
                    ]
                else:
                    command = [
                        executable,
                        "-m",
                        "self.adaptive.run.driver",
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


# --- from candidate_slurm_workers.py ---
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from self.core import worker_io
from self.core.data_io import ensure_dir
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
    submit_candidate_array_fn = submit_candidate_array_fn or _submit_candidate_array_impl
    wait_for_candidate_array_fn = wait_for_candidate_array_fn or _wait_for_candidate_array_impl
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


def _submit_candidate_array_impl(
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


def _wait_for_candidate_array_impl(
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


# --- from workers.py ---
import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from self.core import worker_io
from self.core.data_io import ensure_dir, save_examples
from self.adaptive.proposals.proposal_io import write_trace_jsonl
from self.adaptive.proposals.proposal_prompts import PromptBundle


JsonDict = Dict[str, Any]


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
            "candidate": candidate_payload_from_work_item(item),
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


# --- from workers.py ---
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.adaptive.proposals.proposal_prompts import PromptBundle
from self.core.slurm import cancel_job, slurm_job_active, submit_sbatch


CollectMetricsFn = Callable[..., List[Any]]


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
    return train_candidates_slurm_array_from_specs(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        collect_metrics_fn=collect_metrics_fn,
        submit_candidate_array_fn=submit_candidate_array,
        wait_for_candidate_array_fn=wait_for_candidate_array,
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
    return train_candidates_local_parallel_from_specs(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        collect_metrics_fn=collect_metrics_fn,
        prepare_pack_specs_fn=prepare_candidate_worker_pack_specs,
        subprocess_module=subprocess,
    )


def submit_candidate_array(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    spec_paths: Sequence[Path],
) -> str:
    return _submit_candidate_array_impl(
        args=args,
        round_dir=round_dir,
        spec_paths=spec_paths,
        submit_sbatch_fn=submit_sbatch,
        executable=sys.executable,
        cwd_fn=Path.cwd,
    )


def wait_for_candidate_array(
    *,
    args: argparse.Namespace,
    round_dir: Path,
    work_items: Sequence[Any],
    job_id: str,
    cancel_job_fn: Callable[[str], None] | None = None,
    slurm_job_active_fn: Callable[[str], bool] | None = None,
    monotonic_fn: Callable[[], float] | None = None,
    sleep_fn: Callable[[float], None] | None = None,
) -> None:
    return _wait_for_candidate_array_impl(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        job_id=job_id,
        cancel_job_fn=cancel_job_fn or cancel_job,
        slurm_job_active_fn=slurm_job_active_fn or slurm_job_active,
        monotonic_fn=monotonic_fn or time.monotonic,
        sleep_fn=sleep_fn or time.sleep,
    )

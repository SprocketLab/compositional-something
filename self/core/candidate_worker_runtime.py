"""Candidate-worker entry point runtime."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

import torch
from transformers import set_seed

from self.core.candidate_worker_payloads import candidate_payload_to_work_item
from self.core.data_io import load_examples
from self.core.experience_trace_models import outcome_trace_from_json, proposal_trace_from_json
from self.core.model_io import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem, float_or_nan
from self.core.proposals import PromptBundle


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


def _load_shared_inputs(
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


def _candidate_item_from_payload(payload: JsonDict, pseudo_examples: Sequence[Any]) -> CandidateWorkItem:
    candidate_payload = dict(payload["candidate"])
    return candidate_payload_to_work_item(
        payload=candidate_payload,
        pseudo_examples=pseudo_examples,
    )


def run_candidate_worker_from_spec(
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    shared_cache: Optional[SharedInputCache] = None,
) -> CandidateMetrics:
    payload = deps.load_json(spec_path)
    shared = _load_shared_inputs(payload, spec_path, deps=deps, shared_cache=shared_cache)
    args = copy.copy(shared.args)
    args.candidate_worker_spec = spec_path
    seed = int(payload["seed"])
    set_seed(seed)
    pseudo_examples = load_examples(Path(payload["pseudo_examples_path"]), shared.task.deserialize_example)
    item = _candidate_item_from_payload(payload, pseudo_examples)
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


def _write_candidate_failure_from_spec(
    spec_path: Path,
    exc: Exception,
    *,
    deps: CandidateWorkerRuntimeDeps,
) -> JsonDict:
    payload = deps.load_json(spec_path)
    candidate_index = int(payload["candidate_index"])
    round_dir = Path(payload["round_dir"])
    failure_path = round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "worker_failure.json"
    failure_payload: JsonDict = {
        "spec_path": str(spec_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    deps.write_json(failure_path, failure_payload)
    return failure_payload


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
        payload: Optional[JsonDict] = None
        try:
            payload = deps.load_json(spec_path)
            candidate_index = int(payload["candidate_index"])
            round_dir = Path(payload["round_dir"])
            failure_path = round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "worker_failure.json"
            deps.write_json(
                failure_path,
                {
                    "spec_path": str(spec_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception:
            payload = None
        if payload is None:
            print(f"[ERROR] Candidate worker failed before failure artifact could be written: {exc}", flush=True)
        raise


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
                failure_payload = _write_candidate_failure_from_spec(spec_path, exc, deps=deps)
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

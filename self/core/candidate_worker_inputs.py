"""Shared input loading and caching for candidate-worker processes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

import torch

from self.core.candidate_worker_payloads import candidate_payload_to_work_item
from self.core.data_io import load_examples
from self.core.experience_trace_models import outcome_trace_from_json, proposal_trace_from_json
from self.core.model_io import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposal_prompts import PromptBundle


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

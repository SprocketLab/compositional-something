"""Candidate-worker entry point runtime."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import set_seed

from self.core.candidate_worker_inputs import (
    CandidateWorkerRuntimeDeps,
    SharedInputCache,
    candidate_item_from_payload,
    load_candidate_worker_shared_inputs,
)
from self.core.candidate_worker_failures import write_candidate_worker_failure_from_spec
from self.core.candidate_worker_pack_runtime import run_candidate_worker_pack_from_spec
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

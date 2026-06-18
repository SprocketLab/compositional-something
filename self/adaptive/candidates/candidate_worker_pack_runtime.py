"""Packed candidate-worker execution with shared input/model caches."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from self.adaptive.candidates.candidate_worker_failures import write_candidate_worker_failure_from_spec
from self.adaptive.candidates.candidate_worker_inputs import CandidateWorkerRuntimeDeps, SharedInputCache
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

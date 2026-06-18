"""Compatibility-aware worker entrypoint wiring for the adaptive driver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

from self.core.candidate_worker_runtime import (
    CandidateWorkerRuntimeDeps,
    run_candidate_worker as _run_candidate_worker_impl,
    run_candidate_worker_from_spec as _run_candidate_worker_from_spec_impl,
)
from self.core.candidate_worker_pack_runtime import (
    run_candidate_worker_pack_from_spec as _run_candidate_worker_pack_from_spec_impl,
)
from self.core.controller_worker_runtime import (
    ControllerWorkerRuntimeDeps,
    run_controller_worker as _run_controller_worker_impl,
    run_controller_worker_from_spec as _run_controller_worker_from_spec_impl,
    run_proposal_grpo_controller_worker_from_spec as _run_proposal_grpo_controller_worker_from_spec_impl,
    run_round_model_controller_worker_from_spec as _run_round_model_controller_worker_from_spec_impl,
    run_seed_controller_worker_from_spec as _run_seed_controller_worker_from_spec_impl,
)
from self.core.models import CandidateMetrics


JsonDict = Dict[str, Any]


@dataclass(frozen=True, slots=True)
class WorkerEntrypointDeps:
    load_json: Any
    namespace_from_json_args: Any
    normalize_args: Any
    default_bf16_on_cuda: Any
    task_for_name: Any
    make_config: Any
    load_trace_jsonl: Any
    train_and_score_candidate: Any
    write_json: Any
    load_key_set: Any
    run_seed_phase: Any
    run_round_model_phase: Any
    apply_proposal_grpo_update: Any
    candidate_metrics_from_json: Any
    work_item_to_worker_payload: Any
    run_controller_worker_generic: Any


def candidate_worker_runtime_deps(deps: WorkerEntrypointDeps) -> CandidateWorkerRuntimeDeps:
    return CandidateWorkerRuntimeDeps(
        load_json=deps.load_json,
        namespace_from_json_args=deps.namespace_from_json_args,
        normalize_args=deps.normalize_args,
        task_for_name=deps.task_for_name,
        make_config=deps.make_config,
        load_trace_jsonl=deps.load_trace_jsonl,
        train_and_score_candidate=deps.train_and_score_candidate,
        write_json=deps.write_json,
    )


def controller_worker_runtime_deps(deps: WorkerEntrypointDeps) -> ControllerWorkerRuntimeDeps:
    return ControllerWorkerRuntimeDeps(
        load_json=deps.load_json,
        namespace_from_json_args=deps.namespace_from_json_args,
        normalize_args=deps.normalize_args,
        default_bf16_on_cuda=deps.default_bf16_on_cuda,
        task_for_name=deps.task_for_name,
        make_config=deps.make_config,
        load_key_set=deps.load_key_set,
        run_seed_phase=deps.run_seed_phase,
        run_round_model_phase=deps.run_round_model_phase,
        apply_proposal_grpo_update=deps.apply_proposal_grpo_update,
        candidate_metrics_from_json=deps.candidate_metrics_from_json,
        work_item_to_worker_payload=deps.work_item_to_worker_payload,
        run_controller_worker_generic=deps.run_controller_worker_generic,
    )


def run_candidate_worker_from_spec(
    spec_path: Path,
    *,
    deps: WorkerEntrypointDeps,
    shared_cache: Optional[MutableMapping[str, Any]] = None,
) -> CandidateMetrics:
    return _run_candidate_worker_from_spec_impl(
        spec_path,
        deps=candidate_worker_runtime_deps(deps),
        shared_cache=shared_cache,
    )


def run_candidate_worker(
    spec_path: Path,
    *,
    deps: WorkerEntrypointDeps,
    run_from_spec_fn: Optional[Callable[..., CandidateMetrics]] = None,
) -> JsonDict:
    runner = run_from_spec_fn or (lambda path, shared_cache=None: run_candidate_worker_from_spec(path, deps=deps))
    return _run_candidate_worker_impl(
        spec_path,
        deps=candidate_worker_runtime_deps(deps),
        run_from_spec_fn=runner,
    )


def run_candidate_worker_pack_from_spec(
    pack_spec_path: Path,
    *,
    deps: WorkerEntrypointDeps,
    run_from_spec_fn: Optional[Callable[..., CandidateMetrics]] = None,
) -> JsonDict:
    runner = run_from_spec_fn or (
        lambda path, shared_cache=None: run_candidate_worker_from_spec(path, deps=deps, shared_cache=shared_cache)
    )
    return _run_candidate_worker_pack_from_spec_impl(
        pack_spec_path,
        deps=candidate_worker_runtime_deps(deps),
        run_from_spec_fn=runner,
    )


def run_candidate_pack_worker(
    pack_spec_path: Path,
    *,
    deps: WorkerEntrypointDeps,
    run_from_spec_fn: Optional[Callable[..., CandidateMetrics]] = None,
) -> JsonDict:
    return run_candidate_worker_pack_from_spec(pack_spec_path, deps=deps, run_from_spec_fn=run_from_spec_fn)


def run_seed_controller_worker_from_spec(spec_path: Path, *, deps: WorkerEntrypointDeps) -> JsonDict:
    return _run_seed_controller_worker_from_spec_impl(
        spec_path,
        deps=controller_worker_runtime_deps(deps),
    )


def run_round_model_controller_worker_from_spec(spec_path: Path, *, deps: WorkerEntrypointDeps) -> JsonDict:
    return _run_round_model_controller_worker_from_spec_impl(
        spec_path,
        deps=controller_worker_runtime_deps(deps),
    )


def run_proposal_grpo_controller_worker_from_spec(spec_path: Path, *, deps: WorkerEntrypointDeps) -> JsonDict:
    return _run_proposal_grpo_controller_worker_from_spec_impl(
        spec_path,
        deps=controller_worker_runtime_deps(deps),
    )


def run_controller_worker_from_spec(spec_path: Path, *, deps: WorkerEntrypointDeps) -> JsonDict:
    return _run_controller_worker_from_spec_impl(
        spec_path,
        deps=controller_worker_runtime_deps(deps),
    )


def run_controller_worker(
    spec_path: Path,
    *,
    deps: WorkerEntrypointDeps,
    run_from_spec_fn: Optional[Callable[[Path], JsonDict]] = None,
) -> JsonDict:
    runner = run_from_spec_fn or (lambda path: run_controller_worker_from_spec(path, deps=deps))
    return _run_controller_worker_impl(
        spec_path,
        deps=controller_worker_runtime_deps(deps),
        run_from_spec_fn=runner,
    )

"""Driver-binding bridge for adaptive worker entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional

from self.core.models import CandidateMetrics
from self.adaptive.controller.worker_entrypoints import (
    WorkerEntrypointDeps,
    run_candidate_pack_worker as _run_candidate_pack_worker_entrypoint,
    run_candidate_worker as _run_candidate_worker_entrypoint,
    run_candidate_worker_from_spec as _run_candidate_worker_from_spec_entrypoint,
    run_candidate_worker_pack_from_spec as _run_candidate_worker_pack_from_spec_entrypoint,
    run_controller_worker as _run_controller_worker_entrypoint,
    run_controller_worker_from_spec as _run_controller_worker_from_spec_entrypoint,
    run_proposal_grpo_controller_worker_from_spec as _run_proposal_grpo_controller_worker_from_spec_entrypoint,
    run_round_model_controller_worker_from_spec as _run_round_model_controller_worker_from_spec_entrypoint,
    run_seed_controller_worker_from_spec as _run_seed_controller_worker_from_spec_entrypoint,
)


JsonDict = Dict[str, Any]


def worker_entrypoint_deps(bindings: Any) -> WorkerEntrypointDeps:
    return WorkerEntrypointDeps(
        load_json=bindings._load_json,
        namespace_from_json_args=bindings._namespace_from_json_args,
        normalize_args=bindings.normalize_args,
        default_bf16_on_cuda=bindings._default_bf16_on_cuda,
        task_for_name=bindings.task_for_name,
        make_config=bindings.make_config,
        load_trace_jsonl=bindings.load_trace_jsonl,
        train_and_score_candidate=bindings.train_and_score_candidate,
        write_json=bindings.write_json,
        load_key_set=bindings.load_key_set,
        run_seed_phase=bindings.run_seed_phase,
        run_round_model_phase=bindings.run_round_model_phase,
        apply_proposal_grpo_update=bindings.apply_proposal_grpo_update,
        candidate_metrics_from_json=bindings.candidate_metrics_from_json,
        work_item_to_worker_payload=bindings.work_item_to_worker_payload,
        run_controller_worker_generic=bindings._run_controller_worker_generic,
    )


def run_candidate_worker_from_spec(
    bindings: Any,
    spec_path: Path,
    shared_cache: Optional[MutableMapping[str, Any]] = None,
) -> CandidateMetrics:
    return _run_candidate_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        shared_cache=shared_cache,
    )


def run_candidate_worker(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_candidate_worker_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_candidate_worker_pack_from_spec(bindings: Any, pack_spec_path: Path) -> JsonDict:
    return _run_candidate_worker_pack_from_spec_entrypoint(
        pack_spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_candidate_pack_worker(bindings: Any, pack_spec_path: Path) -> JsonDict:
    return _run_candidate_pack_worker_entrypoint(
        pack_spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_seed_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_seed_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_round_model_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_round_model_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_proposal_grpo_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_proposal_grpo_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_controller_worker(bindings: Any, spec_path: Path) -> JsonDict:
    return _run_controller_worker_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_controller_worker_from_spec,
    )

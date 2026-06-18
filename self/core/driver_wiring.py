"""Dependency wiring for the adaptive driver entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

from self.core.candidate_dispatch_entrypoints import (
    CandidateDispatchEntrypointDeps,
    candidate_failure_metrics as _candidate_failure_metrics_entrypoint,
    collect_candidate_array_metrics as _collect_candidate_array_metrics_entrypoint,
    train_candidate_metrics as _train_candidate_metrics_entrypoint,
    train_candidates_local_parallel as _train_candidates_local_parallel_entrypoint,
    train_candidates_serial as _train_candidates_serial_entrypoint,
    train_candidates_slurm_array as _train_candidates_slurm_array_entrypoint,
)
from self.core.entrypoint import DriverEntrypointDeps, run_driver_entrypoint
from self.core.models import CandidateMetrics
from self.core.proposal_grpo_dispatch import (
    ProposalGrpoDispatchDeps,
    apply_or_dispatch_proposal_grpo_update as _apply_or_dispatch_proposal_grpo_update_impl,
)
from self.core.proposals import PromptBundle
from self.core.run_orchestration import AdaptiveRunDeps, run_adaptive_candidate_training
from self.core.worker_entrypoints import (
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


def candidate_dispatch_deps(bindings: Any) -> CandidateDispatchEntrypointDeps:
    return CandidateDispatchEntrypointDeps(
        train_and_score_candidate=bindings.train_and_score_candidate,
        candidate_failure_metrics=bindings._candidate_failure_metrics,
        collect_candidate_array_metrics=bindings._collect_candidate_array_metrics,
        train_candidates_serial=bindings.train_candidates_serial,
        train_candidates_local_parallel=bindings.train_candidates_local_parallel,
        train_candidates_slurm_array=bindings.train_candidates_slurm_array,
        subprocess_module=bindings.subprocess,
    )


def candidate_failure_metrics(bindings: Any, **kwargs: Any) -> CandidateMetrics:
    return _candidate_failure_metrics_entrypoint(**kwargs)


def train_candidates_serial(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_serial_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def collect_candidate_array_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _collect_candidate_array_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidates_slurm_array(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_slurm_array_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidates_local_parallel(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidates_local_parallel_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidate_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    return _train_candidate_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


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


def apply_or_dispatch_proposal_grpo_update(
    bindings: Any,
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[CandidateMetrics],
    seed: int,
) -> tuple[str, JsonDict]:
    return _apply_or_dispatch_proposal_grpo_update_impl(
        args=args,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
        seed=seed,
        deps=ProposalGrpoDispatchDeps(
            apply_proposal_grpo_update=bindings.apply_proposal_grpo_update,
            run_controller_worker_slurm=bindings._run_controller_worker_slurm,
            ensure_dir=bindings.ensure_dir,
            write_json=bindings.write_json,
        ),
    )


def adaptive_run_deps(bindings: Any) -> AdaptiveRunDeps:
    return AdaptiveRunDeps(
        normalize_args=bindings.normalize_args,
        task_for_name=bindings.task_for_name,
        ensure_dir=bindings.ensure_dir,
        make_config=bindings.make_config,
        prepare_datasets=bindings.prepare_datasets,
        save_examples=bindings.save_examples,
        write_json=bindings.write_json,
        run_controller_worker_slurm=bindings._run_controller_worker_slurm,
        float_or_nan=bindings._float_or_nan,
        run_seed_phase=bindings.run_seed_phase,
        build_attempt_prompt=bindings.build_attempt_prompt,
        run_dry_attempt=bindings.run_dry_attempt,
        run_round_model_dispatch=bindings.run_round_model_dispatch,
        train_candidate_metrics=bindings.train_candidate_metrics,
        select_candidate=bindings.select_candidate,
        write_round_trace=bindings.write_round_trace,
        handle_attempt_outcome=bindings.handle_attempt_outcome,
        choose_default_program_pair=bindings.choose_default_program_pair,
        render_config_prompt=bindings.render_config_prompt,
        render_program_candidate_prompt=bindings.render_program_candidate_prompt,
        load_fixture_proposals=bindings.load_fixture_proposals,
        rows_for_round=bindings._rows_for_round,
        validate_proposal_rows=bindings.validate_proposal_rows,
        build_candidate_work_items=bindings.build_candidate_work_items,
        write_key_set=bindings.write_key_set,
        load_json=bindings._load_json,
        work_item_from_worker_payload=bindings.work_item_from_worker_payload,
        run_round_model_phase=bindings.run_round_model_phase,
        build_round_outcome_trace_examples=bindings.build_round_outcome_trace_examples,
        build_selected_proposal_trace_example=bindings.build_selected_proposal_trace_example,
        apply_or_dispatch_proposal_grpo_update=bindings.apply_or_dispatch_proposal_grpo_update,
        write_trace_jsonl=bindings.write_trace_jsonl,
        append_plan_log=bindings.append_plan_log,
        sanitize_json_value=bindings.sanitize_json_value,
    )


def run(bindings: Any, args: argparse.Namespace) -> JsonDict:
    return run_adaptive_candidate_training(args=args, deps=adaptive_run_deps(bindings))


def main(bindings: Any, argv: Sequence[str] | None) -> None:
    return run_driver_entrypoint(
        argv,
        deps=DriverEntrypointDeps(
            build_parser=bindings.build_parser,
            run_controller_worker=bindings.run_controller_worker,
            run_candidate_worker=bindings.run_candidate_worker,
            run_candidate_pack_worker=bindings.run_candidate_pack_worker,
            run=bindings.run,
        ),
    )

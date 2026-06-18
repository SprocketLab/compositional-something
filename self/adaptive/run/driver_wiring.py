"""Dependency wiring for the adaptive driver entry points."""

from __future__ import annotations

from typing import Any, Sequence

from self.adaptive.run.driver_candidate_dispatch_wiring import (
    candidate_dispatch_deps,
    candidate_failure_metrics,
    collect_candidate_array_metrics,
    train_candidate_metrics,
    train_candidates_local_parallel,
    train_candidates_serial,
    train_candidates_slurm_array,
)
from self.adaptive.run.driver_worker_wiring import (
    run_candidate_pack_worker,
    run_candidate_worker,
    run_candidate_worker_from_spec,
    run_candidate_worker_pack_from_spec,
    run_controller_worker,
    run_controller_worker_from_spec,
    run_proposal_grpo_controller_worker_from_spec,
    run_round_model_controller_worker_from_spec,
    run_seed_controller_worker_from_spec,
    worker_entrypoint_deps,
)
from self.adaptive.run.entrypoint import DriverEntrypointDeps, run_driver_entrypoint
from self.adaptive.run.driver_proposal_grpo_wiring import (
    apply_or_dispatch_proposal_grpo_update,
    proposal_grpo_dispatch_deps,
)
from self.adaptive.run.driver_run_wiring import adaptive_run_deps, run


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

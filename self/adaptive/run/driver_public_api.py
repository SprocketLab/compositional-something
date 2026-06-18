"""Public compatibility delegate registration for :mod:`self.adaptive.run.driver`."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from typing import Any


DRIVER_WIRING_DELEGATES: tuple[tuple[str, str], ...] = (
    ("_candidate_dispatch_deps", "candidate_dispatch_deps"),
    ("_candidate_failure_metrics", "candidate_failure_metrics"),
    ("train_candidates_serial", "train_candidates_serial"),
    ("_collect_candidate_array_metrics", "collect_candidate_array_metrics"),
    ("train_candidates_slurm_array", "train_candidates_slurm_array"),
    ("train_candidates_local_parallel", "train_candidates_local_parallel"),
    ("train_candidate_metrics", "train_candidate_metrics"),
    ("_worker_entrypoint_deps", "worker_entrypoint_deps"),
    ("run_candidate_worker_from_spec", "run_candidate_worker_from_spec"),
    ("run_candidate_worker", "run_candidate_worker"),
    ("run_candidate_worker_pack_from_spec", "run_candidate_worker_pack_from_spec"),
    ("run_candidate_pack_worker", "run_candidate_pack_worker"),
    ("run_seed_controller_worker_from_spec", "run_seed_controller_worker_from_spec"),
    ("run_round_model_controller_worker_from_spec", "run_round_model_controller_worker_from_spec"),
    ("run_proposal_grpo_controller_worker_from_spec", "run_proposal_grpo_controller_worker_from_spec"),
    ("run_controller_worker_from_spec", "run_controller_worker_from_spec"),
    ("run_controller_worker", "run_controller_worker"),
    ("apply_or_dispatch_proposal_grpo_update", "apply_or_dispatch_proposal_grpo_update"),
    ("run", "run"),
)


def make_driver_wiring_delegate(
    *,
    public_name: str,
    wiring_name: str,
    driver_wiring: Any,
    get_bindings: Callable[[], Any],
) -> Callable[..., Any]:
    """Create a driver compatibility wrapper around a wiring function."""

    def delegate(*args: Any, **kwargs: Any) -> Any:
        return getattr(driver_wiring, wiring_name)(get_bindings(), *args, **kwargs)

    delegate.__name__ = public_name
    delegate.__qualname__ = public_name
    delegate.__doc__ = f"Compatibility delegate for driver_wiring.{wiring_name}."
    return delegate


def install_driver_wiring_delegates(
    namespace: MutableMapping[str, Any],
    *,
    driver_wiring: Any,
    get_bindings: Callable[[], Any],
    delegates: Iterable[tuple[str, str]] = DRIVER_WIRING_DELEGATES,
) -> None:
    """Install public driver delegates into ``namespace``."""

    for public_name, wiring_name in delegates:
        namespace[public_name] = make_driver_wiring_delegate(
            public_name=public_name,
            wiring_name=wiring_name,
            driver_wiring=driver_wiring,
            get_bindings=get_bindings,
        )

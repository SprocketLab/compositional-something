#!/usr/bin/env python3
"""Candidate-training loop for adaptive config self-improvement."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any, List, Optional, Sequence

from self.adaptive.run.driver_compat_exports import COMPAT_EXPORT_NAMES

DEFAULT_BINDING_NAMES = (
    "PromptBundle",
    "_PATH_ARG_NAMES",
    "_default_bf16_on_cuda",
    "_float_or_nan",
    "_json_ready_args",
    "_json_ready_key",
    "_key_from_json",
    "_load_json",
    "_namespace_from_json_args",
    "_prepare_candidate_worker_specs",
    "_rows_for_round",
    "_run_controller_worker_generic",
    "_run_controller_worker_slurm",
    "append_plan_log",
    "apply_proposal_grpo_update",
    "build_attempt_prompt",
    "build_candidate_work_items",
    "build_parser",
    "build_round_outcome_trace_examples",
    "build_selected_proposal_trace_example",
    "candidate_metrics_from_json",
    "choose_default_program_pair",
    "ensure_dir",
    "handle_attempt_outcome",
    "load_fixture_proposals",
    "load_key_set",
    "load_trace_jsonl",
    "make_config",
    "normalize_args",
    "prepare_datasets",
    "render_config_prompt",
    "render_program_candidate_prompt",
    "run_dry_attempt",
    "run_round_model_dispatch",
    "run_round_model_phase",
    "run_seed_dispatch",
    "run_seed_phase",
    "sanitize_json_value",
    "save_examples",
    "select_candidate",
    "subprocess",
    "task_for_name",
    "train_and_score_candidate",
    "validate_proposal_rows",
    "work_item_from_worker_payload",
    "work_item_to_worker_payload",
    "write_json",
    "write_key_set",
    "write_round_trace",
    "write_trace_jsonl",
)

_DEFAULT_BINDING_NAME_SET = frozenset(DEFAULT_BINDING_NAMES)
_COMPAT_EXPORT_NAME_SET = frozenset(COMPAT_EXPORT_NAMES)


def _bindings() -> Any:
    return sys.modules[__name__]


def _default_bindings() -> Any:
    from self.adaptive.run import driver_default_bindings

    return driver_default_bindings


def _driver_wiring() -> Any:
    from self.adaptive.run import driver_wiring

    return driver_wiring


class _LazyDriverWiring:
    def __getattr__(self, name: str) -> Any:
        return getattr(_driver_wiring(), name)


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


def __getattr__(name: str) -> Any:
    if name in _DEFAULT_BINDING_NAME_SET:
        return getattr(_default_bindings(), name)
    if name in _COMPAT_EXPORT_NAME_SET:
        from self.adaptive.run import driver_compat_exports

        return getattr(driver_compat_exports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | _DEFAULT_BINDING_NAME_SET | _COMPAT_EXPORT_NAME_SET)


install_driver_wiring_delegates(globals(), driver_wiring=_LazyDriverWiring(), get_bindings=_bindings)


def _default_bf16_on_cuda(args: argparse.Namespace, label: str) -> None:
    return _default_bindings()._default_bf16_on_cuda(args, label)


def main(argv: Optional[Sequence[str]] = None) -> None:
    return _driver_wiring().main(_bindings(), argv)


__all__ = [name for name in __dir__() if not name.startswith("__")]


if __name__ == "__main__":
    main()

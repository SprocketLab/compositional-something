#!/usr/bin/env python3
"""Candidate-training loop for adaptive config self-improvement."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

from self.core import driver_default_bindings, driver_wiring
from self.core.driver_compat_manifest import COMPAT_EXPORT_NAMES

JsonDict = Dict[str, Any]


_DEFAULT_BINDING_NAME_SET = frozenset(driver_default_bindings.DEFAULT_BINDING_NAMES)
_COMPAT_EXPORT_NAME_SET = frozenset(COMPAT_EXPORT_NAMES)


def _bindings() -> Any:
    return sys.modules[__name__]


def __getattr__(name: str) -> Any:
    if name in _DEFAULT_BINDING_NAME_SET:
        return getattr(driver_default_bindings, name)
    if name in _COMPAT_EXPORT_NAME_SET:
        from self.core import driver_compat_exports

        return getattr(driver_compat_exports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | _DEFAULT_BINDING_NAME_SET | _COMPAT_EXPORT_NAME_SET)


__all__ = [name for name in __dir__() if not name.startswith("__")]


def _candidate_dispatch_deps() -> Any:
    return driver_wiring.candidate_dispatch_deps(_bindings())


def _candidate_failure_metrics(**kwargs: Any) -> Any:
    return driver_wiring.candidate_failure_metrics(_bindings(), **kwargs)


def train_candidates_serial(**kwargs: Any) -> List[Any]:
    return driver_wiring.train_candidates_serial(_bindings(), **kwargs)


def _collect_candidate_array_metrics(**kwargs: Any) -> List[Any]:
    return driver_wiring.collect_candidate_array_metrics(_bindings(), **kwargs)


def train_candidates_slurm_array(**kwargs: Any) -> List[Any]:
    return driver_wiring.train_candidates_slurm_array(_bindings(), **kwargs)


def train_candidates_local_parallel(**kwargs: Any) -> List[Any]:
    return driver_wiring.train_candidates_local_parallel(_bindings(), **kwargs)


def train_candidate_metrics(**kwargs: Any) -> List[Any]:
    return driver_wiring.train_candidate_metrics(_bindings(), **kwargs)


def _worker_entrypoint_deps() -> Any:
    return driver_wiring.worker_entrypoint_deps(_bindings())


def run_candidate_worker_from_spec(
    spec_path: Path,
    shared_cache: Optional[MutableMapping[str, Any]] = None,
) -> Any:
    return driver_wiring.run_candidate_worker_from_spec(_bindings(), spec_path, shared_cache=shared_cache)


def run_candidate_worker(spec_path: Path) -> JsonDict:
    return driver_wiring.run_candidate_worker(_bindings(), spec_path)


def run_candidate_worker_pack_from_spec(pack_spec_path: Path) -> JsonDict:
    return driver_wiring.run_candidate_worker_pack_from_spec(_bindings(), pack_spec_path)


def run_candidate_pack_worker(pack_spec_path: Path) -> JsonDict:
    return driver_wiring.run_candidate_pack_worker(_bindings(), pack_spec_path)


def _default_bf16_on_cuda(args: argparse.Namespace, label: str) -> None:
    return driver_default_bindings._default_bf16_on_cuda(args, label)


def run_seed_controller_worker_from_spec(spec_path: Path) -> JsonDict:
    return driver_wiring.run_seed_controller_worker_from_spec(_bindings(), spec_path)


def run_round_model_controller_worker_from_spec(spec_path: Path) -> JsonDict:
    return driver_wiring.run_round_model_controller_worker_from_spec(_bindings(), spec_path)


def run_proposal_grpo_controller_worker_from_spec(spec_path: Path) -> JsonDict:
    return driver_wiring.run_proposal_grpo_controller_worker_from_spec(_bindings(), spec_path)


def run_controller_worker_from_spec(spec_path: Path) -> JsonDict:
    return driver_wiring.run_controller_worker_from_spec(_bindings(), spec_path)


def run_controller_worker(spec_path: Path) -> JsonDict:
    return driver_wiring.run_controller_worker(_bindings(), spec_path)


def apply_or_dispatch_proposal_grpo_update(
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any],
    seed: int,
) -> Tuple[str, JsonDict]:
    return driver_wiring.apply_or_dispatch_proposal_grpo_update(
        _bindings(),
        args=args,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
        seed=seed,
    )


def run(args: argparse.Namespace) -> JsonDict:
    return driver_wiring.run(_bindings(), args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    return driver_wiring.main(_bindings(), argv)


if __name__ == "__main__":
    main()

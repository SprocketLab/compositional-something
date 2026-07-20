#!/usr/bin/env python3
"""Adaptive self-improvement CLI facade and dependency wiring.

This module stays import-light. Concrete training bindings are loaded only when
the CLI runs or a caller accesses one of the default binding names.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence


JsonDict = Dict[str, Any]


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
    "apply_synthetic_proposal_sft",
    "build_synthetic_proposal_seed_mix",
    "build_attempt_prompt",
    "build_candidate_work_items",
    "build_parser",
    "build_round_outcome_trace_examples",
    "build_selected_proposal_trace_example",
    "candidate_metrics_from_json",
    "ensure_dir",
    "handle_attempt_outcome",
    "load_fixture_proposals",
    "load_key_set",
    "load_trace_jsonl",
    "make_config",
    "normalize_args",
    "prepare_datasets",
    "render_config_prompt",
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
_DEFAULTS_LOADED = False


def _bindings() -> Any:
    return sys.modules[__name__]


def _ensure_default_bindings() -> Any:
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return _bindings()

    import subprocess as subprocess_module

    import torch

    from self.adaptive.args import build_parser, normalize_args
    from self.adaptive.attempts import build_attempt_prompt, handle_attempt_outcome, run_dry_attempt
    from self.adaptive.candidate import (
        build_candidate_work_items,
        make_config,
        prepare_candidate_worker_specs,
        select_candidate,
        train_and_score_candidate,
        work_item_from_worker_payload,
        work_item_to_worker_payload,
    )
    from self.adaptive.controller import (
        run_controller_worker as run_controller_worker_generic,
        run_controller_worker_slurm,
        run_round_model_phase,
        run_seed_phase,
    )
    from self.adaptive.proposal import (
        PromptBundle,
        _rows_for_round,
        apply_proposal_grpo_update,
        apply_synthetic_proposal_sft,
        build_synthetic_proposal_seed_mix,
        load_fixture_proposals,
        render_config_prompt,
        validate_proposal_rows,
        write_trace_jsonl,
    )
    from self.adaptive.run import (
        append_plan_log,
        load_trace_jsonl,
        prepare_datasets,
        run_round_model_dispatch,
        run_seed_dispatch,
    )
    from self.adaptive.traces import (
        build_round_outcome_trace_examples,
        build_selected_proposal_trace_example,
        write_round_trace,
    )
    from self.core import worker_io
    from self.core.data_io import ensure_dir, sanitize_json_value, save_examples, write_json
    from self.core.models import candidate_metrics_from_json, float_or_nan
    from self.core.task_protocols import task_for_name

    def default_bf16_on_cuda(args: argparse.Namespace, label: str) -> None:
        if not args.bf16 and not args.fp16 and torch.cuda.is_available():
            args.bf16 = True
            print(f"[INFO] {label} defaulting to bf16 on CUDA.", flush=True)

    globals().update(
        {
            "PromptBundle": PromptBundle,
            "_PATH_ARG_NAMES": worker_io.PATH_ARG_NAMES,
            "_default_bf16_on_cuda": default_bf16_on_cuda,
            "_float_or_nan": float_or_nan,
            "_json_ready_args": worker_io.json_ready_args,
            "_json_ready_key": worker_io.json_ready_key,
            "_key_from_json": worker_io.key_from_json,
            "_load_json": worker_io.load_json,
            "_namespace_from_json_args": worker_io.namespace_from_json_args,
            "_prepare_candidate_worker_specs": prepare_candidate_worker_specs,
            "_rows_for_round": _rows_for_round,
            "_run_controller_worker_generic": run_controller_worker_generic,
            "_run_controller_worker_slurm": run_controller_worker_slurm,
            "append_plan_log": append_plan_log,
            "apply_proposal_grpo_update": apply_proposal_grpo_update,
            "apply_synthetic_proposal_sft": apply_synthetic_proposal_sft,
            "build_synthetic_proposal_seed_mix": build_synthetic_proposal_seed_mix,
            "build_attempt_prompt": build_attempt_prompt,
            "build_candidate_work_items": build_candidate_work_items,
            "build_parser": build_parser,
            "build_round_outcome_trace_examples": build_round_outcome_trace_examples,
            "build_selected_proposal_trace_example": build_selected_proposal_trace_example,
            "candidate_metrics_from_json": candidate_metrics_from_json,
            "ensure_dir": ensure_dir,
            "handle_attempt_outcome": handle_attempt_outcome,
            "load_fixture_proposals": load_fixture_proposals,
            "load_key_set": worker_io.load_key_set,
            "load_trace_jsonl": load_trace_jsonl,
            "make_config": make_config,
            "normalize_args": normalize_args,
            "prepare_datasets": prepare_datasets,
            "render_config_prompt": render_config_prompt,
            "run_dry_attempt": run_dry_attempt,
            "run_round_model_dispatch": run_round_model_dispatch,
            "run_round_model_phase": run_round_model_phase,
            "run_seed_dispatch": run_seed_dispatch,
            "run_seed_phase": run_seed_phase,
            "sanitize_json_value": sanitize_json_value,
            "save_examples": save_examples,
            "select_candidate": select_candidate,
            "subprocess": subprocess_module,
            "task_for_name": task_for_name,
            "train_and_score_candidate": train_and_score_candidate,
            "validate_proposal_rows": validate_proposal_rows,
            "work_item_from_worker_payload": work_item_from_worker_payload,
            "work_item_to_worker_payload": work_item_to_worker_payload,
            "write_json": write_json,
            "write_key_set": worker_io.write_key_set,
            "write_round_trace": write_round_trace,
            "write_trace_jsonl": write_trace_jsonl,
        }
    )
    _DEFAULTS_LOADED = True
    return _bindings()


def _default_bindings() -> Any:
    return _ensure_default_bindings()


@dataclass(frozen=True, slots=True)
class DriverEntrypointDeps:
    build_parser: Callable[[], argparse.ArgumentParser]
    run_controller_worker: Callable[[Path], JsonDict]
    run_candidate_worker: Callable[[Path], JsonDict]
    run_candidate_pack_worker: Callable[[Path], JsonDict]
    run: Callable[[argparse.Namespace], JsonDict]


def run_driver_entrypoint(argv: Optional[Sequence[str]], deps: DriverEntrypointDeps) -> None:
    parser = deps.build_parser()
    args = parser.parse_args(argv)
    if args.run_controller_worker:
        if args.controller_worker_spec is None:
            parser.error("--controller-worker-spec is required with --run-controller-worker")
        summary = deps.run_controller_worker(args.controller_worker_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.run_candidate_worker:
        if args.candidate_worker_spec is None:
            parser.error("--candidate-worker-spec is required with --run-candidate-worker")
        summary = deps.run_candidate_worker(args.candidate_worker_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.run_candidate_pack_worker:
        if args.candidate_worker_pack_spec is None:
            parser.error("--candidate-worker-pack-spec is required with --run-candidate-pack-worker")
        summary = deps.run_candidate_pack_worker(args.candidate_worker_pack_spec)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.task is None:
        parser.error("--task is required unless --run-candidate-worker is set")
    summary = deps.run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


@dataclass(frozen=True, slots=True)
class AdaptiveRunDeps:
    normalize_args: Any
    task_for_name: Any
    ensure_dir: Any
    make_config: Any
    prepare_datasets: Any
    save_examples: Any
    write_json: Any
    run_controller_worker_slurm: Any
    float_or_nan: Any
    run_seed_phase: Any
    build_synthetic_proposal_seed_mix: Any
    apply_synthetic_proposal_sft: Any
    build_attempt_prompt: Any
    run_dry_attempt: Any
    run_round_model_dispatch: Any
    train_candidate_metrics: Any
    select_candidate: Any
    write_round_trace: Any
    handle_attempt_outcome: Any
    render_config_prompt: Any
    load_fixture_proposals: Any
    rows_for_round: Any
    validate_proposal_rows: Any
    build_candidate_work_items: Any
    write_key_set: Any
    load_json: Any
    work_item_from_worker_payload: Any
    run_round_model_phase: Any
    build_round_outcome_trace_examples: Any
    build_selected_proposal_trace_example: Any
    apply_or_dispatch_proposal_grpo_update: Any
    write_trace_jsonl: Any
    append_plan_log: Any
    sanitize_json_value: Any


def candidate_dispatch_deps(bindings: Any) -> CandidateDispatchEntrypointDeps:
    from self.adaptive.candidate import build_candidate_dispatch_deps

    return build_candidate_dispatch_deps(bindings)


def candidate_failure_metrics(bindings: Any, **kwargs: Any) -> CandidateMetrics:
    from self.adaptive.candidate import (
        candidate_failure_metrics_with_deps as candidate_failure_metrics_entrypoint,
    )

    return candidate_failure_metrics_entrypoint(**kwargs)


def train_candidates_serial(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    from self.adaptive.candidate import (
        train_candidates_serial_with_deps as train_candidates_serial_entrypoint,
    )

    return train_candidates_serial_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def collect_candidate_worker_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    from self.adaptive.candidate import (
        collect_candidate_worker_metrics_with_deps as collect_candidate_worker_metrics_entrypoint,
    )

    return collect_candidate_worker_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))

def train_candidates_local_parallel(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    from self.adaptive.candidate import (
        train_candidates_local_parallel_with_deps as train_candidates_local_parallel_entrypoint,
    )

    return train_candidates_local_parallel_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def train_candidate_metrics(bindings: Any, **kwargs: Any) -> list[CandidateMetrics]:
    from self.adaptive.candidate import (
        train_candidate_metrics_with_deps as train_candidate_metrics_entrypoint,
    )

    return train_candidate_metrics_entrypoint(**kwargs, deps=candidate_dispatch_deps(bindings))


def worker_entrypoint_deps(bindings: Any) -> WorkerEntrypointDeps:
    from self.adaptive.controller import WorkerEntrypointDeps

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
    from self.adaptive.controller import (
        run_candidate_worker_from_spec as run_candidate_worker_from_spec_entrypoint,
    )

    return run_candidate_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        shared_cache=shared_cache,
    )


def run_candidate_worker(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import run_candidate_worker as run_candidate_worker_entrypoint

    return run_candidate_worker_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_candidate_worker_pack_from_spec(bindings: Any, pack_spec_path: Path) -> JsonDict:
    from self.adaptive.controller import (
        run_candidate_worker_pack_from_spec as run_candidate_worker_pack_from_spec_entrypoint,
    )

    return run_candidate_worker_pack_from_spec_entrypoint(
        pack_spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_candidate_pack_worker(bindings: Any, pack_spec_path: Path) -> JsonDict:
    from self.adaptive.controller import run_candidate_pack_worker as run_candidate_pack_worker_entrypoint

    return run_candidate_pack_worker_entrypoint(
        pack_spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_candidate_worker_from_spec,
    )


def run_seed_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import (
        run_seed_controller_worker_from_spec as run_seed_controller_worker_from_spec_entrypoint,
    )

    return run_seed_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_round_model_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import (
        run_round_model_controller_worker_from_spec as run_round_model_controller_worker_from_spec_entrypoint,
    )

    return run_round_model_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_proposal_grpo_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import (
        run_proposal_grpo_controller_worker_from_spec as run_proposal_grpo_controller_worker_from_spec_entrypoint,
    )

    return run_proposal_grpo_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_controller_worker_from_spec(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import run_controller_worker_from_spec as run_controller_worker_from_spec_entrypoint

    return run_controller_worker_from_spec_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
    )


def run_controller_worker(bindings: Any, spec_path: Path) -> JsonDict:
    from self.adaptive.controller import run_controller_worker as run_controller_worker_entrypoint

    return run_controller_worker_entrypoint(
        spec_path,
        deps=worker_entrypoint_deps(bindings),
        run_from_spec_fn=bindings.run_controller_worker_from_spec,
    )


def proposal_grpo_dispatch_deps(bindings: Any) -> ProposalGrpoDispatchDeps:
    from self.adaptive.proposal import ProposalGrpoDispatchDeps

    return ProposalGrpoDispatchDeps(
        apply_proposal_grpo_update=bindings.apply_proposal_grpo_update,
        run_controller_worker_slurm=bindings._run_controller_worker_slurm,
        ensure_dir=bindings.ensure_dir,
        write_json=bindings.write_json,
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
    proposal_trace_buffer: Sequence[Any] = (),
    confirmed_candidate_metrics: Sequence[CandidateMetrics] = (),
) -> tuple[str, JsonDict]:
    from self.adaptive.proposal import (
        apply_or_dispatch_proposal_grpo_update as apply_or_dispatch_proposal_grpo_update_impl,
    )

    return apply_or_dispatch_proposal_grpo_update_impl(
        args=args,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
        proposal_trace_buffer=proposal_trace_buffer,
        confirmed_candidate_metrics=confirmed_candidate_metrics,
        seed=seed,
        deps=proposal_grpo_dispatch_deps(bindings),
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
        build_synthetic_proposal_seed_mix=bindings.build_synthetic_proposal_seed_mix,
        apply_synthetic_proposal_sft=bindings.apply_synthetic_proposal_sft,
        build_attempt_prompt=bindings.build_attempt_prompt,
        run_dry_attempt=bindings.run_dry_attempt,
        run_round_model_dispatch=bindings.run_round_model_dispatch,
        train_candidate_metrics=bindings.train_candidate_metrics,
        select_candidate=bindings.select_candidate,
        write_round_trace=bindings.write_round_trace,
        handle_attempt_outcome=bindings.handle_attempt_outcome,
        render_config_prompt=bindings.render_config_prompt,
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
    from self.adaptive.run import run_adaptive_candidate_training

    return run_adaptive_candidate_training(args=args, deps=adaptive_run_deps(bindings))


def driver_wiring_main(bindings: Any, argv: Sequence[str] | None) -> None:
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


DRIVER_WIRING_DELEGATES: tuple[tuple[str, str], ...] = (
    ("_candidate_dispatch_deps", "candidate_dispatch_deps"),
    ("_candidate_failure_metrics", "candidate_failure_metrics"),
    ("train_candidates_serial", "train_candidates_serial"),
    ("_collect_candidate_worker_metrics", "collect_candidate_worker_metrics"),
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
    delegate.__doc__ = f"Compatibility delegate for driver wiring function {wiring_name}."
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


_WIRE = SimpleNamespace(
    candidate_dispatch_deps=candidate_dispatch_deps,
    candidate_failure_metrics=candidate_failure_metrics,
    train_candidates_serial=train_candidates_serial,
    collect_candidate_worker_metrics=collect_candidate_worker_metrics,
    train_candidates_local_parallel=train_candidates_local_parallel,
    train_candidate_metrics=train_candidate_metrics,
    worker_entrypoint_deps=worker_entrypoint_deps,
    run_candidate_worker_from_spec=run_candidate_worker_from_spec,
    run_candidate_worker=run_candidate_worker,
    run_candidate_worker_pack_from_spec=run_candidate_worker_pack_from_spec,
    run_candidate_pack_worker=run_candidate_pack_worker,
    run_seed_controller_worker_from_spec=run_seed_controller_worker_from_spec,
    run_round_model_controller_worker_from_spec=run_round_model_controller_worker_from_spec,
    run_proposal_grpo_controller_worker_from_spec=run_proposal_grpo_controller_worker_from_spec,
    run_controller_worker_from_spec=run_controller_worker_from_spec,
    run_controller_worker=run_controller_worker,
    proposal_grpo_dispatch_deps=proposal_grpo_dispatch_deps,
    apply_or_dispatch_proposal_grpo_update=apply_or_dispatch_proposal_grpo_update,
    adaptive_run_deps=adaptive_run_deps,
    run=run,
    main=driver_wiring_main,
)


def _driver_wiring() -> Any:
    return _WIRE


def __getattr__(name: str) -> Any:
    if name in _DEFAULT_BINDING_NAME_SET:
        return getattr(_ensure_default_bindings(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | _DEFAULT_BINDING_NAME_SET)


install_driver_wiring_delegates(globals(), driver_wiring=_WIRE, get_bindings=_bindings)


def main(argv: Optional[Sequence[str]] = None) -> None:
    return _WIRE.main(_bindings(), argv)


__all__ = [name for name in __dir__() if not name.startswith("__")]


if __name__ == "__main__":
    main()

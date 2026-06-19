#!/usr/bin/env python3
"""Adaptive controller phases, workers, and entrypoint wiring."""

from __future__ import annotations

# --- from controller_phases.py ---
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from self.adaptive.phases import PHASE_PROPOSAL_GRPO, PHASE_ROUND_MODEL, PHASE_SEED


@dataclass(frozen=True)
class SeedPhaseResult:
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Dict[int, float]
    init_final_accuracy: float
    model_dir: Optional[Path]


@dataclass(frozen=True)
class RoundModelPhaseResult:
    current_final_accuracy: float
    current_per_size_accuracy: Dict[int, float]
    prompt: Any
    proposal_results: List[dict[str, Any]]
    work_items: List[Any]


# --- from controller_workers.py ---
import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict

from self.core import worker_io
from self.core.data_io import ensure_dir
from self.core.slurm import submit_sbatch, wait_for_files_or_job_exit


JsonDict = Dict[str, Any]
RunControllerSpecFn = Callable[[Path], JsonDict]


def controller_worker_failure_path(worker_dir: Path) -> Path:
    return worker_io.controller_worker_failure_path(worker_dir)


def controller_worker_output_path(worker_dir: Path) -> Path:
    return worker_io.controller_worker_output_path(worker_dir)


def controller_worker_time_limit_for_phase(*, args: argparse.Namespace, phase: str) -> str:
    if phase == PHASE_SEED:
        return str(getattr(args, "controller_seed_worker_time_limit", None) or args.controller_worker_time_limit)
    if phase == PHASE_ROUND_MODEL:
        return str(getattr(args, "controller_round_worker_time_limit", None) or args.controller_worker_time_limit)
    if phase == PHASE_PROPOSAL_GRPO:
        return str(getattr(args, "controller_grpo_worker_time_limit", None) or args.controller_worker_time_limit)
    return str(args.controller_worker_time_limit)


def submit_controller_worker(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    spec_path: Path,
    phase: str,
) -> str:
    script = args.controller_worker_sbatch_script
    logs_dir = worker_dir / "logs"
    ensure_dir(logs_dir)
    time_limit = controller_worker_time_limit_for_phase(args=args, phase=phase)
    submission = submit_sbatch(
        script=script,
        job_name=f"adaptive-cand-controller-{phase}",
        output_path=logs_dir / "controller-%j.out",
        error_path=logs_dir / "controller-%j.err",
        time_limit=str(time_limit),
        exports={
            "CONTROLLER_WORKER_SPEC": spec_path,
            "PYTHON_BIN": sys.executable,
            "ROOT_DIR": Path.cwd(),
        },
    )
    worker_io.write_json(
        worker_dir / "slurm_dispatch.json",
        {
            "job_id": submission.job_id,
            "phase": phase,
            "command": list(submission.command),
            "stdout": submission.stdout,
            "stderr": submission.stderr,
        },
    )
    return submission.job_id


def wait_for_controller_worker(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    job_id: str,
    phase: str,
) -> None:
    wait_for_files_or_job_exit(
        job_id=job_id,
        done_paths=[
            controller_worker_output_path(worker_dir),
            controller_worker_failure_path(worker_dir),
        ],
        poll_seconds=float(args.controller_worker_poll_seconds),
        on_first_poll_message=f"[INFO] Controller GPU worker {job_id} ({phase}) is running/pending.",
    )


def run_controller_worker_slurm(
    *,
    args: argparse.Namespace,
    worker_dir: Path,
    phase: str,
    payload: JsonDict,
) -> JsonDict:
    ensure_dir(worker_dir)
    spec_path = worker_dir / "worker_spec.json"
    output_path = controller_worker_output_path(worker_dir)
    failure_path = controller_worker_failure_path(worker_dir)
    if output_path.exists():
        return worker_io.load_json(output_path)
    if failure_path.exists():
        raise RuntimeError(f"Controller worker {phase} has existing failure: {worker_io.load_json(failure_path)}")
    args_payload = worker_io.clear_worker_entry_flags(worker_io.json_ready_args(args))
    worker_io.write_json(
        spec_path,
        {
            "phase": phase,
            "args": args_payload,
            "worker_dir": str(worker_dir),
            **payload,
        },
    )
    job_id = submit_controller_worker(args=args, worker_dir=worker_dir, spec_path=spec_path, phase=phase)
    print(f"[INFO] Submitted controller GPU worker {job_id} for phase={phase}.", flush=True)
    wait_for_controller_worker(args=args, worker_dir=worker_dir, job_id=job_id, phase=phase)
    if failure_path.exists():
        raise RuntimeError(f"Controller worker {phase} failed: {worker_io.load_json(failure_path)}")
    if not output_path.exists():
        raise RuntimeError(f"Controller worker {phase} finished without worker_output.json")
    return worker_io.load_json(output_path)


def run_controller_worker(
    *,
    spec_path: Path,
    run_from_spec_fn: RunControllerSpecFn,
) -> JsonDict:
    payload: JsonDict | None = None
    try:
        payload = worker_io.load_json(spec_path)
        worker_dir = Path(payload["worker_dir"])
        output = run_from_spec_fn(spec_path)
        worker_io.write_json(controller_worker_output_path(worker_dir), output)
        return output
    except Exception as exc:
        if payload is not None:
            try:
                worker_dir = Path(payload["worker_dir"])
                worker_io.write_json(
                    controller_worker_failure_path(worker_dir),
                    {
                        "spec_path": str(spec_path),
                        "phase": payload.get("phase"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
        raise


# --- from controller_worker_runtime.py ---
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from transformers import set_seed

from self.core.data_io import load_examples
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ControllerWorkerRuntimeDeps:
    load_json: Callable[[Path], Any]
    namespace_from_json_args: Callable[[Any], argparse.Namespace]
    normalize_args: Callable[[argparse.Namespace], argparse.Namespace]
    default_bf16_on_cuda: Callable[[argparse.Namespace, str], None]
    task_for_name: Callable[[str], Any]
    make_config: Callable[[argparse.Namespace], Any]
    load_key_set: Callable[[Path], set[Any]]
    run_seed_phase: Callable[..., Any]
    run_round_model_phase: Callable[..., Any]
    apply_proposal_grpo_update: Callable[..., tuple[str, JsonDict]]
    candidate_metrics_from_json: Callable[[Any], Any]
    work_item_to_worker_payload: Callable[..., JsonDict]
    run_controller_worker_generic: Callable[..., JsonDict]


def run_seed_controller_worker_from_spec(
    spec_path: Path,
    *,
    deps: ControllerWorkerRuntimeDeps,
) -> JsonDict:
    payload = deps.load_json(spec_path)
    args = deps.namespace_from_json_args(payload["args"])
    args.run_controller_worker = True
    args.controller_worker_spec = spec_path
    args = deps.normalize_args(args)
    deps.default_bf16_on_cuda(args, "Controller seed worker")
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    config = deps.make_config(args)
    seed = int(payload["seed"])
    set_seed(seed)
    source_examples = load_examples(Path(payload["source_examples_path"]), task.deserialize_example)
    eval_examples = load_examples(Path(payload["eval_examples_path"]), task.deserialize_example)
    output_dir = Path(payload["output_dir"])
    result = deps.run_seed_phase(
        args=args,
        task=task,
        config=config,
        source_examples=source_examples,
        eval_examples=eval_examples,
        output_dir=output_dir,
        seed=seed,
    )
    return {
        "current_checkpoint": result.current_checkpoint,
        "current_final_accuracy": result.current_final_accuracy,
        "current_per_size_accuracy": {
            str(size): score for size, score in result.current_per_size_accuracy.items()
        },
        "init_final_accuracy": result.init_final_accuracy,
        "model_dir": str(result.model_dir) if result.model_dir is not None else None,
    }


def run_round_model_controller_worker_from_spec(
    spec_path: Path,
    *,
    deps: ControllerWorkerRuntimeDeps,
) -> JsonDict:
    payload = deps.load_json(spec_path)
    args = deps.namespace_from_json_args(payload["args"])
    args.run_controller_worker = True
    args.controller_worker_spec = spec_path
    args = deps.normalize_args(args)
    deps.default_bf16_on_cuda(args, "Controller round worker")
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    config = deps.make_config(args)
    seed = int(payload["seed"])

    round_dir = Path(payload["round_dir"])
    source_examples = load_examples(Path(payload["source_examples_path"]), task.deserialize_example)
    eval_examples = load_examples(Path(payload["eval_examples_path"]), task.deserialize_example)
    exclude_keys = deps.load_key_set(Path(payload["exclude_keys_path"]))
    source_sizes = {int(size) for size in payload["source_sizes"]}
    result = deps.run_round_model_phase(
        args=args,
        task=task,
        config=config,
        current_checkpoint=str(payload["current_checkpoint"]),
        round_dir=round_dir,
        source_examples=source_examples,
        eval_examples=eval_examples,
        exclude_keys=exclude_keys,
        source_sizes=source_sizes,
        selected_round_for_prompt=int(payload["selected_round_for_prompt"]),
        attempt_index=int(payload["attempt_index"]),
        selected_rounds=int(payload["selected_rounds"]),
        consecutive_no_selection=int(payload["consecutive_no_selection"]),
        init_final_accuracy=float(payload["init_final_accuracy"]),
        seed=seed,
    )
    return {
        "current_final_accuracy": result.current_final_accuracy,
        "current_per_size_accuracy": {
            str(size): score for size, score in result.current_per_size_accuracy.items()
        },
        "prompt_path": str(round_dir / "proposal_prompt.json"),
        "proposal_results_path": str(round_dir / "proposal_results.json"),
        "raw_proposals_path": str(round_dir / "raw_proposals.json"),
        "work_items": [
            deps.work_item_to_worker_payload(item=item, round_dir=round_dir)
            for item in result.work_items
        ],
    }


def run_proposal_grpo_controller_worker_from_spec(
    spec_path: Path,
    *,
    deps: ControllerWorkerRuntimeDeps,
) -> JsonDict:
    payload = deps.load_json(spec_path)
    args = deps.namespace_from_json_args(payload["args"])
    args.run_controller_worker = True
    args.controller_worker_spec = spec_path
    args = deps.normalize_args(args)
    deps.default_bf16_on_cuda(args, "Controller GRPO worker")
    prompt_payload = deps.load_json(Path(payload["prompt_path"]))
    prompt = PromptBundle(
        system=str(prompt_payload.get("system", "")),
        user=str(prompt_payload.get("user", "")),
    )
    next_checkpoint, metrics = deps.apply_proposal_grpo_update(
        args=args,
        source_checkpoint=str(payload["source_checkpoint"]),
        output_dir=Path(payload["proposal_grpo_dir"]),
        prompt=prompt,
        proposal_results=deps.load_json(Path(payload["proposal_results_path"])),
        candidate_metrics=[
            deps.candidate_metrics_from_json(item)
            for item in (deps.load_json(Path(payload["candidate_metrics_path"])) or [])
        ],
        seed=int(payload["seed"]),
    )
    return {
        "next_checkpoint": next_checkpoint,
        "proposal_grpo_metrics": metrics,
    }


def run_controller_worker_from_spec(
    spec_path: Path,
    *,
    deps: ControllerWorkerRuntimeDeps,
) -> JsonDict:
    payload = deps.load_json(spec_path)
    phase = str(payload.get("phase", ""))
    if phase == PHASE_SEED:
        return run_seed_controller_worker_from_spec(spec_path, deps=deps)
    if phase == PHASE_ROUND_MODEL:
        return run_round_model_controller_worker_from_spec(spec_path, deps=deps)
    if phase == PHASE_PROPOSAL_GRPO:
        return run_proposal_grpo_controller_worker_from_spec(spec_path, deps=deps)
    raise ValueError(f"Unsupported controller worker phase={phase!r}.")


def run_controller_worker(
    spec_path: Path,
    *,
    deps: ControllerWorkerRuntimeDeps,
    run_from_spec_fn: Callable[[Path], JsonDict],
) -> JsonDict:
    return deps.run_controller_worker_generic(
        spec_path=spec_path,
        run_from_spec_fn=run_from_spec_fn,
    )


# --- from controller_phase_runtime.py ---
import argparse
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from transformers import set_seed

from self.adaptive.attempts import build_attempt_prompt
from self.adaptive.candidate import (
    attach_pseudo_labels,
    build_candidate_work_items,
    evaluate_model,
    train_checkpoint,
)
from self.adaptive.proposal import load_or_generate_proposal_rows
from self.adaptive.proposal import validate_proposal_rows
from self.core import worker_io
from self.core.data_io import ensure_dir
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import TrainingConfig

JsonDict = Dict[str, Any]


def run_seed_phase(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    output_dir: Path,
    seed: int,
) -> SeedPhaseResult:
    if args.treat_seed_as_round_zero:
        current_checkpoint = args.model_name
        model, tokenizer = instantiate_model_and_tokenizer(
            current_checkpoint,
            bf16=args.bf16,
            fp16=args.fp16,
            init_from_scratch=args.init_from_scratch,
            tokenizer_mode=args.tokenizer_mode,
            recipe=args.recipe,
        )
        model_dir: Optional[Path] = None
    else:
        seed_dir = output_dir / "round_00" / "seed_training"
        model, tokenizer, model_dir = train_checkpoint(
            source_checkpoint=args.model_name,
            train_examples=source_examples,
            output_dir=seed_dir,
            task=task,
            args=args,
            config=config,
            seed=seed,
            recipe_phase_name="seed",
        )
        current_checkpoint = str(model_dir)

    try:
        current_final_accuracy, current_per_size_accuracy = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            task=task,
            examples=eval_examples,
            batch_size=config.per_device_eval_batch_size,
            decode_max_new_tokens=config.decode_max_new_tokens,
        )
        init_final_accuracy = (
            float(args.init_final_accuracy) if args.init_final_accuracy is not None else current_final_accuracy
        )
        round_zero_dir = output_dir / "round_00"
        ensure_dir(round_zero_dir)
        metrics_payload: JsonDict = {
            "round": 0,
            "eval_accuracy": current_final_accuracy,
            "per_size_accuracy": current_per_size_accuracy,
            "init_final_accuracy": init_final_accuracy,
        }
        if args.treat_seed_as_round_zero:
            metrics_payload.update(
                {
                    "seed_checkpoint": current_checkpoint,
                    "treat_seed_as_round_zero": True,
                }
            )
        else:
            metrics_payload.update(
                {
                    "model_dir": current_checkpoint,
                    "train_examples": len(source_examples),
                }
            )
        worker_io.write_json(round_zero_dir / "metrics.json", metrics_payload)
        return SeedPhaseResult(
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy={int(size): float(score) for size, score in current_per_size_accuracy.items()},
            init_final_accuracy=init_final_accuracy,
            model_dir=model_dir,
        )
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_round_model_phase(
    *,
    args: argparse.Namespace,
    task: Any,
    config: TrainingConfig,
    current_checkpoint: str,
    round_dir: Path,
    source_examples: Sequence[Any],
    eval_examples: Sequence[Any],
    exclude_keys: set[Any],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    init_final_accuracy: float,
    seed: int,
) -> RoundModelPhaseResult:
    set_seed(seed)
    rng = random.Random(seed)
    frontier_min = args.frontier_min_size
    frontier_max = args.frontier_max_size
    current_model, current_tokenizer = instantiate_model_and_tokenizer(
        current_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        current_final_accuracy, current_per_size_accuracy = evaluate_model(
            model=current_model,
            tokenizer=current_tokenizer,
            task=task,
            examples=eval_examples,
            batch_size=config.per_device_eval_batch_size,
            decode_max_new_tokens=config.decode_max_new_tokens,
        )
        attempt_prompt = build_attempt_prompt(
            args=args,
            current_checkpoint=current_checkpoint,
            current_final_accuracy=current_final_accuracy,
            init_final_accuracy=init_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            source_sizes=source_sizes,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            extra_aggregate_metrics={"proposal_output_schema": args.proposal_output_schema},
        )
        prompt = attempt_prompt.prompt
        default_program_pair = attempt_prompt.default_program_pair
        worker_io.write_json(round_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})

        rows = load_or_generate_proposal_rows(
            args=args,
            prompt=prompt,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
            round_index=selected_round_for_prompt,
            attempt_index=attempt_index,
        )
        proposal_results = validate_proposal_rows(
            rows=rows,
            args=args,
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
            default_pair=default_program_pair,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
        )
        worker_io.write_json(round_dir / "raw_proposals.json", rows)
        worker_io.write_json(round_dir / "proposal_results.json", proposal_results)

        work_items = build_candidate_work_items(
            args=args,
            task=task,
            round_dir=round_dir,
            proposal_results=proposal_results,
            source_examples=source_examples,
            exclude_keys=exclude_keys,
            rng=rng,
        )
        work_items = attach_pseudo_labels(
            args=args,
            task=task,
            round_dir=round_dir,
            work_items=work_items,
            source_examples=source_examples,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
            config=config,
        )
        return RoundModelPhaseResult(
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy={
                int(size): float(score) for size, score in current_per_size_accuracy.items()
            },
            prompt=prompt,
            proposal_results=list(proposal_results),
            work_items=list(work_items),
        )
    finally:
        del current_model
        del current_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- from worker_entrypoints.py ---
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional

from self.adaptive.candidate import (
    CandidateWorkerRuntimeDeps,
    run_candidate_worker as _run_candidate_worker_impl,
    run_candidate_worker_from_spec as _run_candidate_worker_from_spec_impl,
)
from self.adaptive.candidate import (
    run_candidate_worker_pack_from_spec as _run_candidate_worker_pack_from_spec_impl,
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

"""Controller-worker spec entry point runtime."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from transformers import set_seed

from self.adaptive.controller.controller_phases import PHASE_PROPOSAL_GRPO, PHASE_ROUND_MODEL, PHASE_SEED
from self.core.data_io import load_examples
from self.adaptive.proposals.proposal_prompts import PromptBundle


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

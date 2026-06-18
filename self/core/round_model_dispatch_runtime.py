"""Round-model local/Slurm dispatch for adaptive attempts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.core.controller_phases import PHASE_ROUND_MODEL
from self.core.models import CandidateWorkItem
from self.core.proposals import PromptBundle
from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class RoundModelDispatchDeps:
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]
    write_key_set: Callable[[Path, set[Any]], None]
    run_controller_worker_slurm: Callable[..., Mapping[str, Any]]
    float_or_nan: Callable[[Any], float]
    load_json: Callable[[Path], Any]
    work_item_from_worker_payload: Callable[..., CandidateWorkItem]
    run_round_model_phase: Callable[..., Any]


@dataclass(frozen=True)
class RoundModelDispatchResult:
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    prompt: PromptBundle
    proposal_results: Sequence[Mapping[str, Any]]
    work_items: Sequence[CandidateWorkItem]


def run_round_model_dispatch(
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
    deps: RoundModelDispatchDeps,
) -> RoundModelDispatchResult:
    if args.controller_execution_mode == "slurm":
        return _run_round_model_slurm(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            round_dir=round_dir,
            source_examples=source_examples,
            eval_examples=eval_examples,
            exclude_keys=exclude_keys,
            source_sizes=source_sizes,
            selected_round_for_prompt=selected_round_for_prompt,
            attempt_index=attempt_index,
            selected_rounds=selected_rounds,
            consecutive_no_selection=consecutive_no_selection,
            init_final_accuracy=init_final_accuracy,
            deps=deps,
        )

    round_result = deps.run_round_model_phase(
        args=args,
        task=task,
        config=config,
        current_checkpoint=current_checkpoint,
        round_dir=round_dir,
        source_examples=source_examples,
        eval_examples=eval_examples,
        exclude_keys=exclude_keys,
        source_sizes=source_sizes,
        selected_round_for_prompt=selected_round_for_prompt,
        attempt_index=attempt_index,
        selected_rounds=selected_rounds,
        consecutive_no_selection=consecutive_no_selection,
        init_final_accuracy=init_final_accuracy,
        seed=args.seed + attempt_index * 7919,
    )
    return RoundModelDispatchResult(
        current_final_accuracy=round_result.current_final_accuracy,
        current_per_size_accuracy=round_result.current_per_size_accuracy,
        prompt=round_result.prompt,
        proposal_results=round_result.proposal_results,
        work_items=round_result.work_items,
    )


def _run_round_model_slurm(
    *,
    args: argparse.Namespace,
    task: Any,
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
    deps: RoundModelDispatchDeps,
) -> RoundModelDispatchResult:
    controller_input_dir = round_dir / "controller_worker" / "inputs"
    controller_input_dir.mkdir(parents=True, exist_ok=True)
    source_examples_path = controller_input_dir / "source_examples.jsonl"
    eval_examples_path = controller_input_dir / "eval_examples.jsonl"
    exclude_keys_path = controller_input_dir / "exclude_keys.json"
    deps.save_examples(source_examples_path, source_examples, task.serialize_example)
    deps.save_examples(eval_examples_path, eval_examples, task.serialize_example)
    deps.write_key_set(exclude_keys_path, exclude_keys)
    round_output = deps.run_controller_worker_slurm(
        args=args,
        worker_dir=round_dir / "controller_worker",
        phase=PHASE_ROUND_MODEL,
        payload={
            "current_checkpoint": current_checkpoint,
            "round_dir": str(round_dir),
            "source_examples_path": str(source_examples_path),
            "eval_examples_path": str(eval_examples_path),
            "exclude_keys_path": str(exclude_keys_path),
            "source_sizes": sorted(source_sizes),
            "selected_round_for_prompt": selected_round_for_prompt,
            "attempt_index": attempt_index,
            "selected_rounds": selected_rounds,
            "consecutive_no_selection": consecutive_no_selection,
            "init_final_accuracy": init_final_accuracy,
            "seed": args.seed + attempt_index * 7919,
        },
    )
    prompt_payload = deps.load_json(Path(round_output["prompt_path"]))
    prompt = PromptBundle(
        system=str(prompt_payload.get("system", "")),
        user=str(prompt_payload.get("user", "")),
    )
    proposal_results = deps.load_json(Path(round_output["proposal_results_path"]))
    work_items = [
        deps.work_item_from_worker_payload(payload=item_payload, task=task)
        for item_payload in round_output.get("work_items", [])
    ]
    return RoundModelDispatchResult(
        current_final_accuracy=deps.float_or_nan(round_output.get("current_final_accuracy")),
        current_per_size_accuracy={
            int(size): float(score)
            for size, score in dict(round_output.get("current_per_size_accuracy", {})).items()
            if score is not None
        },
        prompt=prompt,
        proposal_results=proposal_results,
        work_items=work_items,
    )

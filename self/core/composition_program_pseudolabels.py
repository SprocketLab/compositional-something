"""Executable composition-program pseudolabel helpers."""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from self.core.models import ExecutableProposal
from self.core.program_sandbox import execute_program_cases
from self.core.program_sandbox_models import SandboxCase
from self.tasks.bit_common import normalize_bit_target_mode
from self.tasks.bit_parsing import RUN_LENGTH_TARGET_RUN_STATE


JsonDict = Dict[str, Any]


def target_pattern_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return r"\d+"
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return r"\d+\|[0-9A-Z]\|\d+\|[0-9A-Z]\|\d+"
        return r"\d+\|\d+\|\d+"
    raise ValueError(f"Unsupported task={task_name!r}.")


def _component_payload(
    *,
    key: Any,
    prediction: str,
    part_index: int,
) -> JsonDict:
    size = int(key[0])
    return {
        "size": size,
        "input_id": f"component_{part_index}",
        "prediction": str(prediction),
        "metadata": {
            "part_index": part_index,
            "size": size,
        },
    }


def compose_program_pseudo_examples(
    *,
    task_name: str,
    task: Any,
    proposal: ExecutableProposal,
    composed_examples: Sequence[Any],
    component_map: Dict[Any, List[Any]],
    component_predictions: Mapping[Any, str],
    args: argparse.Namespace,
) -> Tuple[List[Any], JsonDict]:
    cases: List[SandboxCase] = []
    case_examples: List[Any] = []
    missing = 0
    for index, example in enumerate(composed_examples):
        key = task.key_for_example(example)
        component_keys = component_map.get(key)
        if not component_keys:
            missing += 1
            continue
        components: List[JsonDict] = []
        failed = False
        for part_index, component_key in enumerate(component_keys):
            prediction = component_predictions.get(component_key)
            if prediction is None:
                failed = True
                break
            components.append(
                _component_payload(
                    key=component_key,
                    prediction=prediction,
                    part_index=part_index,
                )
            )
        if failed:
            missing += 1
            continue
        cases.append(
            SandboxCase(
                name=f"candidate_{index}",
                components=components,
                metadata={
                    "task": task_name,
                    "condition": proposal.condition,
                    "target_size": proposal.target,
                    "source_sizes": [proposal.left, proposal.right],
                    "guard": proposal.guard,
                    "component_count": len(components),
                    "representation": proposal.representation,
                    "target_format": proposal.target_format,
                },
                target_pattern=target_pattern_for_task(task_name, args),
            )
        )
        case_examples.append(example)

    if not cases:
        return [], {
            "mode": f"{proposal.condition}_program",
            "task": task_name,
            "left": proposal.left,
            "right": proposal.right,
            "target": proposal.target,
            "guard": proposal.guard,
            "candidate_total": len(composed_examples),
            "retained_total": 0,
            "missing_total": missing,
            "rejected_total": 0,
            "invalid_target_total": 0,
            "sandbox_category": "",
            "sandbox_message": "",
            "retained_fraction": math.nan,
        }

    batch_timeout = min(
        args.program_batch_timeout_seconds,
        max(args.program_timeout_seconds, 1.0 + 0.005 * len(cases)),
    )
    execution = execute_program_cases(
        proposal.code,
        cases=cases,
        timeout_seconds=batch_timeout,
    )
    if not execution.valid:
        return [], {
            "mode": f"{proposal.condition}_program",
            "task": task_name,
            "left": proposal.left,
            "right": proposal.right,
            "target": proposal.target,
            "guard": proposal.guard,
            "candidate_total": len(composed_examples),
            "retained_total": 0,
            "missing_total": missing,
            "rejected_total": 0,
            "invalid_target_total": 0,
            "sandbox_category": execution.category,
            "sandbox_message": execution.message,
            "retained_fraction": 0.0,
        }

    pseudo_examples: List[Any] = []
    rejected = 0
    invalid_target = 0
    for example, output in zip(case_examples, execution.outputs):
        if not output.get("accept"):
            rejected += 1
            continue
        target = str(output.get("target", ""))
        parsed = task.prediction_parser(target, example) if task_name == "run_length" else task.prediction_parser(target)
        if parsed is None:
            invalid_target += 1
            continue
        pseudo_examples.append(task.clone_with_override(example, parsed))

    return pseudo_examples, {
        "mode": f"{proposal.condition}_program",
        "task": task_name,
        "left": proposal.left,
        "right": proposal.right,
        "target": proposal.target,
        "guard": proposal.guard,
        "representation": proposal.representation,
        "target_format": proposal.target_format,
        "candidate_total": len(composed_examples),
        "executable_case_total": len(cases),
        "retained_total": len(pseudo_examples),
        "missing_total": missing,
        "rejected_total": rejected,
        "invalid_target_total": invalid_target,
        "sandbox_category": "",
        "sandbox_message": "",
        "batch_timeout_seconds": batch_timeout,
        "retained_fraction": len(pseudo_examples) / len(composed_examples) if composed_examples else math.nan,
    }

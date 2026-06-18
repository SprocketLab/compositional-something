"""Pseudolabel composition helpers for exact-pair adaptive proposals."""

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.addition_pipeline import AdditionExample, build_composed_pseudo_map, example_key
from self.core.models import ExecutableProposal
from self.core.program_sandbox import SandboxCase, execute_program_cases
from self.core.proposals import ConfigProposal
from self.tasks.bit_common import (
    RUN_LENGTH_TARGET_RUN_STATE,
    normalize_bit_target_mode,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
)
from self.tasks.run_length_data import (
    RunLengthExample,
    clone_run_length_with_override,
    run_length_key,
)
from self.tasks.run_length_logic import format_run_length_run_state, merge_run_state


JsonDict = Dict[str, Any]


def target_pattern_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return r"\d+"
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return r"\d+\|[0-9A-Z]\|\d+\|[0-9A-Z]\|\d+"
        return r"\d+\|\d+\|\d+"
    raise ValueError(f"Unsupported task={task_name!r}.")


def compose_addition_pseudo_examples(
    *,
    proposal: ConfigProposal,
    composed_examples: Sequence[AdditionExample],
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    component_predictions: Mapping[Tuple[int, int, int], str],
) -> Tuple[List[AdditionExample], JsonDict]:
    pseudo_map = build_composed_pseudo_map(
        {},
        composed_examples,
        component_map,
        dict(component_predictions),
        filter_component_carries=False,
        carry_error_fraction=0.0,
    )
    pseudo_examples: List[AdditionExample] = []
    missing = 0
    for example in composed_examples:
        override = pseudo_map.get(example_key(example))
        if override is None:
            missing += 1
            continue
        pseudo_examples.append(
            AdditionExample(
                a=example.a,
                b=example.b,
                result=example.result,
                digits=example.digits,
                has_carry=example.has_carry,
                operand_width=example.block_width,
                target_override=override,
            )
        )
    return pseudo_examples, {
        "mode": "compose_exact_pair",
        "task": "addition",
        "left": proposal.left,
        "right": proposal.right,
        "target": proposal.target,
        "guard": proposal.guard,
        "candidate_total": len(composed_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing,
        "retained_fraction": len(pseudo_examples) / len(composed_examples) if composed_examples else math.nan,
    }


def _parse_run_state_component(
    prediction: str,
    component_key: Tuple[int, str],
) -> Optional[Tuple[int, int, str, int, str, int]]:
    parsed = parse_run_length_run_state_prediction(
        prediction,
        RunLengthExample(
            bitstring=component_key[1],
            bits=component_key[0],
            max_run=0,
            prefix_run=0,
            suffix_run=0,
            target_mode=RUN_LENGTH_TARGET_RUN_STATE,
        ),
    )
    if parsed is None:
        return None
    max_text, prefix_symbol, prefix_text, suffix_symbol, suffix_text = parsed.split("|")
    return (
        component_key[0],
        int(max_text),
        prefix_symbol,
        int(prefix_text),
        suffix_symbol,
        int(suffix_text),
    )


def _parse_default_run_length_component(
    prediction: str,
    component_key: Tuple[int, str],
) -> Optional[Tuple[int, int, str, int, str, int]]:
    parsed = parse_run_length_prediction(
        prediction,
        RunLengthExample(
            bitstring=component_key[1],
            bits=component_key[0],
            max_run=0,
            prefix_run=0,
            suffix_run=0,
            target_mode="default",
        ),
    )
    if parsed is None:
        return None
    pieces = parsed.split("|")
    if len(pieces) != 3:
        return None
    max_run = int(pieces[0])
    prefix_run = int(pieces[1])
    suffix_run = int(pieces[2])
    bitstring = component_key[1]
    if max_run < 0 or prefix_run < 0 or suffix_run < 0:
        return None
    if max_run > component_key[0] or prefix_run > component_key[0] or suffix_run > component_key[0]:
        return None
    return (
        component_key[0],
        max_run,
        bitstring[0] if bitstring else "",
        prefix_run,
        bitstring[-1] if bitstring else "",
        suffix_run,
    )


def compose_run_length_pseudo_examples(
    *,
    proposal: ConfigProposal,
    composed_examples: Sequence[RunLengthExample],
    component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    component_predictions: Mapping[Tuple[int, str], str],
    target_mode: str,
) -> Tuple[List[RunLengthExample], JsonDict]:
    if target_mode not in {"default", RUN_LENGTH_TARGET_RUN_STATE}:
        raise ValueError(
            "Adaptive run-length candidate composition currently supports target_mode=default or run_state."
        )
    pseudo_examples: List[RunLengthExample] = []
    missing = 0
    for example in composed_examples:
        component_keys = component_map.get(run_length_key(example))
        if not component_keys:
            missing += 1
            continue
        parsed_components: List[Tuple[int, int, str, int, str, int]] = []
        failed = False
        for component_key in component_keys:
            prediction = component_predictions.get(component_key)
            if prediction is None:
                failed = True
                break
            if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
                parsed = _parse_run_state_component(prediction, component_key)
            else:
                parsed = _parse_default_run_length_component(prediction, component_key)
            if parsed is None:
                failed = True
                break
            parsed_components.append(parsed)
        if failed or not parsed_components:
            missing += 1
            continue
        merged = parsed_components[0]
        for nxt in parsed_components[1:]:
            merged = merge_run_state(merged, nxt)
        _, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = merged
        if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
            override = format_run_length_run_state(
                (max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run)
            )
        else:
            override = f"{max_run}|{prefix_run}|{suffix_run}"
        pseudo_examples.append(clone_run_length_with_override(example, override))
    return pseudo_examples, {
        "mode": "compose_exact_pair",
        "task": "run_length",
        "left": proposal.left,
        "right": proposal.right,
        "target": proposal.target,
        "guard": proposal.guard,
        "target_mode": target_mode,
        "candidate_total": len(composed_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing,
        "retained_fraction": len(pseudo_examples) / len(composed_examples) if composed_examples else math.nan,
    }


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


def compose_pseudo_examples(
    *,
    task_name: str,
    task: Any,
    proposal: ConfigProposal,
    composed_examples: Sequence[Any],
    component_map: Dict[Any, List[Any]],
    component_predictions: Mapping[Any, str],
    args: argparse.Namespace,
) -> Tuple[List[Any], JsonDict]:
    if isinstance(proposal, ExecutableProposal):
        return compose_program_pseudo_examples(
            task_name=task_name,
            task=task,
            proposal=proposal,
            composed_examples=composed_examples,
            component_map=component_map,
            component_predictions=component_predictions,
            args=args,
        )
    if task_name == "addition":
        return compose_addition_pseudo_examples(
            proposal=proposal,
            composed_examples=composed_examples,
            component_map=component_map,
            component_predictions=component_predictions,
        )
    if task_name == "run_length":
        return compose_run_length_pseudo_examples(
            proposal=proposal,
            composed_examples=composed_examples,
            component_map=component_map,
            component_predictions=component_predictions,
            target_mode=normalize_bit_target_mode(args),
        )
    raise ValueError(f"Unsupported task={task_name!r}.")

"""Proposal prompt rendering and task-specific prompt metadata."""

from __future__ import annotations

import argparse
import json
from typing import Any, List, Mapping, Optional

from self.core.data_io import sanitize_json_value
from self.core.program_sandbox_cases import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.core.program_sandbox_models import (
    SandboxCase,
)
from self.core.proposal_config_schema import ConfigProposal
from self.core.proposals import PromptBundle
from self.tasks.bit_common import RUN_LENGTH_TARGET_RUN_STATE, normalize_bit_target_mode


def target_format_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return "a non-negative integer string formed only from component prediction strings"
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
        return "max_run|prefix_run|suffix_run"
    raise ValueError(f"Unsupported task={task_name!r}.")


def component_prediction_examples_for_task(task_name: str, args: argparse.Namespace) -> List[str]:
    if task_name == "addition":
        return ["46", "064", "1002"]
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return ["3|0|2|1|1", "5|1|5|1|5"]
        return ["3|2|1", "5|5|5"]
    raise ValueError(f"Unsupported task={task_name!r}.")


def program_validation_cases(task_name: str, args: argparse.Namespace) -> List[SandboxCase]:
    if task_name == "addition":
        return build_addition_program_cases()
    if task_name == "run_length":
        if normalize_bit_target_mode(args) != RUN_LENGTH_TARGET_RUN_STATE:
            return []
        return build_run_length_program_cases(random_seed=args.seed, random_count=8)
    raise ValueError(f"Unsupported task={task_name!r}.")


def choose_default_program_pair(
    *,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    allow_repeat_targets: bool,
) -> ConfigProposal:
    for target in range(frontier_min, frontier_max + 1):
        if not allow_repeat_targets and target in source_sizes:
            continue
        pairs = [(left, target - left) for left in sorted(source_sizes) if target - left in source_sizes]
        if not pairs:
            continue
        left, right = sorted(pairs, key=lambda pair: (abs(pair[0] - pair[1]), pair[0]))[0]
        return ConfigProposal(left=left, right=right, guard="none", target=target)
    raise ValueError(
        "No driver-selected program pair is available from the current source pool "
        f"for frontier={frontier_min}..{frontier_max}."
    )


def render_program_candidate_prompt(
    *,
    args: argparse.Namespace,
    round_index: int,
    current_checkpoint: str,
    current_source: Mapping[str, Any],
    allowed_target_frontier: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
    default_pair: Optional[ConfigProposal],
) -> PromptBundle:
    target_format = target_format_for_task(args.task, args)
    examples = component_prediction_examples_for_task(args.task, args)
    common = (
        f"Task: {args.task}\n\n"
        "Current round context:\n"
        f"- round_index: {round_index}\n"
        f"- current_source_slices: {json.dumps(sanitize_json_value(dict(current_source)), sort_keys=True)}\n"
        f"- allowed_target_frontier: {json.dumps(sanitize_json_value(dict(allowed_target_frontier)), sort_keys=True)}\n"
        f"- model: {current_checkpoint}\n"
        "Aggregate diagnostics:\n"
        f"{json.dumps(sanitize_json_value(dict(aggregate_metrics)), sort_keys=True, indent=2)}\n\n"
        "Self-labeling rule:\n"
        "- The target label must be derived from current-model component predictions.\n"
        "- Do not use oracle labels or write a direct task solver from raw target inputs.\n"
        "- The driver constructs target inputs and held-out evaluation; your code only composes pseudo-labels.\n\n"
        "Function contract:\n"
        "def compose(components, metadata):\n"
        "    ...\n\n"
        "Each component is a dictionary with size, input_id, prediction, and metadata.\n"
        "The input_id is opaque. Use component predictions and sizes.\n"
        f"Target output format: {target_format}\n"
        f"Example component predictions: {json.dumps(examples)}\n"
        '- Return {"accept": True, "target": "<target string>"} or '
        '{"accept": False, "reason": "<short reason>"}.\n\n'
        "Sandbox rules:\n"
        "- exactly one top-level compose function\n"
        "- no imports, filesystem, network, subprocess, eval, exec, compile, open, globals, locals, vars, dir, getattr, setattr, delattr, or __import__\n"
        "- deterministic output on repeated calls\n"
    )
    if args.condition == "program":
        if default_pair is None:
            raise ValueError("program condition requires a driver-selected default pair")
        system = (
            "You are generating a restricted Python composition program.\n"
            "Output only Python code. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "Driver-selected source policy:\n"
            f"- left: {default_pair.left}\n"
            f"- right: {default_pair.right}\n"
            f"- target: {default_pair.target}\n"
            "- You only write the compose function for this source pair.\n\n"
            "Output only Python code."
        )
        return PromptBundle(system=system, user=user)

    if args.condition == "policy":
        system = (
            "You are generating a source-selection policy plus a restricted Python composer.\n"
            "Output only valid JSON. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "JSON schema:\n"
            "{\n"
            '  "left": integer source slice size,\n'
            '  "right": integer source slice size,\n'
            '  "guard": "none" or a listed task guard,\n'
            '  "code": "Python code defining compose(components, metadata)",\n'
            '  "notes": "optional short rationale"\n'
            "}\n\n"
            "Policy constraints:\n"
            "- left and right must be current source slices.\n"
            "- left + right must be inside the allowed target frontier.\n"
            "- Choose slices based on current slice accuracy and expected self-labeling reliability.\n\n"
            "Output only JSON."
        )
        return PromptBundle(system=system, user=user)

    if args.condition == "meta":
        system = (
            "You are generating an intermediate representation policy plus a restricted Python composer.\n"
            "Output only valid JSON. Do not include markdown or explanations."
        )
        user = (
            common
            +
            "JSON schema:\n"
            "{\n"
            '  "left": integer source slice size,\n'
            '  "right": integer source slice size,\n'
            '  "guard": "none" or a listed task guard,\n'
            '  "representation": "short name for the intermediate state you are composing through",\n'
            '  "target_format": "final target format your compose function returns",\n'
            '  "code": "Python code defining compose(components, metadata)",\n'
            '  "notes": "optional short rationale"\n'
            "}\n\n"
            "Meta-composition constraints:\n"
            "- You may invent the intermediate representation used inside compose.\n"
            f"- The final returned target must still match the task target format: {target_format}.\n"
            "- left and right must be current source slices, and left + right must be inside the allowed target frontier.\n\n"
            "Output only JSON."
        )
        return PromptBundle(system=system, user=user)

    raise ValueError(f"Unsupported condition={args.condition!r}.")

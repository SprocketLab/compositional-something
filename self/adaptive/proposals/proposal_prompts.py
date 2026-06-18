"""Proposal prompt rendering and task-specific prompt metadata."""

from __future__ import annotations

import argparse
from typing import List

from self.adaptive.sandbox.program_sandbox import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.adaptive.sandbox.program_sandbox import SandboxCase
from self.adaptive.proposals.proposal_config_schema import ConfigProposal

RUN_LENGTH_TARGET_RUN_STATE = "run_state"


def _normalize_bit_target_mode(args: argparse.Namespace, default: str = "default") -> str:
    return str(getattr(args, "target_mode", default))


def target_format_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return "a non-negative integer string formed only from component prediction strings"
    if task_name == "run_length":
        if _normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
        return "max_run|prefix_run|suffix_run"
    raise ValueError(f"Unsupported task={task_name!r}.")


def component_prediction_examples_for_task(task_name: str, args: argparse.Namespace) -> List[str]:
    if task_name == "addition":
        return ["46", "064", "1002"]
    if task_name == "run_length":
        if _normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return ["3|0|2|1|1", "5|1|5|1|5"]
        return ["3|2|1", "5|5|5"]
    raise ValueError(f"Unsupported task={task_name!r}.")


def program_validation_cases(task_name: str, args: argparse.Namespace) -> List[SandboxCase]:
    if task_name == "addition":
        return build_addition_program_cases()
    if task_name == "run_length":
        if _normalize_bit_target_mode(args) != RUN_LENGTH_TARGET_RUN_STATE:
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


__all__ = [
    "choose_default_program_pair",
    "component_prediction_examples_for_task",
    "program_validation_cases",
    "target_format_for_task",
]


import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.proposals.proposal_config_schema import ConfigProposal


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str

    def text(self) -> str:
        return f"System:\n{self.system}\n\nUser:\n{self.user}"


def _format_choices(values: Sequence[str]) -> str:
    return ", ".join(json.dumps(value) for value in values)


def render_config_prompt(
    *,
    task_name: str,
    round_index: int,
    current_source: Mapping[str, Any],
    allowed_target_frontier: Mapping[str, Any],
    aggregate_metrics: Mapping[str, Any],
    guard_choices: Sequence[str],
    model_name: str = "Qwen/Qwen3-1.7B",
    proposal_output_schema: str = "plain",
) -> PromptBundle:
    system = (
        "You are generating a composition configuration for a compositional self-improvement pipeline.\n"
        "You must output only valid JSON matching the requested schema.\n"
        "Do not include explanations, markdown, or comments.\n"
        "You cannot call tools. You only choose a composition configuration."
    )
    if proposal_output_schema == "plain":
        schema = {
            "left": "integer source slice size",
            "right": "integer source slice size",
            "guard": f"one of {_format_choices(guard_choices)}",
            "notes": "optional short string",
        }
        schema_notes = (
            "Output the configuration object directly. The driver will execute this proposal and measure reward."
        )
    elif proposal_output_schema == "action_prediction":
        schema = {
            "proposal": {
                "left": "integer source slice size",
                "right": "integer source slice size",
                "guard": f"one of {_format_choices(guard_choices)}",
                "notes": "optional short string",
            },
            "prediction": {
                "target": "integer equal to proposal.left + proposal.right",
                "expected_frontier_delta": "number; expected mean accuracy change on the static frontier",
                "expected_final_delta_from_init": "number; expected final accuracy minus initial final accuracy after training",
                "rationale": "short string; no more than one sentence",
            },
        }
        schema_notes = (
            "Output one object containing both the executable proposal and your predicted outcome. "
            "The driver will execute only proposal, then compare prediction against the realized outcome."
        )
    else:
        raise ValueError(f"Unsupported proposal_output_schema={proposal_output_schema!r}.")
    user = (
        f"Task: {task_name}\n\n"
        "Goal:\n"
        "Choose two source slices and a guard rule for composing model component predictions into pseudo-labels. "
        "Also predict the improvement you expect from this choice when the schema asks for prediction.\n\n"
        "Current round:\n"
        f"- round_index: {round_index}\n"
        f"- current_source_slices: {json.dumps(sanitize_json_value(dict(current_source)), sort_keys=True)}\n"
        f"- allowed_target_frontier: {json.dumps(sanitize_json_value(dict(allowed_target_frontier)), sort_keys=True)}\n"
        f"- model: {model_name}\n\n"
        "Aggregate diagnostics from prior evaluation:\n"
        f"{json.dumps(sanitize_json_value(dict(aggregate_metrics)), sort_keys=True, indent=2)}\n\n"
        "Objective:\n"
        "- Choose the valid configuration with the highest expected reward.\n"
        "- Reward is frontier_delta + lambda_final * (candidate_final_accuracy - init_final_accuracy).\n"
        "- Reusing an existing target slice is allowed; measured reward determines whether rehearsal helps.\n\n"
        "Output schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        f"{schema_notes}\n\n"
        "Constraints:\n"
        "- left and right must be source slice sizes currently available to the model.\n"
        "- left + right must land inside allowed_target_frontier.\n"
        "- Choose only a listed guard value.\n"
        "- If prediction is requested, prediction.target must equal left + right.\n"
        "- If prediction is requested, expected deltas must be finite numbers.\n"
        "- Do not choose frontier ranges, data budgets, sampling schedules, or composition paths; the driver owns those.\n"
        "- The driver composes pseudo-labels from model component predictions, not oracle labels.\n\n"
        "Output only JSON."
    )
    return PromptBundle(system=system, user=user)


def render_program_prompt(
    *,
    task_name: str,
    target_format: str,
    component_prediction_examples: Sequence[str],
) -> PromptBundle:
    system = (
        "You are generating a restricted Python composition program for a self-improvement pipeline.\n"
        "Output only Python code. Do not include markdown or explanations.\n"
        "The code will be statically checked and sandboxed.\n"
        "Imports, file access, network access, subprocesses, eval, exec, and global mutation are forbidden."
    )
    user = (
        f"Task: {task_name}\n\n"
        "Goal:\n"
        "Write a composition function that combines component predictions into one pseudo-label.\n\n"
        "Allowed function signature:\n"
        "def compose(components, metadata):\n"
        "    ...\n\n"
        "Inputs:\n"
        "- components: ordered list of dictionaries.\n"
        "- Each component has:\n"
        '  - "size": integer\n'
        '  - "input_id": string\n'
        '  - "prediction": string produced by the current model\n'
        '  - "metadata": safe component metadata\n'
        "- metadata: safe composed-example metadata.\n"
        "- You do not receive oracle labels.\n\n"
        "Output:\n"
        '- Return {"accept": True, "target": "<target string>"} if composition succeeds.\n'
        '- Return {"accept": False, "reason": "<short reason>"} if it should be skipped.\n\n'
        "Task target format:\n"
        f"{target_format}\n\n"
        "Examples of valid component predictions:\n"
        f"{json.dumps(list(component_prediction_examples), indent=2)}\n\n"
        "Constraints:\n"
        "- Use only component predictions and metadata.\n"
        "- Do not solve directly from the full input.\n"
        "- Reject malformed component predictions.\n"
        "- Be deterministic.\n"
        "- Keep code short.\n\n"
        "Output only Python code."
    )
    return PromptBundle(system=system, user=user)


def render_program_repair_prompt(
    *,
    task_name: str,
    target_format: str,
    failure_category: str,
    failure_summary: str,
    previous_program: str,
) -> PromptBundle:
    system = (
        "You are repairing a restricted Python composition program for a self-improvement pipeline.\n"
        "Output only the corrected Python code. Do not include markdown or explanations.\n"
        "The same sandbox rules still apply: no imports, file access, network access, subprocesses, eval, exec, or global mutation."
    )
    user = (
        "The previous program failed validation.\n\n"
        f"Task: {task_name}\n\n"
        "Target format:\n"
        f"{target_format}\n\n"
        "Validation failure category:\n"
        f"{failure_category}\n\n"
        "Sanitized validation summary:\n"
        f"{failure_summary}\n\n"
        "Previous program:\n"
        f"{previous_program}\n\n"
        "Repair requirements:\n"
        "- Keep the same function signature: def compose(components, metadata):\n"
        "- Use only component predictions and safe metadata.\n"
        "- Do not inspect oracle labels.\n"
        "- Do not solve directly from full inputs.\n"
        '- Return {"accept": True, "target": "<target string>"} or {"accept": False, "reason": "<short reason>"}.\n'
        "- Make the smallest correction needed to pass validation.\n\n"
        "Output only Python code."
    )
    return PromptBundle(system=system, user=user)


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

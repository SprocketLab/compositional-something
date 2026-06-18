#!/usr/bin/env python3
"""Proposal prompts, schemas, and trace helpers for adaptive composition."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

from self.core.data_io import sanitize_json_value
from self.core.proposal_config_schema import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    PROPOSAL_OUTPUT_SCHEMAS,
    ConfigProposal,
    ProposalValidation,
    extract_json_object,
    normalized_config_completion,
    parse_config_proposal,
    proposal_output_schema,
    proposal_payload_for_schema,
    validate_config_prediction,
)
from self.core.proposal_io import build_trace_row, load_fixture_proposals, write_trace_jsonl


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str

    def text(self) -> str:
        return f"System:\n{self.system}\n\nUser:\n{self.user}"


@dataclass(frozen=True)
class ProgramProposal:
    proposal_type: str
    task: str
    code: str

    def to_json_dict(self) -> JsonDict:
        return asdict(self)

    def to_completion(self) -> str:
        return self.code


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

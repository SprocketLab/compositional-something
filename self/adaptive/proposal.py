#!/usr/bin/env python3
"""Proposal schemas, prompts, generation, validation, IO, and GRPO."""

from __future__ import annotations


# --- from proposal_config_schema.py ---
"""Config proposal schemas, parsing, and normalization helpers."""

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]
PROPOSAL_OUTPUT_SCHEMAS = ("plain", "action_prediction")


@dataclass(frozen=True)
class ConfigProposal:
    left: int
    right: int
    guard: str
    target: int
    notes: str = ""

    def to_json_dict(self) -> JsonDict:
        return asdict(self)

    def to_completion(self) -> str:
        payload: JsonDict = {
            "left": self.left,
            "right": self.right,
            "guard": self.guard,
        }
        if self.notes:
            payload["notes"] = self.notes
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ProposalValidation:
    valid: bool
    category: str = ""
    message: str = ""
    proposal: Optional[Any] = None
    raw: Optional[Any] = None


DEFAULT_CONFIG_SEARCH_SPACES: Dict[str, JsonDict] = {
    "addition": {
        "guards": ["none", "reject_boundary_carry"],
    },
    "run_length": {
        "guards": ["none", "reject_boundary_continue", "require_boundary_continue"],
    },
}


def extract_json_object(raw: str) -> Optional[JsonDict]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def proposal_output_schema(args: Any) -> str:
    return str(getattr(args, "proposal_output_schema", "plain"))


def _row_payload(raw: Any) -> Optional[JsonDict]:
    if isinstance(raw, dict):
        return dict(raw)
    return extract_json_object(str(raw))


def _finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def proposal_payload_for_schema(raw: Any, schema: str) -> Tuple[Any, Optional[JsonDict], Optional[str], Optional[str]]:
    if schema == "plain":
        return raw, None, None, None
    if schema != "action_prediction":
        return raw, None, "schema_error", f"unsupported proposal output schema={schema!r}"
    payload = _row_payload(raw)
    if payload is None:
        return raw, None, "parse_error", "raw output is not a JSON object"
    proposal_payload = payload.get("proposal")
    if not isinstance(proposal_payload, Mapping):
        return payload, None, "schema_error", "action_prediction output requires object field proposal"
    prediction_payload = payload.get("prediction")
    if not isinstance(prediction_payload, Mapping):
        return proposal_payload, None, "schema_error", "action_prediction output requires object field prediction"
    return dict(proposal_payload), dict(prediction_payload), None, None


def validate_config_prediction(
    *,
    prediction_payload: Optional[Mapping[str, Any]],
    proposal: ConfigProposal,
    schema: str,
) -> Tuple[Optional[JsonDict], Optional[str]]:
    if schema == "plain":
        return None, None
    if prediction_payload is None:
        return None, "missing prediction"
    try:
        target = int(prediction_payload["target"])
    except (KeyError, TypeError, ValueError):
        return None, "prediction.target must be an integer equal to left + right"
    if target != int(proposal.target):
        return None, "prediction.target must equal left + right"
    expected_frontier_delta = _finite_float(prediction_payload.get("expected_frontier_delta"))
    if expected_frontier_delta is None:
        return None, "prediction.expected_frontier_delta must be a finite number"
    expected_final_delta_from_init = _finite_float(prediction_payload.get("expected_final_delta_from_init"))
    if expected_final_delta_from_init is None:
        return None, "prediction.expected_final_delta_from_init must be a finite number"
    return (
        {
            "target": target,
            "expected_frontier_delta": expected_frontier_delta,
            "expected_final_delta_from_init": expected_final_delta_from_init,
            "rationale": str(prediction_payload.get("rationale", ""))[:240],
        },
        None,
    )


def proposal_payload_for_completion(proposal: ConfigProposal) -> JsonDict:
    payload: JsonDict = {
        "left": int(proposal.left),
        "right": int(proposal.right),
        "guard": str(proposal.guard),
    }
    if proposal.notes:
        payload["notes"] = proposal.notes
    return payload


def normalized_config_completion(
    *,
    proposal: ConfigProposal,
    prediction: Optional[Mapping[str, Any]],
    schema: str,
) -> str:
    if schema == "plain":
        return proposal.to_completion()
    if schema != "action_prediction":
        raise ValueError(f"Unsupported proposal output schema={schema!r}.")
    payload = {
        "proposal": proposal_payload_for_completion(proposal),
        "prediction": dict(prediction or {}),
    }
    return json.dumps(sanitize_json_value(payload), sort_keys=True, separators=(",", ":"))


def parse_config_proposal(
    raw: Any,
    *,
    task_name: str,
    source_min_allowed: int,
    source_max_allowed: int,
    source_sizes_allowed: Optional[Sequence[int]] = None,
    frontier_min_allowed: int,
    frontier_max_allowed: int,
    guards: Sequence[str],
) -> ProposalValidation:
    payload = raw if isinstance(raw, dict) else extract_json_object(str(raw))
    if payload is None:
        return ProposalValidation(False, "parse_error", "raw output is not a JSON object", raw=raw)

    forbidden_driver_fields = {
        "proposal_type",
        "task",
        "frontier_min",
        "frontier_max",
        "composition_path",
        "compose_arity",
        "filter_rule",
        "examples_per_size",
        "composition_error_percent",
    }
    present_forbidden = sorted(field for field in forbidden_driver_fields if field in payload)
    if present_forbidden:
        return ProposalValidation(
            False,
            "schema_error",
            "config may only choose left/right/guard; driver-owned fields are forbidden: "
            + ", ".join(present_forbidden),
            raw=raw,
        )

    required = ["left", "right", "guard"]
    missing = [field for field in required if field not in payload]
    if missing:
        return ProposalValidation(False, "schema_error", f"missing fields: {', '.join(missing)}", raw=raw)

    try:
        left = int(payload["left"])
        right = int(payload["right"])
    except (TypeError, ValueError):
        return ProposalValidation(False, "schema_error", "numeric fields have invalid types", raw=raw)

    if source_min_allowed > source_max_allowed:
        return ProposalValidation(False, "range_error", "source slice bounds are invalid", raw=raw)
    if frontier_min_allowed > frontier_max_allowed:
        return ProposalValidation(False, "range_error", "target frontier bounds are invalid", raw=raw)

    if left < source_min_allowed or left > source_max_allowed:
        return ProposalValidation(False, "range_error", "left source slice is outside the allowed bounds", raw=raw)
    if right < source_min_allowed or right > source_max_allowed:
        return ProposalValidation(False, "range_error", "right source slice is outside the allowed bounds", raw=raw)
    if source_sizes_allowed is not None:
        source_size_set = {int(size) for size in source_sizes_allowed}
        if left not in source_size_set:
            return ProposalValidation(False, "range_error", "left source slice is not in the current source pool", raw=raw)
        if right not in source_size_set:
            return ProposalValidation(False, "range_error", "right source slice is not in the current source pool", raw=raw)
    target = left + right
    if target < frontier_min_allowed or target > frontier_max_allowed:
        return ProposalValidation(False, "range_error", "left + right target is outside the allowed frontier", raw=raw)

    guard = str(payload["guard"])
    if guard not in set(guards):
        return ProposalValidation(False, "enum_error", f"invalid guard={guard!r}", raw=raw)

    proposal = ConfigProposal(
        left=left,
        right=right,
        guard=guard,
        target=target,
        notes=str(payload.get("notes", "")),
    )
    return ProposalValidation(True, proposal=proposal, raw=raw)


# --- from proposal_config_validation.py ---
"""Config proposal row validation."""

import argparse
from typing import Any, Dict, List, Mapping, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    normalized_config_completion,
    parse_config_proposal,
    proposal_output_schema,
    proposal_payload_for_schema,
    validate_config_prediction,
)

JsonDict = Dict[str, Any]


def _raw_output(row: Mapping[str, Any]) -> Any:
    if "code_lines" in row:
        code_lines = row["code_lines"]
        if isinstance(code_lines, list):
            return "\n".join(str(line) for line in code_lines)
    for key in ("raw_output", "output", "completion", "proposal", "code"):
        if key in row:
            return row[key]
    return row


def validate_config_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
) -> List[JsonDict]:
    source_min = min(source_sizes) if source_sizes else args.initial_min_size
    source_max = max(source_sizes) if source_sizes else args.initial_max_size
    guards = DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"]
    schema = proposal_output_schema(args)
    results: List[JsonDict] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        raw = _raw_output(row)
        proposal_raw, prediction_payload, pre_category, pre_message = proposal_payload_for_schema(
            raw,
            schema,
        )
        if pre_category is not None:
            completion = ""
            validation_valid = False
            category = pre_category
            message = str(pre_message)
            proposal_payload = None
            parsed_prediction = None
            duplicate = False
            repeat_target = False
            results.append(
                sanitize_json_value(
                    {
                        "proposal_index": index,
                        "id": row.get("id"),
                        "raw_output": raw,
                        "valid": validation_valid,
                        "validation_category": category,
                        "validation_message": message,
                        "parsed_proposal": proposal_payload,
                        "parsed_prediction": parsed_prediction,
                        "proposal_output_schema": schema,
                        "completion": completion,
                        "duplicate": duplicate,
                        "repeat_target": repeat_target,
                    }
                )
            )
            continue
        validation = parse_config_proposal(
            proposal_raw,
            task_name=args.task,
            source_min_allowed=source_min,
            source_max_allowed=source_max,
            source_sizes_allowed=sorted(source_sizes),
            frontier_min_allowed=frontier_min,
            frontier_max_allowed=frontier_max,
            guards=guards,
        )
        parsed_prediction = None
        prediction_error = None
        if validation.valid:
            parsed_prediction, prediction_error = validate_config_prediction(
                prediction_payload=prediction_payload,
                proposal=validation.proposal,
                schema=schema,
            )
        validation_valid = bool(validation.valid and prediction_error is None)
        category = validation.category
        message = validation.message
        if validation.valid and prediction_error is not None:
            category = "schema_error"
            message = prediction_error
        completion = (
            normalized_config_completion(
                proposal=validation.proposal,
                prediction=parsed_prediction,
                schema=schema,
            )
            if validation_valid
            else ""
        )
        duplicate = bool(completion and completion in seen)
        if completion:
            seen.add(completion)
        repeat_target = bool(validation_valid and validation.proposal.target in source_sizes)
        proposal_payload = validation.proposal.to_json_dict() if validation.valid else None
        results.append(
            sanitize_json_value(
                {
                    "proposal_index": index,
                    "id": row.get("id"),
                    "raw_output": raw,
                    "valid": validation_valid,
                    "validation_category": category,
                    "validation_message": message,
                    "parsed_proposal": proposal_payload,
                    "parsed_prediction": parsed_prediction,
                    "proposal_output_schema": schema,
                    "completion": completion,
                    "duplicate": duplicate,
                    "repeat_target": repeat_target,
                }
            )
        )
    return results


# --- from proposal_prompts.py ---
"""Proposal prompt rendering and task-specific prompt metadata."""

import argparse
from typing import List

from self.adaptive.program_sandbox import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.adaptive.program_sandbox import SandboxCase
from self.adaptive.proposal import ConfigProposal

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
from self.adaptive.proposal import ConfigProposal


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


# --- from proposal_io.py ---
"""Proposal fixture and trace JSONL IO helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]


def load_fixture_proposals(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def build_trace_row(
    *,
    round_index: int,
    task: str,
    condition: str,
    reward: float,
    frontier_delta: float,
    final_accuracy: float,
    prompt: str,
    completion: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    row: JsonDict = {
        "round": round_index,
        "task": task,
        "condition": condition,
        "reward": reward,
        "frontier_delta": frontier_delta,
        "final_accuracy": final_accuracy,
        "prompt": prompt,
        "completion": completion,
    }
    if extra_metadata:
        row["metadata"] = dict(extra_metadata)
    return sanitize_json_value(row)


def write_trace_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(sanitize_json_value(dict(row)), handle, sort_keys=True)
            handle.write("\n")


# --- from proposal_generation.py ---
"""Proposal-row loading and model generation for adaptive self-improvement."""

import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.adaptive.proposal import load_fixture_proposals
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


def _rows_for_round(
    rows: Sequence[Mapping[str, Any]],
    round_index: int,
    *,
    attempt_index: Optional[int] = None,
) -> List[Mapping[str, Any]]:
    if attempt_index is not None:
        attempt_matching = [
            row for row in rows if "attempt" in row and int(row.get("attempt", -1)) == attempt_index
        ]
        if attempt_matching:
            return attempt_matching
    matching = [row for row in rows if int(row.get("round", round_index)) == round_index]
    return matching if matching else list(rows)


def generate_proposals_from_model(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: PromptBundle,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[JsonDict]:
    import torch
    from self.core.evaluation import build_generation_encodings

    device = next(model.parameters()).device
    encodings = build_generation_encodings(tokenizer, [prompt.text()], device)
    rows: List[JsonDict] = []
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for idx in range(num_candidates):
            generation_kwargs: Dict[str, Any] = {
                **encodings,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
            prompt_width = encodings["input_ids"].shape[1]
            decoded = tokenizer.decode(output_ids[0, prompt_width:].tolist(), skip_special_tokens=True)
            rows.append({"id": f"model_candidate_{idx}", "raw_output": decoded})
    if model_was_training:
        model.train()
    return rows


def load_or_generate_proposal_rows(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    current_model: AutoModelForCausalLM,
    current_tokenizer: AutoTokenizer,
    round_index: int,
    attempt_index: Optional[int] = None,
) -> List[JsonDict]:
    if args.proposal_fixture_jsonl is not None:
        rows = _rows_for_round(
            load_fixture_proposals(args.proposal_fixture_jsonl),
            round_index,
            attempt_index=attempt_index,
        )
        return [dict(row) for row in rows[: args.num_candidates]]

    if args.proposal_model_name == "current":
        return generate_proposals_from_model(
            model=current_model,
            tokenizer=current_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )

    from self.core.model_io import instantiate_model_and_tokenizer

    proposal_model, proposal_tokenizer = instantiate_model_and_tokenizer(
        args.proposal_model_name,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe="none",
    )
    try:
        return generate_proposals_from_model(
            model=proposal_model,
            tokenizer=proposal_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
        )
    finally:
        del proposal_model
        del proposal_tokenizer
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- from proposal_executable_validation.py ---
"""Executable proposal validation and repair for adaptive self-improvement."""

import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.core.models import ExecutableProposal
from self.adaptive.program_sandbox import validate_program_with_repair
from self.adaptive.program_sandbox import ProgramValidationResult
from self.adaptive.proposal import _raw_output
from self.adaptive.proposal import (
    DEFAULT_CONFIG_SEARCH_SPACES,
    ConfigProposal,
    extract_json_object,
)
from self.adaptive.proposal import generate_proposals_from_model
from self.adaptive.proposal import (
    program_validation_cases,
    target_format_for_task,
)
from self.adaptive.proposal import render_program_repair_prompt


JsonDict = Dict[str, Any]


def _extract_python_code(raw: Any, payload: Optional[Mapping[str, Any]] = None) -> str:
    if payload is not None:
        if "code_lines" in payload and isinstance(payload["code_lines"], list):
            return "\n".join(str(line) for line in payload["code_lines"]).strip()
        for key in ("code", "completion", "program"):
            if key in payload:
                return str(payload[key]).strip()
    text = str(raw).strip()
    if "```" in text:
        pieces = text.split("```")
        for piece in pieces:
            stripped = piece.strip()
            if stripped.startswith("python"):
                stripped = stripped[len("python") :].strip()
            if "def compose" in stripped:
                text = stripped
                break
    start = text.find("def compose")
    if start > 0:
        text = text[start:]
    return text.strip()


def _row_payload(raw: Any) -> Optional[JsonDict]:
    if isinstance(raw, dict):
        return dict(raw)
    return extract_json_object(str(raw))


def _row_repair_output(row: Mapping[str, Any]) -> Optional[str]:
    if "repair_output_lines" in row:
        repair_lines = row["repair_output_lines"]
        if not isinstance(repair_lines, list):
            raise ValueError("repair_output_lines must be a list of strings")
        return "\n".join(str(line) for line in repair_lines)
    for key in ("repair_output", "repaired_output", "repair_code"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _repair_program_with_model(
    *,
    args: argparse.Namespace,
    current_model: Optional[Any],
    current_tokenizer: Optional[Any],
    category: str,
    message: str,
    code: str,
) -> Optional[str]:
    if current_model is None or current_tokenizer is None:
        return None
    repair_prompt = render_program_repair_prompt(
        task_name=args.task,
        target_format=target_format_for_task(args.task, args),
        failure_category=category,
        failure_summary=message,
        previous_program=code,
    )
    rows = generate_proposals_from_model(
        model=current_model,
        tokenizer=current_tokenizer,
        prompt=repair_prompt,
        num_candidates=1,
        max_new_tokens=args.proposal_max_new_tokens,
        temperature=args.proposal_temperature,
        top_p=args.proposal_top_p,
    )
    if not rows:
        return None
    return _extract_python_code(rows[0].get("raw_output", ""))


def validate_executable_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    default_pair: Optional[ConfigProposal],
    current_model: Optional[Any] = None,
    current_tokenizer: Optional[Any] = None,
) -> List[JsonDict]:
    source_min = min(source_sizes) if source_sizes else args.initial_min_size
    source_max = max(source_sizes) if source_sizes else args.initial_max_size
    guards = set(DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"])
    results: List[JsonDict] = []
    seen: set[str] = set()
    cases = program_validation_cases(args.task, args)

    for index, row in enumerate(rows):
        raw = _raw_output(row)
        payload = _row_payload(raw)
        condition = str(row.get("condition", args.condition))
        row_raw_for_code: Any = raw
        if payload is None and isinstance(row, Mapping):
            payload = dict(row)
        if condition != args.condition:
            continue

        if args.condition == "program":
            if default_pair is None:
                validation_valid = False
                category = "range_error"
                message = "program condition has no driver-selected source pair"
                proposal_payload = None
                completion = ""
                duplicate = False
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": validation_valid,
                            "validation_category": category,
                            "validation_message": message,
                            "parsed_proposal": proposal_payload,
                            "completion": completion,
                            "duplicate": duplicate,
                        }
                    )
                )
                continue
            left = default_pair.left
            right = default_pair.right
            target = default_pair.target
            guard = "none"
        else:
            if payload is None:
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": False,
                            "validation_category": "parse_error",
                            "validation_message": f"{args.condition} proposal must be a JSON object",
                            "parsed_proposal": None,
                            "completion": "",
                            "duplicate": False,
                        }
                    )
                )
                continue
            try:
                left = int(payload["left"])
                right = int(payload["right"])
            except (KeyError, TypeError, ValueError):
                results.append(
                    sanitize_json_value(
                        {
                            "proposal_index": index,
                            "id": row.get("id"),
                            "raw_output": raw,
                            "valid": False,
                            "validation_category": "schema_error",
                            "validation_message": f"{args.condition} proposal requires integer left and right",
                            "parsed_proposal": None,
                            "completion": "",
                            "duplicate": False,
                        }
                    )
                )
                continue
            target = left + right
            guard = str(payload.get("guard", "none"))
            row_raw_for_code = payload

        range_error = ""
        if left < source_min or left > source_max or right < source_min or right > source_max:
            range_error = "source slice is outside allowed source bounds"
        elif left not in source_sizes or right not in source_sizes:
            range_error = "source slice is not in the current source pool"
        elif target < frontier_min or target > frontier_max:
            range_error = "left + right target is outside the allowed frontier"
        elif not args.allow_repeat_targets and target in source_sizes:
            range_error = "target slice is already in the current source pool"
        elif guard not in guards:
            range_error = f"invalid guard={guard!r}"

        code = _extract_python_code(row_raw_for_code, payload if isinstance(row_raw_for_code, Mapping) else None)
        if range_error:
            results.append(
                sanitize_json_value(
                    {
                        "proposal_index": index,
                        "id": row.get("id"),
                        "raw_output": raw,
                        "valid": False,
                        "validation_category": "range_error" if "guard" not in range_error else "enum_error",
                        "validation_message": range_error,
                        "parsed_proposal": None,
                        "completion": "",
                        "duplicate": False,
                    }
                )
            )
            continue

        repair_prompt_text: Optional[str] = None

        def repair_callback(category: str, message: str, previous_program: str) -> Optional[str]:
            nonlocal repair_prompt_text
            repair_prompt = render_program_repair_prompt(
                task_name=args.task,
                target_format=target_format_for_task(args.task, args),
                failure_category=category,
                failure_summary=message,
                previous_program=previous_program,
            )
            repair_prompt_text = repair_prompt.text()
            fixture_repair = _row_repair_output(row)
            if fixture_repair is not None:
                return _extract_python_code(fixture_repair)
            return _repair_program_with_model(
                args=args,
                current_model=current_model,
                current_tokenizer=current_tokenizer,
                category=category,
                message=message,
                code=previous_program,
            )

        validation: ProgramValidationResult = validate_program_with_repair(
            code,
            repair_callback=repair_callback,
            cases=cases,
            timeout_seconds=args.program_timeout_seconds,
            repair_attempts=args.repair_attempts,
        )
        valid_code = validation.repaired_code if validation.valid and validation.repaired_code else code
        completion = valid_code if validation.valid else ""
        duplicate = bool(completion and completion in seen)
        if completion:
            seen.add(completion)
        proposal = None
        if validation.valid:
            proposal = ExecutableProposal(
                left=left,
                right=right,
                guard=guard,
                target=target,
                code=valid_code,
                condition=args.condition,
                notes=str((payload or {}).get("notes", "")),
                representation=str((payload or {}).get("representation", "")),
                target_format=str((payload or {}).get("target_format", target_format_for_task(args.task, args))),
                repaired=validation.repaired,
                original_validation_category=validation.original_category,
                original_validation_message=validation.original_message,
            )
        results.append(
            sanitize_json_value(
                {
                    "proposal_index": index,
                    "id": row.get("id"),
                    "raw_output": raw,
                    "valid": validation.valid,
                    "validation_category": validation.category,
                    "validation_message": validation.message,
                    "original_validation_category": validation.original_category,
                    "original_validation_message": validation.original_message,
                    "repair_attempted": validation.repair_attempted,
                    "repair_prompt": repair_prompt_text,
                    "repaired": validation.repaired,
                    "repaired_output": validation.repaired_code,
                    "parsed_proposal": proposal.to_json_dict() if proposal is not None else None,
                    "completion": completion,
                    "duplicate": duplicate,
                }
            )
        )
    return results


# --- from proposal_runtime.py ---
"""Runtime proposal generation and validation for adaptive self-improvement."""

import argparse
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.adaptive.proposal import (
    _extract_python_code,
    _repair_program_with_model,
    _row_payload,
    _row_repair_output,
    validate_executable_rows,
)
from self.adaptive.proposal import (
    _rows_for_round,
    generate_proposals_from_model,
    load_or_generate_proposal_rows,
)
from self.adaptive.proposal import _raw_output, validate_config_rows
from self.adaptive.proposal import ConfigProposal
from self.adaptive.proposal import (
    choose_default_program_pair,
    component_prediction_examples_for_task,
    program_validation_cases,
    target_format_for_task,
)
from self.adaptive.proposal import (
    render_program_candidate_prompt,
    render_program_repair_prompt,
)

JsonDict = Dict[str, Any]


def validate_proposal_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    default_pair: Optional[ConfigProposal],
    current_model: Optional[Any] = None,
    current_tokenizer: Optional[Any] = None,
) -> List[JsonDict]:
    if args.condition == "config":
        return validate_config_rows(
            rows=rows,
            args=args,
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
        )
    return validate_executable_rows(
        rows=rows,
        args=args,
        source_sizes=source_sizes,
        frontier_min=frontier_min,
        frontier_max=frontier_max,
        default_pair=default_pair,
        current_model=current_model,
        current_tokenizer=current_tokenizer,
    )


# --- from proposal_grpo_traces.py ---
"""Proposal-GRPO reward shaping and trace construction."""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]

PROPOSAL_GRPO_ZERO_VARIANCE_MODES = ("fixed_baseline", "skip")
PROPOSAL_GRPO_REWARD_MODES = ("outcome", "validity")
PROPOSAL_GRPO_REWARD_BY_CATEGORY: Dict[str, float] = {
    "valid": 1.0,
    "range_error": 0.6,
    "enum_error": 0.5,
    "schema_error": 0.25,
    "parse_error": 0.0,
}
PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY: Dict[str, float] = {
    "range_error": -0.4,
    "enum_error": -0.5,
    "schema_error": -0.7,
    "parse_error": -1.0,
}


@dataclass(frozen=True)
class ProposalGRPOTrace:
    """Raw proposal completion and shaped reward used for one GRPO update."""

    proposal_index: int
    proposal_id: Optional[str]
    prompt_text: str
    completion: str
    reward: float
    advantage: float
    validation_category: str
    validation_message: str
    valid: bool
    metadata: JsonDict

    def to_json_dict(self) -> JsonDict:
        return sanitize_json_value(
            {
                "proposal_index": self.proposal_index,
                "proposal_id": self.proposal_id,
                "prompt": self.prompt_text,
                "completion": self.completion,
                "reward": self.reward,
                "advantage": self.advantage,
                "validation_category": self.validation_category,
                "validation_message": self.validation_message,
                "valid": self.valid,
                "metadata": self.metadata,
            }
        )


def proposal_grpo_reward(result: Mapping[str, Any]) -> float:
    if bool(result.get("valid")):
        return PROPOSAL_GRPO_REWARD_BY_CATEGORY["valid"]
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_REWARD_BY_CATEGORY.get(category, 0.0))


def proposal_grpo_outcome_invalid_reward(result: Mapping[str, Any]) -> float:
    category = str(result.get("validation_category") or "parse_error")
    return float(PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY.get(category, -1.0))


def _proposal_index(result: Mapping[str, Any], default: int) -> int:
    try:
        return int(result.get("proposal_index", default))
    except (TypeError, ValueError):
        return default


def _is_system_candidate_failure(metric: Any) -> bool:
    if metric.valid:
        return False
    reason = str(metric.failure_reason or "").lower()
    if reason == "no pseudo labels retained":
        return False
    return bool(reason)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def proposal_grpo_reward_for_result(
    result: Mapping[str, Any],
    *,
    metric: Optional[Any],
    reward_mode: str,
    outcome_scale: float,
) -> Tuple[Optional[float], str]:
    if reward_mode == "validity":
        return proposal_grpo_reward(result), "validity"
    if reward_mode != "outcome":
        raise ValueError(f"Unsupported proposal_grpo_reward_mode={reward_mode!r}.")
    if not bool(result.get("valid")):
        return proposal_grpo_outcome_invalid_reward(result), "invalid"
    if metric is None:
        return 0.0, "valid_untrained"
    if not metric.valid:
        if _is_system_candidate_failure(metric):
            return None, "skipped_system_failure"
        return 0.0, "valid_untrained"
    return _clamp(float(metric.reward) / float(outcome_scale), -1.0, 1.0), "outcome"


def proposal_grpo_advantages(
    rewards: Sequence[float],
    *,
    zero_variance: str,
    fixed_baseline: float,
    eps: float = 1e-6,
) -> Tuple[List[float], bool, str]:
    if not rewards:
        return [], True, "no_rewards"
    reward_values = [float(reward) for reward in rewards]
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((reward - mean_reward) ** 2 for reward in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)
    if std_reward > eps:
        return [(reward - mean_reward) / (std_reward + eps) for reward in reward_values], False, "normalized"
    if zero_variance == "skip":
        return [0.0 for _ in reward_values], True, "zero_variance"
    if zero_variance != "fixed_baseline":
        raise ValueError(f"Unsupported proposal_grpo_zero_variance={zero_variance!r}.")
    return [reward - fixed_baseline for reward in reward_values], False, "fixed_baseline"


def build_proposal_grpo_traces(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any] = (),
) -> Tuple[List[ProposalGRPOTrace], JsonDict]:
    prompt_text = prompt.text()
    metrics_by_index = {int(metric.index): metric for metric in candidate_metrics}
    rewards: List[float] = []
    included_results: List[Tuple[int, Mapping[str, Any], Optional[Any], float, str]] = []
    reward_source_counts: Dict[str, int] = {}
    skipped_system_failure_count = 0
    for index, result in enumerate(proposal_results):
        proposal_index = _proposal_index(result, index)
        metric = metrics_by_index.get(proposal_index)
        reward, reward_source = proposal_grpo_reward_for_result(
            result,
            metric=metric,
            reward_mode=args.proposal_grpo_reward_mode,
            outcome_scale=args.proposal_grpo_outcome_scale,
        )
        reward_source_counts[reward_source] = reward_source_counts.get(reward_source, 0) + 1
        if reward is None:
            skipped_system_failure_count += 1
            continue
        rewards.append(float(reward))
        included_results.append((proposal_index, result, metric, float(reward), reward_source))
    advantages, skip_update, advantage_mode = proposal_grpo_advantages(
        rewards,
        zero_variance=args.proposal_grpo_zero_variance,
        fixed_baseline=args.proposal_grpo_fixed_baseline,
    )
    traces: List[ProposalGRPOTrace] = []
    for proposal_index, result, metric, reward, reward_source, advantage in (
        (*payload, advantage)
        for payload, advantage in zip(included_results, advantages)
    ):
        raw = result.get("raw_output", "")
        normalized_completion = result.get("completion")
        if result.get("valid") and isinstance(normalized_completion, str) and normalized_completion:
            completion = normalized_completion
            completion_source = "normalized"
        else:
            completion = raw if isinstance(raw, str) else json.dumps(sanitize_json_value(raw), sort_keys=True)
            completion_source = "raw"
        traces.append(
            ProposalGRPOTrace(
                proposal_index=proposal_index,
                proposal_id=str(result["id"]) if result.get("id") is not None else None,
                prompt_text=prompt_text,
                completion=completion,
                reward=float(reward),
                advantage=float(advantage),
                validation_category=str(
                    result.get("validation_category") or ("valid" if result.get("valid") else "unknown")
                ),
                validation_message=str(result.get("validation_message") or ""),
                valid=bool(result.get("valid")),
                metadata={
                    "duplicate": bool(result.get("duplicate")),
                    "repeat_target": bool(result.get("repeat_target")),
                    "parsed_proposal": result.get("parsed_proposal"),
                    "parsed_prediction": result.get("parsed_prediction"),
                    "proposal_output_schema": result.get("proposal_output_schema"),
                    "completion_source": completion_source,
                    "reward_source": reward_source,
                    "candidate_reward": metric.reward if metric is not None else None,
                    "frontier_delta": metric.frontier_delta if metric is not None else None,
                    "target_delta": metric.target_delta if metric is not None else None,
                    "proposal_prediction": metric.proposal_prediction if metric is not None else None,
                },
            )
        )
    mean_reward = sum(rewards) / len(rewards) if rewards else math.nan
    reward_std = (
        math.sqrt(sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards))
        if rewards
        else math.nan
    )
    return traces, {
        "reward_mean": mean_reward,
        "reward_std": reward_std,
        "advantage_mode": advantage_mode,
        "zero_variance_skip": bool(skip_update),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "reward_source_counts": reward_source_counts,
        "skipped_system_failure_count": skipped_system_failure_count,
        "input_proposal_count": len(proposal_results),
        "trace_candidate_metric_count": len(candidate_metrics),
    }


# --- from proposal_grpo.py ---
"""Proposal-GRPO lightweight policy updates."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import (
    PROPOSAL_GRPO_OUTCOME_INVALID_REWARD_BY_CATEGORY,
    PROPOSAL_GRPO_REWARD_BY_CATEGORY,
    PROPOSAL_GRPO_REWARD_MODES,
    PROPOSAL_GRPO_ZERO_VARIANCE_MODES,
    ProposalGRPOTrace,
    build_proposal_grpo_traces,
    proposal_grpo_advantages,
    proposal_grpo_outcome_invalid_reward,
    proposal_grpo_reward,
    proposal_grpo_reward_for_result,
)
from self.adaptive.proposal import write_trace_jsonl
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


def _encode_proposal_grpo_sample(
    *,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    completion: str,
) -> Optional[JsonDict]:
    if completion == "":
        return None
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        return None
    input_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(int(tokenizer.bos_token_id))
    input_ids.extend(int(token_id) for token_id in prompt_ids)
    completion_start = len(input_ids)
    input_ids.extend(int(token_id) for token_id in completion_ids)
    if len(input_ids) < 2:
        return None
    completion_mask = [bool(position + 1 >= completion_start) for position in range(len(input_ids) - 1)]
    if not any(completion_mask):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "completion_mask": completion_mask,
        "completion_tokens": len(completion_ids),
    }


def _collate_proposal_grpo_samples(
    *,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    device: torch.device,
) -> JsonDict:
    import torch

    if not samples:
        raise ValueError("Expected at least one proposal GRPO sample.")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for proposal GRPO padding.")
    max_length = max(len(sample["input_ids"]) for sample in samples)
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    completion_mask: List[List[bool]] = []
    for sample in samples:
        sample_ids = list(sample["input_ids"])
        pad_count = max_length - len(sample_ids)
        input_ids.append(sample_ids + [int(pad_token_id)] * pad_count)
        attention_mask.append(list(sample["attention_mask"]) + [0] * pad_count)
        sample_completion_mask = list(sample["completion_mask"])
        completion_mask.append(sample_completion_mask + [False] * pad_count)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "completion_mask": torch.tensor(completion_mask, dtype=torch.bool, device=device),
    }


def _proposal_completion_mean_logprobs(model: AutoModelForCausalLM, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    import torch

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["input_ids"][:, 1:]
    mask = batch["completion_mask"][:, : labels.shape[1]]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    masked_logprobs = token_logprobs * mask.float()
    token_counts = mask.sum(dim=1).clamp_min(1).float()
    return masked_logprobs.sum(dim=1) / token_counts


def apply_proposal_grpo_update(
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any],
    seed: int,
) -> Tuple[str, JsonDict]:
    import torch
    from transformers import set_seed
    from self.core.model_io import instantiate_model_and_tokenizer

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: JsonDict = {
        "enabled": bool(args.proposal_grpo_steps > 0),
        "skipped": True,
        "skip_reason": None,
        "source_checkpoint": source_checkpoint,
        "model_dir": None,
        "proposal_count": len(proposal_results),
        "steps": int(args.proposal_grpo_steps),
        "learning_rate": float(args.proposal_grpo_learning_rate),
        "kl_coef": float(args.proposal_grpo_kl_coef),
        "grad_clip": float(args.proposal_grpo_grad_clip),
        "zero_variance": args.proposal_grpo_zero_variance,
        "fixed_baseline": float(args.proposal_grpo_fixed_baseline),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "candidate_metric_count": len(candidate_metrics),
    }
    if args.proposal_grpo_steps <= 0:
        metrics["skip_reason"] = "disabled"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.condition != "config":
        metrics["skip_reason"] = "non_config_condition"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.proposal_model_name != "current":
        metrics["skip_reason"] = "off_policy_proposal_model"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if args.proposal_fixture_jsonl is not None:
        metrics["skip_reason"] = "off_policy_fixture"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics
    if not proposal_results:
        metrics["skip_reason"] = "no_proposals"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics

    traces, trace_summary = build_proposal_grpo_traces(
        args=args,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
    )
    write_trace_jsonl(output_dir / "proposal_grpo_traces.jsonl", [trace.to_json_dict() for trace in traces])
    metrics.update(trace_summary)
    if trace_summary["zero_variance_skip"]:
        metrics["skip_reason"] = "zero_variance"
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return source_checkpoint, metrics

    set_seed(seed)
    model, tokenizer = instantiate_model_and_tokenizer(
        source_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        device = next(model.parameters()).device
        encoded_samples: List[JsonDict] = []
        encoded_traces: List[ProposalGRPOTrace] = []
        for trace in traces:
            sample = _encode_proposal_grpo_sample(
                tokenizer=tokenizer,
                prompt_text=trace.prompt_text,
                completion=trace.completion,
            )
            if sample is None:
                continue
            encoded_samples.append(sample)
            encoded_traces.append(trace)
        metrics["trace_count"] = len(traces)
        metrics["trainable_trace_count"] = len(encoded_samples)
        metrics["completion_token_counts"] = [int(sample["completion_tokens"]) for sample in encoded_samples]
        if not encoded_samples:
            metrics["skip_reason"] = "no_tokenizable_completions"
            write_json(output_dir / "proposal_grpo_metrics.json", metrics)
            return source_checkpoint, metrics

        batch = _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=encoded_samples, device=device)
        advantages = torch.tensor(
            [trace.advantage for trace in encoded_traces],
            dtype=torch.float32,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            old_logprobs = _proposal_completion_mean_logprobs(model, batch).detach()

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.proposal_grpo_learning_rate))
        loss_history: List[JsonDict] = []
        for step_index in range(int(args.proposal_grpo_steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            new_logprobs = _proposal_completion_mean_logprobs(model, batch)
            policy_loss = -(advantages * new_logprobs).mean()
            kl_proxy = ((new_logprobs - old_logprobs) ** 2).mean()
            loss = policy_loss + float(args.proposal_grpo_kl_coef) * kl_proxy
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.proposal_grpo_grad_clip))
            optimizer.step()
            loss_history.append(
                sanitize_json_value(
                    {
                        "step": step_index + 1,
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "kl_proxy": float(kl_proxy.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                        "mean_logprob_before": float(old_logprobs.mean().detach().cpu()),
                        "mean_logprob_after": float(new_logprobs.mean().detach().cpu()),
                    }
                )
            )

        model_dir = output_dir / "model"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        metrics.update(
            {
                "skipped": False,
                "skip_reason": None,
                "model_dir": str(model_dir),
                "loss_history": loss_history,
                "reward_values": [trace.reward for trace in traces],
                "advantages": [trace.advantage for trace in traces],
            }
        )
        write_json(output_dir / "proposal_grpo_metrics.json", metrics)
        return str(model_dir), metrics
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


# --- from proposal_grpo_dispatch.py ---
"""Proposal-GRPO update dispatch helpers."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.adaptive.phases import PHASE_PROPOSAL_GRPO
from self.core.models import CandidateMetrics
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ProposalGrpoDispatchDeps:
    apply_proposal_grpo_update: Callable[..., tuple[str, JsonDict]]
    run_controller_worker_slurm: Callable[..., JsonDict]
    ensure_dir: Callable[[Path], None]
    write_json: Callable[[Path, Any], None]


def apply_or_dispatch_proposal_grpo_update(
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[CandidateMetrics],
    seed: int,
    deps: ProposalGrpoDispatchDeps,
) -> tuple[str, JsonDict]:
    if args.controller_execution_mode != "slurm":
        return deps.apply_proposal_grpo_update(
            args=args,
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            prompt=prompt,
            proposal_results=proposal_results,
            candidate_metrics=candidate_metrics,
            seed=seed,
        )
    deps.ensure_dir(output_dir)
    prompt_path = output_dir / "proposal_prompt.json"
    proposal_results_path = output_dir / "proposal_results.json"
    candidate_metrics_path = output_dir / "candidate_metrics.json"
    deps.write_json(prompt_path, {"system": prompt.system, "user": prompt.user})
    deps.write_json(proposal_results_path, proposal_results)
    deps.write_json(candidate_metrics_path, [metric.to_json_dict() for metric in candidate_metrics])
    worker_output = deps.run_controller_worker_slurm(
        args=args,
        worker_dir=output_dir / "controller_worker",
        phase=PHASE_PROPOSAL_GRPO,
        payload={
            "source_checkpoint": source_checkpoint,
            "proposal_grpo_dir": str(output_dir),
            "prompt_path": str(prompt_path),
            "proposal_results_path": str(proposal_results_path),
            "candidate_metrics_path": str(candidate_metrics_path),
            "seed": seed,
        },
    )
    return str(worker_output["next_checkpoint"]), dict(worker_output["proposal_grpo_metrics"])


# --- from proposal_pilot_runtime.py ---
"""Dry-run proposal processing for the adaptive proposal pilot."""

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.proposal import DEFAULT_CONFIG_SEARCH_SPACES, parse_config_proposal
from self.adaptive.proposal import build_trace_row
from self.adaptive.proposal import (
    PromptBundle,
    render_config_prompt,
    render_program_prompt,
    render_program_repair_prompt,
)
from self.adaptive.program_sandbox import validate_program_with_repair
from self.adaptive.program_sandbox import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.adaptive.program_sandbox import ProgramValidationResult


JsonDict = Dict[str, Any]


def pilot_reward(
    row: Mapping[str, Any],
    *,
    lambda_final: float,
    init_final_accuracy: Optional[float],
) -> JsonDict:
    frontier_delta = float(row.get("frontier_delta", 0.0))
    final_accuracy = float(row.get("final_accuracy", 0.0))
    row_init_final_accuracy = row.get("init_final_accuracy", init_final_accuracy)
    init_final_accuracy_value = (
        None if row_init_final_accuracy is None else float(row_init_final_accuracy)
    )
    final_accuracy_delta = float(
        row.get(
            "final_accuracy_delta",
            final_accuracy - (init_final_accuracy_value if init_final_accuracy_value is not None else 0.0),
        )
    )
    if "reward" in row:
        reward = float(row["reward"])
    else:
        reward = frontier_delta + lambda_final * final_accuracy_delta
    return {
        "frontier_delta": frontier_delta,
        "final_accuracy": final_accuracy,
        "init_final_accuracy": init_final_accuracy_value,
        "final_accuracy_delta": final_accuracy_delta,
        "reward": reward,
    }


def target_format_for_pilot_task(task: str) -> str:
    if task == "run_length":
        return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
    if task == "addition":
        return "integer result string formed from component prediction strings"
    raise ValueError(f"Unsupported task={task!r}")


def component_prediction_examples_for_pilot_task(task: str) -> List[str]:
    if task == "run_length":
        return ["3|0|2|2|3", "5|1|5|1|5"]
    if task == "addition":
        return ["46", "064", "1002"]
    raise ValueError(f"Unsupported task={task!r}")


def program_cases_for_pilot_task(task: str):
    if task == "run_length":
        return build_run_length_program_cases(random_seed=0, random_count=8)
    if task == "addition":
        return build_addition_program_cases()
    raise ValueError(f"Unsupported task={task!r}")


def raw_pilot_output(row: Mapping[str, Any]) -> Any:
    if "code_lines" in row:
        code_lines = row["code_lines"]
        if not isinstance(code_lines, list):
            raise ValueError("code_lines must be a list of strings")
        return "\n".join(str(line) for line in code_lines)
    for key in ("raw_output", "output", "completion", "proposal", "code"):
        if key in row:
            return row[key]
    return row


def completion_for_pilot_trace(parsed: Any, raw: Any) -> str:
    if hasattr(parsed, "to_completion"):
        return parsed.to_completion()
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, sort_keys=True)


def pilot_source_bounds(args: argparse.Namespace) -> tuple[int, int]:
    source_min = args.source_min_allowed
    if source_min is None:
        source_min = args.current_frontier_min if args.current_frontier_min > 0 else 1
    source_max = args.source_max_allowed
    if source_max is None:
        source_max = args.current_frontier_max if args.current_frontier_max > 0 else args.frontier_max_allowed
    return int(source_min), int(source_max)


def render_pilot_prompt(args: argparse.Namespace, aggregate_metrics: Mapping[str, Any]) -> PromptBundle:
    if args.condition == "config":
        space = DEFAULT_CONFIG_SEARCH_SPACES[args.task]
        source_min, source_max = pilot_source_bounds(args)
        return render_config_prompt(
            task_name=args.task,
            round_index=args.round_index,
            current_source={
                "min": source_min,
                "max": source_max,
            },
            allowed_target_frontier={
                "min": args.frontier_min_allowed,
                "max": args.frontier_max_allowed,
            },
            aggregate_metrics=aggregate_metrics,
            guard_choices=space["guards"],
            model_name=args.model_name,
        )
    return render_program_prompt(
        task_name=args.task,
        target_format=target_format_for_pilot_task(args.task),
        component_prediction_examples=component_prediction_examples_for_pilot_task(args.task),
    )


def validate_config_pilot_row(row: Mapping[str, Any], args: argparse.Namespace) -> JsonDict:
    space = DEFAULT_CONFIG_SEARCH_SPACES[args.task]
    source_min, source_max = pilot_source_bounds(args)
    validation = parse_config_proposal(
        raw_pilot_output(row),
        task_name=args.task,
        source_min_allowed=source_min,
        source_max_allowed=source_max,
        frontier_min_allowed=args.frontier_min_allowed,
        frontier_max_allowed=args.frontier_max_allowed,
        guards=space["guards"],
    )
    reward = pilot_reward(
        row,
        lambda_final=args.lambda_final,
        init_final_accuracy=args.init_final_accuracy,
    )
    completion = (
        completion_for_pilot_trace(validation.proposal, raw_pilot_output(row))
        if validation.valid
        else ""
    )
    return sanitize_json_value(
        {
            "id": row.get("id"),
            "condition": "config",
            "raw_output": raw_pilot_output(row),
            "valid": validation.valid,
            "validation_category": validation.category,
            "validation_message": validation.message,
            "parsed_proposal": validation.proposal.to_json_dict() if validation.valid else None,
            "completion": completion,
            **reward,
        }
    )


def validate_program_pilot_row(row: Mapping[str, Any], args: argparse.Namespace) -> JsonDict:
    code = str(raw_pilot_output(row))
    repair_prompt_text: Optional[str] = None

    def repair_callback(category: str, message: str, previous_program: str) -> Optional[str]:
        nonlocal repair_prompt_text
        repair_prompt = render_program_repair_prompt(
            task_name=args.task,
            target_format=target_format_for_pilot_task(args.task),
            failure_category=category,
            failure_summary=message,
            previous_program=previous_program,
        )
        repair_prompt_text = repair_prompt.text()
        if "repair_output_lines" in row:
            repair_lines = row["repair_output_lines"]
            if not isinstance(repair_lines, list):
                raise ValueError("repair_output_lines must be a list of strings")
            repaired = "\n".join(str(line) for line in repair_lines)
        else:
            repaired = row.get("repair_output", row.get("repaired_output"))
        return str(repaired) if repaired is not None else None

    validation: ProgramValidationResult = validate_program_with_repair(
        code,
        repair_callback=repair_callback,
        cases=program_cases_for_pilot_task(args.task),
        timeout_seconds=args.program_timeout_seconds,
        repair_attempts=args.repair_attempts,
    )
    reward = pilot_reward(
        row,
        lambda_final=args.lambda_final,
        init_final_accuracy=args.init_final_accuracy,
    )
    valid_code = validation.repaired_code if validation.valid and validation.repaired_code is not None else code
    return sanitize_json_value(
        {
            "id": row.get("id"),
            "condition": "program",
            "raw_output": code,
            "valid": validation.valid,
            "validation_category": validation.category,
            "validation_message": validation.message,
            "original_validation_category": validation.original_category,
            "original_validation_message": validation.original_message,
            "repair_attempted": validation.repair_attempted,
            "repaired": validation.repaired,
            "repair_prompt": repair_prompt_text,
            "repaired_output": validation.repaired_code,
            "completion": valid_code if validation.valid else "",
            **reward,
        }
    )


def process_pilot_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[JsonDict]:
    results: List[JsonDict] = []
    seen_completions: set[str] = set()
    for index, row in enumerate(rows):
        row_condition = row.get("condition", args.condition)
        if row_condition != args.condition:
            continue
        if args.condition == "config":
            result = validate_config_pilot_row(row, args)
        else:
            result = validate_program_pilot_row(row, args)
        result["proposal_index"] = index
        completion = str(result.get("completion", ""))
        duplicate = bool(completion) and completion in seen_completions
        result["duplicate"] = duplicate
        if completion:
            seen_completions.add(completion)
        if duplicate:
            result["trace_include"] = False
            result["selection_eligible"] = False
        else:
            result["trace_include"] = bool(result["valid"] and result["reward"] > 0)
            result["selection_eligible"] = bool(result["valid"] and result["reward"] > 0)
        results.append(result)
    return results


def select_best_pilot_result(results: Sequence[Mapping[str, Any]]) -> Optional[JsonDict]:
    eligible = [dict(result) for result in results if result.get("selection_eligible")]
    if not eligible:
        return None
    eligible.sort(key=lambda result: (float(result["reward"]), float(result["frontier_delta"])), reverse=True)
    selected = eligible[0]
    selected["selected"] = True
    return selected


def build_pilot_trace_rows(
    *,
    results: Sequence[Mapping[str, Any]],
    prompt: str,
    args: argparse.Namespace,
) -> List[JsonDict]:
    positives = [dict(result) for result in results if result.get("trace_include")]
    positives.sort(key=lambda result: float(result["reward"]), reverse=True)
    rows: List[JsonDict] = []
    for result in positives[: max(0, args.max_traces)]:
        rows.append(
            build_trace_row(
                round_index=args.round_index,
                task=args.task,
                condition=args.condition,
                reward=float(result["reward"]),
                frontier_delta=float(result["frontier_delta"]),
                final_accuracy=float(result["final_accuracy"]),
                prompt=prompt,
                completion=str(result["completion"]),
                extra_metadata={
                    "proposal_index": result.get("proposal_index"),
                    "proposal_id": result.get("id"),
                    "repaired": result.get("repaired", False),
                    "init_final_accuracy": result.get("init_final_accuracy"),
                    "final_accuracy_delta": result.get("final_accuracy_delta"),
                },
            )
        )
    return rows


# --- from proposals.py ---
"""Proposal generation, validation, prompting, and GRPO modules."""

from dataclasses import asdict, dataclass
from typing import Any, Dict

from self.adaptive.proposal import (
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
from self.adaptive.proposal import build_trace_row, load_fixture_proposals, write_trace_jsonl
from self.adaptive.proposal import (
    PromptBundle,
    render_config_prompt,
    render_program_prompt,
    render_program_repair_prompt,
)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class ProgramProposal:
    proposal_type: str
    task: str
    code: str

    def to_json_dict(self) -> JsonDict:
        return asdict(self)

    def to_completion(self) -> str:
        return self.code


__all__ = [
    "DEFAULT_CONFIG_SEARCH_SPACES",
    "PROPOSAL_OUTPUT_SCHEMAS",
    "ConfigProposal",
    "JsonDict",
    "ProgramProposal",
    "PromptBundle",
    "ProposalValidation",
    "build_trace_row",
    "extract_json_object",
    "load_fixture_proposals",
    "normalized_config_completion",
    "parse_config_proposal",
    "proposal_output_schema",
    "proposal_payload_for_schema",
    "render_config_prompt",
    "render_program_prompt",
    "render_program_repair_prompt",
    "validate_config_prediction",
    "write_trace_jsonl",
]

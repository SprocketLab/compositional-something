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
PROPOSAL_OUTPUT_SCHEMAS = ("plain", "action_observation")
PREDICTION_FIELD_ALIASES: Dict[str, str] = {
    "expected_avg_delta_from_current": "expected_avg_delta_from_current",
    "expected_current_avg_delta": "expected_avg_delta_from_current",
    "predicted_avg_delta_from_current": "expected_avg_delta_from_current",
    "predicted_current_avg_delta": "expected_avg_delta_from_current",
    "expected_avg_delta_from_init": "expected_avg_delta_from_init",
    "expected_final_delta_from_init": "expected_avg_delta_from_init",
    "predicted_avg_delta_from_init": "expected_avg_delta_from_init",
    "expected_target_delta": "expected_target_delta",
    "predicted_target_delta": "expected_target_delta",
    "expected_frontier_delta": "expected_frontier_delta",
    "predicted_frontier_delta": "expected_frontier_delta",
}


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
    text = str(raw).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        payload = None
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                payload = candidate
                break
        if payload is None:
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


def _normalize_prediction_payload(payload: Mapping[str, Any]) -> JsonDict:
    prediction: JsonDict = {}
    for raw_key, value in payload.items():
        key = PREDICTION_FIELD_ALIASES.get(str(raw_key))
        if key is None:
            continue
        numeric = _finite_float(value)
        if numeric is not None:
            prediction[key] = float(numeric)
    if "expected_avg_delta_from_init" in prediction:
        prediction.setdefault("expected_final_delta_from_init", prediction["expected_avg_delta_from_init"])
    return prediction


def _extract_tagged_block(text: str, tag: str, *, start: int = 0) -> Optional[Tuple[int, int, int, int, str]]:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    open_start = text.find(open_tag, start)
    if open_start < 0:
        return None
    content_start = open_start + len(open_tag)
    close_start = text.find(close_tag, content_start)
    if close_start < 0:
        return None
    close_end = close_start + len(close_tag)
    return open_start, content_start, close_start, close_end, text[content_start:close_start].strip()


def _extract_json_observation_payload(raw: Any) -> Tuple[Any, Optional[JsonDict], Optional[str], Optional[str]]:
    payload = _row_payload(raw)
    if payload is None:
        return raw, None, "parse_error", "action_observation output must contain a JSON object"
    if not isinstance(payload, dict):
        return raw, None, "parse_error", "action_observation output must contain a JSON object"
    allowed = {"reasoning", "left", "right", "guard", "notes", "rationale", *PREDICTION_FIELD_ALIASES}
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        return payload, None, "schema_error", (
            "action_observation JSON may only contain reasoning/prediction fields/left/right/guard; unexpected: "
            + ", ".join(unexpected)
        )
    normalized = dict(payload)
    reasoning = str(
        normalized.pop("reasoning", None)
        or normalized.pop("notes", None)
        or normalized.pop("rationale", None)
        or ""
    ).strip()
    if reasoning:
        normalized["notes"] = reasoning[:500]
    prediction = _normalize_prediction_payload(payload)
    return normalized, prediction or None, None, None


def proposal_payload_for_schema(raw: Any, schema: str) -> Tuple[Any, Optional[JsonDict], Optional[str], Optional[str]]:
    if schema == "plain":
        return raw, None, None, None
    if schema == "action_observation":
        return _extract_json_observation_payload(raw)
    return raw, None, "schema_error", f"unsupported proposal output schema={schema!r}"


def proposal_payload_for_completion(proposal: ConfigProposal) -> JsonDict:
    payload: JsonDict = {
        "left": int(proposal.left),
        "right": int(proposal.right),
        "guard": str(proposal.guard),
    }
    if proposal.notes:
        payload["notes"] = proposal.notes
    return payload


def action_payload_for_completion(proposal: ConfigProposal) -> JsonDict:
    return {
        "left": int(proposal.left),
        "right": int(proposal.right),
        "guard": str(proposal.guard),
    }


def _reasoning_for_completion(proposal: ConfigProposal) -> str:
    reasoning = str(proposal.notes or "").strip()
    if reasoning:
        return reasoning
    return (
        f"I will compose source sizes {int(proposal.left)} and {int(proposal.right)} "
        f"to target {int(proposal.target)} using guard {proposal.guard}."
    )


def normalized_config_completion(
    *,
    proposal: ConfigProposal,
    prediction: Optional[Mapping[str, Any]],
    schema: str,
) -> str:
    if schema == "plain":
        return proposal.to_completion()
    if schema == "action_observation":
        payload: JsonDict = {
            "reasoning": _reasoning_for_completion(proposal),
        }
        if prediction:
            for key in (
                "expected_avg_delta_from_current",
                "expected_avg_delta_from_init",
                "expected_target_delta",
                "expected_frontier_delta",
            ):
                if key in prediction:
                    payload[key] = prediction[key]
            if "expected_avg_delta_from_init" not in payload and "expected_final_delta_from_init" in prediction:
                payload["expected_avg_delta_from_init"] = prediction["expected_final_delta_from_init"]
        payload.update(
            {
                "left": int(proposal.left),
                "right": int(proposal.right),
                "guard": str(proposal.guard),
            }
        )
        return json.dumps(sanitize_json_value(payload), separators=(",", ":"))
    raise ValueError(f"Unsupported proposal output schema={schema!r}.")


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
        notes=str(payload.get("notes") or payload.get("reasoning") or payload.get("rationale") or ""),
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
        parsed_prediction = dict(prediction_payload) if prediction_payload else None
        validation_valid = bool(validation.valid)
        category = validation.category
        message = validation.message
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


def _guard_decision_rule_text(task_name: str) -> str:
    if task_name == "addition":
        return (
            "- Guard choice: use none when carrying across the component boundary is acceptable; "
            "use reject_boundary_carry when boundary carries are likely to make composed pseudo-labels noisy.\n"
        )
    if task_name == "run_length":
        return (
            "- Guard choice: use none for broad coverage; use reject_boundary_continue when boundary-spanning runs "
            "look risky; use require_boundary_continue when the target weakness seems tied to boundary-spanning runs.\n"
        )
    return "- Guard choice: choose the listed guard that best matches the likely composition failure mode.\n"


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
    if proposal_output_schema == "plain":
        system = (
            "You are generating a composition configuration for a compositional self-improvement pipeline.\n"
            "You must output only valid JSON matching the requested schema.\n"
            "Do not include explanations, markdown, or comments.\n"
            "You cannot call tools. You only choose a composition configuration."
        )
        schema = {
            "left": "integer source slice size",
            "right": "integer source slice size",
            "guard": f"one of {_format_choices(guard_choices)}",
            "notes": "optional short string",
        }
        schema_notes = (
            f"{json.dumps(schema, indent=2)}\n\n"
            "Output the configuration object directly. The driver will execute this proposal and measure reward."
        )
        goal_extra = ""
        objective_extra = ""
        constraints_extra = ""
    elif proposal_output_schema == "action_observation":
        system = (
            "You are generating a JSON action for a compositional self-improvement pipeline.\n"
            "Output exactly one valid JSON object with reasoning, outcome predictions, left, right, and guard.\n"
            "Do not include markdown or code fences.\n"
            "You cannot call tools. You only choose a composition configuration."
        )
        schema_notes = (
            "Return exactly one JSON object with these keys:\n"
            "- reasoning: brief string explaining which current source slices look reliable, which target size left + right reaches, "
            "and why the guard is appropriate.\n"
            "- expected_avg_delta_from_current: number predicting candidate_avg_accuracy - current_avg_accuracy.\n"
            "- expected_target_delta: number predicting target-slice accuracy improvement.\n"
            "- expected_frontier_delta: number predicting static-frontier accuracy improvement.\n"
            "- left: integer source slice size from current_source_slices.\n"
            "- right: integer source slice size from current_source_slices.\n"
            f"- guard: one of {_format_choices(guard_choices)}.\n\n"
            "Put prediction fields after reasoning and before left/right/guard.\n"
            "Choose numeric left and right values only from current_source_slices in the current state above; "
            "do not copy numbers from schema text, examples, or unrelated instructions. "
            "Ensure left + right lands inside allowed_target_frontier.\n\n"
            "The driver executes only left, right, and guard, stores reasoning and predictions, then appends the environment observation during training."
        )
        goal_extra = "Put reasoning first, then your expected outcome deltas, then the config action."
        objective_extra = (
            "- In reasoning, state which source slices look reliable, which target size you are aiming at, "
            "and why the guard is appropriate.\n"
            "- Predict expected outcome deltas before choosing left/right/guard so the proposal states its acquisition hypothesis.\n"
        )
        constraints_extra = (
            "- The JSON object may contain only reasoning, expected_* prediction fields, left, right, and guard.\n"
            "- Predict only expected deltas, not validity labels or post-execution observations; the driver supplies realized observations after execution.\n"
        )
    else:
        raise ValueError(f"Unsupported proposal_output_schema={proposal_output_schema!r}.")
    goal_sentence = (
        "Choose two source slices and a guard rule for composing model component predictions into pseudo-labels."
    )
    if goal_extra:
        goal_sentence = f"{goal_sentence} {goal_extra}"
    aggregate_payload = dict(aggregate_metrics)
    final_output_instruction = (
        "Output only JSON."
    )
    user = (
        f"Task: {task_name}\n\n"
        "Goal:\n"
        f"{goal_sentence}\n\n"
        "Current attempt context:\n"
        f"- next_selected_trace_index_if_chosen: {round_index}\n"
        f"- current_source_slices: {json.dumps(sanitize_json_value(dict(current_source)), sort_keys=True)}\n"
        f"- allowed_target_frontier: {json.dumps(sanitize_json_value(dict(allowed_target_frontier)), sort_keys=True)}\n\n"
        "Aggregate diagnostics from prior evaluation:\n"
        f"{json.dumps(sanitize_json_value(aggregate_payload), sort_keys=True, indent=2)}\n\n"
        "Objective:\n"
        "- Choose the valid configuration with the highest expected reward.\n"
        "- Reward is candidate_avg_accuracy - current_avg_accuracy.\n"
        "- Reusing an existing target slice is allowed; measured reward determines whether rehearsal helps.\n"
        f"{objective_extra}"
        "\n"
        "Decision rules:\n"
        "- Identify reliable source sizes from per_size_accuracy; higher source-slice accuracy means component predictions are more trustworthy.\n"
        "- Identify weak reachable targets: target = left + right must be inside allowed_target_frontier, and low current target accuracy means there is more room to improve.\n"
        "- Prefer actions expected to improve current_avg_accuracy, not just the target slice in isolation.\n"
        "- Prefer actions that may grow the source pool when source_admission_target_accuracy_threshold is shown in diagnostics.\n"
        "- If recent_selected_actions is provided, avoid exact repeats of left/right/guard/target unless the current diagnostics justify trying the same action again.\n"
        f"{_guard_decision_rule_text(task_name)}"
        "\n"
        "Output format:\n"
        f"{schema_notes}\n\n"
        "Constraints:\n"
        "- left and right must be source slice sizes currently available to the model.\n"
        "- left + right must land inside allowed_target_frontier.\n"
        "- Choose only a listed guard value.\n"
        f"{constraints_extra}"
        "- Do not choose frontier ranges, data budgets, sampling schedules, or composition paths; the driver owns those.\n"
        "- The driver composes pseudo-labels from model component predictions, not oracle labels.\n\n"
        f"{final_output_instruction}"
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
        "Current attempt context:\n"
        f"- next_selected_trace_index_if_chosen: {round_index}\n"
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
    batch_size: int = 1,
) -> List[JsonDict]:
    import torch
    from self.core.evaluation import build_generation_encodings

    device = next(model.parameters()).device
    rows: List[JsonDict] = []
    model_was_training = model.training
    model.eval()
    batch_size = max(1, int(batch_size))
    prompt_text = prompt.text()
    with torch.no_grad():
        for start in range(0, int(num_candidates), batch_size):
            current_batch_size = min(batch_size, int(num_candidates) - start)
            encodings = build_generation_encodings(
                tokenizer,
                [prompt_text] * current_batch_size,
                device,
            )
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
            for batch_index in range(current_batch_size):
                decoded = tokenizer.decode(
                    output_ids[batch_index, prompt_width:].tolist(),
                    skip_special_tokens=True,
                )
                rows.append({"id": f"model_candidate_{start + batch_index}", "raw_output": decoded})
    if model_was_training:
        model.train()
    return rows


def _config_valid_action_key(result: Mapping[str, Any]) -> Optional[Tuple[Any, ...]]:
    if not bool(result.get("valid")):
        return None
    parsed = result.get("parsed_proposal")
    if not isinstance(parsed, Mapping):
        return None
    try:
        left = int(parsed["left"])
        right = int(parsed["right"])
        target = int(parsed.get("target", left + right))
        guard = str(parsed["guard"])
    except (KeyError, TypeError, ValueError):
        return None
    return ("proposal", left, right, guard, target)


def _proposal_unique_max_draws(args: argparse.Namespace) -> int:
    configured = int(getattr(args, "proposal_unique_max_draws", 0) or 0)
    if configured > 0:
        return configured
    num_candidates = int(args.num_candidates)
    return max(num_candidates * 8, num_candidates + 16)


def generate_unique_config_proposals_from_model(
    *,
    args: argparse.Namespace,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: PromptBundle,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
) -> Tuple[List[JsonDict], JsonDict, List[JsonDict]]:
    """Sample until config proposals contain unique normalized actions.

    The returned candidate rows are strictly valid, unique config actions. Invalid
    or duplicate draws are rejected for candidate training but returned in
    draw_results so proposal-GRPO can still learn from the verifier outcome.
    """

    requested = int(args.num_candidates)
    max_draws = _proposal_unique_max_draws(args)
    sampling_batch_size = max(1, int(getattr(args, "proposal_sampling_batch_size", 1)))
    kept_rows: List[JsonDict] = []
    seen_valid_actions: set[Tuple[Any, ...]] = set()
    draw_records: List[JsonDict] = []
    draw_results: List[JsonDict] = []
    draw_index = 0

    while len(kept_rows) < requested and draw_index < max_draws:
        batch_rows = generate_proposals_from_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            num_candidates=min(sampling_batch_size, max_draws - draw_index),
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
            batch_size=sampling_batch_size,
        )
        for row in batch_rows:
            if draw_index >= max_draws:
                break
            row = dict(row)
            row["id"] = f"model_candidate_draw_{draw_index}"
            row["unique_draw_index"] = draw_index

            result = validate_config_rows(
                rows=[row],
                args=args,
                source_sizes=source_sizes,
                frontier_min=frontier_min,
                frontier_max=frontier_max,
            )[0]
            action_key = _config_valid_action_key(result)
            kept_reason: Optional[str]
            candidate_index: Optional[int] = None
            if len(kept_rows) < requested and action_key is not None and action_key not in seen_valid_actions:
                seen_valid_actions.add(action_key)
                kept_reason = "unique_valid_action"
                candidate_index = len(kept_rows)
                kept_row = dict(row)
                kept_row["id"] = f"model_candidate_{candidate_index}"
                kept_row["unique_generation_kept_reason"] = kept_reason
                kept_rows.append(kept_row)
            else:
                kept_reason = "duplicate_valid_action" if action_key is not None else "invalid"
                row["unique_generation_reject_reason"] = kept_reason

            draw_result = dict(result)
            draw_result["proposal_index"] = draw_index
            draw_result["draw_index"] = draw_index
            draw_result["id"] = row["id"]
            draw_result["action_key"] = list(action_key) if action_key is not None else None
            draw_result["kept_for_candidate"] = kept_reason == "unique_valid_action"
            draw_result["unique_generation_reason"] = kept_reason
            if candidate_index is not None:
                draw_result["candidate_proposal_index"] = candidate_index
                draw_result["candidate_id"] = f"model_candidate_{candidate_index}"
            draw_results.append(sanitize_json_value(draw_result))
            draw_records.append(
                sanitize_json_value(
                    {
                        "draw_index": draw_index,
                        "valid": bool(result.get("valid")),
                        "validation_category": result.get("validation_category"),
                        "validation_message": result.get("validation_message"),
                        "action_key": list(action_key) if action_key is not None else None,
                        "kept": kept_reason == "unique_valid_action",
                        "reason": kept_reason,
                        "candidate_proposal_index": candidate_index,
                    }
                )
            )
            draw_index += 1
            if len(kept_rows) >= requested:
                break

    unique_valid_count = len(seen_valid_actions)
    summary: JsonDict = sanitize_json_value(
        {
            "enabled": True,
            "semantic": "config_action",
            "strict_valid_unique": True,
            "requested_unique_proposals": requested,
            "max_draws": max_draws,
            "sampling_batch_size": sampling_batch_size,
            "total_draws": draw_index,
            "unique_valid_actions": unique_valid_count,
            "returned_rows": len(kept_rows),
            "fallback_rows_returned": 0,
            "draw_result_count": len(draw_results),
            "reached_requested_unique_count": unique_valid_count >= requested,
            "temperature": args.proposal_temperature,
            "top_p": args.proposal_top_p,
            "draws": draw_records,
        }
    )
    return kept_rows, summary, draw_results


def load_or_generate_proposal_rows(
    *,
    args: argparse.Namespace,
    prompt: PromptBundle,
    current_model: AutoModelForCausalLM,
    current_tokenizer: AutoTokenizer,
    round_index: int,
    attempt_index: Optional[int] = None,
    source_sizes: Optional[set[int]] = None,
    frontier_min: Optional[int] = None,
    frontier_max: Optional[int] = None,
    unique_log_path: Optional[Path] = None,
    draw_results_log_path: Optional[Path] = None,
) -> List[JsonDict]:
    if args.proposal_fixture_jsonl is not None:
        rows = _rows_for_round(
            load_fixture_proposals(args.proposal_fixture_jsonl),
            round_index,
            attempt_index=attempt_index,
        )
        return [dict(row) for row in rows[: args.num_candidates]]

    if args.proposal_model_name == "current":
        if (
            bool(getattr(args, "force_unique_proposals", False))
            and args.condition == "config"
            and source_sizes is not None
            and frontier_min is not None
            and frontier_max is not None
        ):
            rows, summary, draw_results = generate_unique_config_proposals_from_model(
                args=args,
                model=current_model,
                tokenizer=current_tokenizer,
                prompt=prompt,
                source_sizes=source_sizes,
                frontier_min=int(frontier_min),
                frontier_max=int(frontier_max),
            )
            if unique_log_path is not None:
                unique_log_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if draw_results_log_path is not None:
                draw_results_log_path.write_text(json.dumps(draw_results, indent=2) + "\n", encoding="utf-8")
            return rows
        return generate_proposals_from_model(
            model=current_model,
            tokenizer=current_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
            batch_size=getattr(args, "proposal_sampling_batch_size", 1),
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
        if (
            bool(getattr(args, "force_unique_proposals", False))
            and args.condition == "config"
            and source_sizes is not None
            and frontier_min is not None
            and frontier_max is not None
        ):
            rows, summary, draw_results = generate_unique_config_proposals_from_model(
                args=args,
                model=proposal_model,
                tokenizer=proposal_tokenizer,
                prompt=prompt,
                source_sizes=source_sizes,
                frontier_min=int(frontier_min),
                frontier_max=int(frontier_max),
            )
            if unique_log_path is not None:
                unique_log_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            if draw_results_log_path is not None:
                draw_results_log_path.write_text(json.dumps(draw_results, indent=2) + "\n", encoding="utf-8")
            return rows
        return generate_proposals_from_model(
            model=proposal_model,
            tokenizer=proposal_tokenizer,
            prompt=prompt,
            num_candidates=args.num_candidates,
            max_new_tokens=args.proposal_max_new_tokens,
            temperature=args.proposal_temperature,
            top_p=args.proposal_top_p,
            batch_size=getattr(args, "proposal_sampling_batch_size", 1),
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
            from self.core.models import ExecutableProposal

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
PROPOSAL_GRPO_REWARD_MODES = ("outcome", "rank", "validity")
PROPOSAL_UPDATE_LOSS_MODES = ("legacy_grpo", "merged_agent")
PROPOSAL_GRPO_SPAN_MODES = ("reasoning_action", "action_only")
PROPOSAL_GRPO_OBJECTIVES = ("grpo", "dr_grpo")
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


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _proposal_action_key(result: Mapping[str, Any]) -> Tuple[Any, ...]:
    parsed = result.get("parsed_proposal")
    if isinstance(parsed, Mapping):
        try:
            left = int(parsed["left"])
            right = int(parsed["right"])
            target = int(parsed.get("target", left + right))
            guard = str(parsed.get("guard", "none"))
            return ("proposal", left, right, guard, target)
        except (KeyError, TypeError, ValueError):
            pass
    completion = result.get("completion")
    if isinstance(completion, str) and completion:
        return ("completion", completion)
    raw = result.get("raw_output", "")
    if isinstance(raw, str):
        return ("raw", raw)
    return ("raw_json", json.dumps(sanitize_json_value(raw), sort_keys=True))


def _is_config_proposal_action_key(key: Tuple[Any, ...]) -> bool:
    return bool(key and key[0] == "proposal")


def _proposal_trace_action_key(trace: Any) -> Optional[Tuple[Any, ...]]:
    metadata = getattr(trace, "metadata", None)
    if metadata is None and isinstance(trace, Mapping):
        metadata = trace.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    try:
        left = int(metadata["left"])
        right = int(metadata["right"])
        target = int(metadata.get("target", left + right))
        guard = str(metadata.get("guard", "none"))
    except (KeyError, TypeError, ValueError):
        return None
    return ("proposal", left, right, guard, target)


def _count_action_keys(keys: Sequence[Tuple[Any, ...]]) -> Dict[Tuple[Any, ...], int]:
    counts: Dict[Tuple[Any, ...], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    return counts


def _action_entropy_summary(keys: Sequence[Tuple[Any, ...]], *, prefix: str) -> JsonDict:
    total = len(keys)
    counts = _count_action_keys(keys)
    unique = len(counts)
    if total <= 0:
        entropy = math.nan
        effective = math.nan
        duplicate_rate = math.nan
    else:
        entropy = 0.0
        for count in counts.values():
            probability = float(count) / float(total)
            entropy -= probability * math.log(probability)
        effective = math.exp(entropy)
        duplicate_rate = 1.0 - (float(unique) / float(total))
    top_counts = [
        {"action_key": list(key), "count": int(count)}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[:10]
    ]
    return {
        f"{prefix}_action_count": total,
        f"{prefix}_unique_action_count": unique,
        f"{prefix}_action_entropy": entropy,
        f"{prefix}_effective_action_count": effective,
        f"{prefix}_duplicate_action_rate": duplicate_rate,
        f"{prefix}_top_action_counts": top_counts,
    }


def _dedup_entry_score(entry: Mapping[str, Any]) -> Tuple[float, int]:
    rank_score = _finite_float_or_none(entry.get("rank_score"))
    reward = _finite_float_or_none(entry.get("reward"))
    score = rank_score if rank_score is not None else (reward if reward is not None else float("-inf"))
    proposal_index = int(entry.get("proposal_index", 0))
    return score, -proposal_index


def _deduplicate_reward_entries(entries: Sequence[JsonDict]) -> Tuple[List[JsonDict], int, int]:
    best_by_key: Dict[Tuple[Any, ...], JsonDict] = {}
    duplicate_count = 0
    for entry in entries:
        key = tuple(entry["action_key"])
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = dict(entry)
            continue
        duplicate_count += 1
        if _dedup_entry_score(entry) > _dedup_entry_score(current):
            best_by_key[key] = dict(entry)
    kept = sorted(best_by_key.values(), key=lambda payload: int(payload["proposal_index"]))
    return kept, duplicate_count, len(best_by_key)


def _assign_rank_rewards(entries: Sequence[JsonDict]) -> None:
    ranked = [
        (index, _finite_float_or_none(entry.get("rank_score")))
        for index, entry in enumerate(entries)
        if entry.get("reward_source") == "rank_outcome"
    ]
    ranked = [(index, score) for index, score in ranked if score is not None]
    if not ranked:
        return
    ranked.sort(key=lambda item: float(item[1]))
    denominator = max(1, len(ranked) - 1)
    position = 0
    while position < len(ranked):
        score = ranked[position][1]
        end = position + 1
        while end < len(ranked) and ranked[end][1] == score:
            end += 1
        mean_position = (position + end - 1) / 2.0
        reward = 1.0 if len(ranked) == 1 else float(mean_position / denominator)
        for entry_index, _ in ranked[position:end]:
            entries[entry_index]["reward"] = reward
        position = end


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
    objective: str = "grpo",
    eps: float = 1e-6,
) -> Tuple[List[float], bool, str]:
    if not rewards:
        return [], True, "no_rewards"
    if objective not in PROPOSAL_GRPO_OBJECTIVES:
        raise ValueError(f"Unsupported proposal_grpo_objective={objective!r}.")
    reward_values = [float(reward) for reward in rewards]
    mean_reward = sum(reward_values) / len(reward_values)
    variance = sum((reward - mean_reward) ** 2 for reward in reward_values) / len(reward_values)
    std_reward = math.sqrt(variance)
    if std_reward > eps:
        if objective == "dr_grpo":
            return [reward - mean_reward for reward in reward_values], False, "mean_centered"
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
    action_history: Sequence[Any] = (),
) -> Tuple[List[ProposalGRPOTrace], JsonDict]:
    prompt_text = prompt.text()
    metrics_by_index = {int(metric.index): metric for metric in candidate_metrics}

    def metric_for_result(result: Mapping[str, Any], proposal_index: int) -> Optional[Any]:
        candidate_index = result.get("candidate_proposal_index")
        if candidate_index is None and result.get("kept_for_candidate") is not False:
            candidate_index = proposal_index
        try:
            return metrics_by_index.get(int(candidate_index)) if candidate_index is not None else None
        except (TypeError, ValueError):
            return None

    included_entries: List[JsonDict] = []
    reward_source_counts: Dict[str, int] = {}
    skipped_system_failure_count = 0
    skipped_candidate_duplicate_count = 0
    reward_mode = args.proposal_grpo_reward_mode
    for index, result in enumerate(proposal_results):
        proposal_index = _proposal_index(result, index)
        if bool(result.get("candidate_dedup_skipped")):
            reward_source_counts["candidate_dedup_skipped"] = (
                reward_source_counts.get("candidate_dedup_skipped", 0) + 1
            )
            skipped_candidate_duplicate_count += 1
            continue
        metric = metric_for_result(result, proposal_index)
        rank_score: Optional[float] = None
        if reward_mode == "rank":
            if not bool(result.get("valid")):
                reward, reward_source = proposal_grpo_outcome_invalid_reward(result), "invalid"
            elif metric is None:
                reward, reward_source = 0.0, "valid_untrained"
            elif not metric.valid:
                if _is_system_candidate_failure(metric):
                    reward, reward_source = None, "skipped_system_failure"
                else:
                    reward, reward_source = 0.0, "valid_untrained"
            else:
                rank_score = _finite_float_or_none(metric.reward)
                reward, reward_source = None, "rank_outcome"
        else:
            reward, reward_source = proposal_grpo_reward_for_result(
                result,
                metric=metric,
                reward_mode=reward_mode,
                outcome_scale=args.proposal_grpo_outcome_scale,
            )
        reward_source_counts[reward_source] = reward_source_counts.get(reward_source, 0) + 1
        if reward is None:
            if reward_source == "skipped_system_failure":
                skipped_system_failure_count += 1
                continue
            if reward_source != "rank_outcome":
                continue
        included_entries.append(
            {
                "proposal_index": proposal_index,
                "result": result,
                "metric": metric,
                "reward": reward,
                "reward_source": reward_source,
                "rank_score": rank_score,
                "action_key": _proposal_action_key(result),
            }
        )

    pre_dedup_count = len(included_entries)
    pre_dedup_action_keys = [tuple(entry["action_key"]) for entry in included_entries]
    current_action_counts = _count_action_keys(pre_dedup_action_keys)
    action_history_keys = [
        key for key in (_proposal_trace_action_key(trace) for trace in action_history) if key is not None
    ]
    action_history_counts = _count_action_keys(action_history_keys)
    deduplicated_action_count = 0
    unique_action_count = len(set(pre_dedup_action_keys))
    if bool(getattr(args, "proposal_grpo_deduplicate_actions", True)) and included_entries:
        included_entries, deduplicated_action_count, unique_action_count = _deduplicate_reward_entries(
            included_entries
        )
    if reward_mode == "rank":
        _assign_rank_rewards(included_entries)

    novelty_beta = float(getattr(args, "proposal_grpo_novelty_bonus_beta", 0.0))
    novelty_bonus_values: List[float] = []
    for entry in included_entries:
        entry["base_reward"] = entry.get("reward")
        entry["novelty_bonus"] = 0.0
        entry["action_history_count"] = 0
        entry["current_action_multiplicity"] = current_action_counts.get(tuple(entry["action_key"]), 0)
        if novelty_beta <= 0.0:
            continue
        result = entry["result"]
        action_key = tuple(entry["action_key"])
        if not bool(result.get("valid")) or not _is_config_proposal_action_key(action_key):
            continue
        reward = _finite_float_or_none(entry.get("reward"))
        if reward is None:
            continue
        history_count = int(action_history_counts.get(action_key, 0))
        current_multiplicity = int(current_action_counts.get(action_key, 1))
        novelty_count = history_count + max(0, current_multiplicity - 1)
        novelty_bonus = float(novelty_beta) / math.sqrt(float(novelty_count + 1))
        entry["action_history_count"] = history_count
        entry["novelty_count"] = novelty_count
        entry["novelty_bonus"] = novelty_bonus
        entry["reward"] = _clamp(float(reward) + novelty_bonus, -1.0, 1.0)
        novelty_bonus_values.append(novelty_bonus)

    trainable_reward_source_counts: Dict[str, int] = {}
    rewards: List[float] = []
    for entry in included_entries:
        reward = _finite_float_or_none(entry.get("reward"))
        if reward is None:
            continue
        entry["reward"] = float(reward)
        reward_source = str(entry["reward_source"])
        trainable_reward_source_counts[reward_source] = trainable_reward_source_counts.get(reward_source, 0) + 1
        rewards.append(float(reward))
    included_entries = [entry for entry in included_entries if _finite_float_or_none(entry.get("reward")) is not None]
    invalid_only_fixed_baseline = bool(rewards) and trainable_reward_source_counts.get("invalid", 0) == len(rewards)
    zero_variance_mode = str(args.proposal_grpo_zero_variance)
    if invalid_only_fixed_baseline and zero_variance_mode == "skip":
        zero_variance_mode = "fixed_baseline"
    advantages, skip_update, advantage_mode = proposal_grpo_advantages(
        rewards,
        zero_variance=zero_variance_mode,
        fixed_baseline=args.proposal_grpo_fixed_baseline,
        objective=str(getattr(args, "proposal_grpo_objective", "grpo")),
    )
    traces: List[ProposalGRPOTrace] = []
    for entry, advantage in zip(included_entries, advantages):
        proposal_index = int(entry["proposal_index"])
        result = entry["result"]
        metric = entry["metric"]
        reward = float(entry["reward"])
        reward_source = str(entry["reward_source"])
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
                    "base_reward": entry.get("base_reward"),
                    "novelty_bonus": entry.get("novelty_bonus", 0.0),
                    "action_history_count": entry.get("action_history_count", 0),
                    "current_action_multiplicity": entry.get("current_action_multiplicity", 0),
                    "novelty_count": entry.get("novelty_count", entry.get("action_history_count", 0)),
                    "rank_score": entry.get("rank_score"),
                    "deduplicated_for_policy": bool(
                        getattr(args, "proposal_grpo_deduplicate_actions", True)
                    ),
                    "candidate_reward": metric.reward if metric is not None else None,
                    "frontier_delta": metric.frontier_delta if metric is not None else None,
                    "final_accuracy": metric.final_accuracy if metric is not None else None,
                    "final_accuracy_delta": metric.final_accuracy_delta if metric is not None else None,
                    "final_accuracy_delta_from_current": (
                        metric.final_accuracy_delta_from_current if metric is not None else None
                    ),
                    "target_delta": metric.target_delta if metric is not None else None,
                    "per_size_delta": getattr(metric, "per_size_delta", None) if metric is not None else None,
                    "per_size_accuracy": getattr(metric, "per_size_accuracy", None) if metric is not None else None,
                    "failure_reason": getattr(metric, "failure_reason", None) if metric is not None else None,
                    "proposal_prediction": getattr(metric, "proposal_prediction", None) if metric is not None else None,
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
        "invalid_only_fixed_baseline": bool(invalid_only_fixed_baseline),
        "requested_zero_variance_mode": str(args.proposal_grpo_zero_variance),
        "reward_mode": args.proposal_grpo_reward_mode,
        "objective": str(getattr(args, "proposal_grpo_objective", "grpo")),
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "novelty_bonus_beta": float(getattr(args, "proposal_grpo_novelty_bonus_beta", 0.0)),
        "novelty_bonus_mean": (
            sum(novelty_bonus_values) / len(novelty_bonus_values) if novelty_bonus_values else 0.0
        ),
        "novelty_bonus_max": max(novelty_bonus_values) if novelty_bonus_values else 0.0,
        "reward_source_counts": reward_source_counts,
        "trainable_reward_source_counts": trainable_reward_source_counts,
        "skipped_system_failure_count": skipped_system_failure_count,
        "skipped_candidate_duplicate_count": skipped_candidate_duplicate_count,
        "deduplicate_actions": bool(getattr(args, "proposal_grpo_deduplicate_actions", True)),
        "pre_dedup_trace_count": pre_dedup_count,
        "deduplicated_action_count": deduplicated_action_count,
        "unique_action_count": unique_action_count,
        "action_history_count": len(action_history_keys),
        "input_proposal_count": len(proposal_results),
        "trace_candidate_metric_count": len(candidate_metrics),
        **_action_entropy_summary(pre_dedup_action_keys, prefix="pre_dedup"),
        **_action_entropy_summary([tuple(entry["action_key"]) for entry in included_entries], prefix="trainable"),
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
    PROPOSAL_GRPO_OBJECTIVES,
    PROPOSAL_GRPO_SPAN_MODES,
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
    completion_char_span: Optional[Tuple[int, int]] = None,
    completion_char_exclude_spans: Sequence[Tuple[int, int]] = (),
) -> Optional[JsonDict]:
    if completion == "":
        return None
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if not completion_ids:
        return None
    completion_token_mask = _completion_token_mask(
        tokenizer=tokenizer,
        completion=completion,
        completion_token_count=len(completion_ids),
        char_span=completion_char_span,
        exclude_spans=completion_char_exclude_spans,
    )
    input_ids: List[int] = []
    if tokenizer.bos_token_id is not None:
        input_ids.append(int(tokenizer.bos_token_id))
    input_ids.extend(int(token_id) for token_id in prompt_ids)
    completion_start = len(input_ids)
    input_ids.extend(int(token_id) for token_id in completion_ids)
    if len(input_ids) < 2:
        return None
    completion_mask: List[bool] = []
    for position in range(len(input_ids) - 1):
        target_index = position + 1
        if completion_start <= target_index < completion_start + len(completion_ids):
            completion_mask.append(bool(completion_token_mask[target_index - completion_start]))
        else:
            completion_mask.append(False)
    if not any(completion_mask):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "completion_mask": completion_mask,
        "completion_tokens": int(sum(1 for keep in completion_token_mask if keep)),
        "total_completion_tokens": len(completion_ids),
    }


def _completion_token_mask(
    *,
    tokenizer: AutoTokenizer,
    completion: str,
    completion_token_count: int,
    char_span: Optional[Tuple[int, int]],
    exclude_spans: Sequence[Tuple[int, int]] = (),
) -> List[bool]:
    if char_span is None:
        mask = [True] * completion_token_count
    else:
        include_span = _completion_char_span_to_token_span(
            tokenizer=tokenizer,
            completion=completion,
            completion_token_count=completion_token_count,
            char_span=char_span,
            force_nonempty=True,
        )
        if include_span is None:
            mask = [True] * completion_token_count
        else:
            start_token, end_token = include_span
            mask = [start_token <= index < end_token for index in range(completion_token_count)]
    for exclude_span in exclude_spans:
        token_span = _completion_char_span_to_token_span(
            tokenizer=tokenizer,
            completion=completion,
            completion_token_count=completion_token_count,
            char_span=exclude_span,
            force_nonempty=False,
        )
        if token_span is None:
            continue
        start_token, end_token = token_span
        for index in range(start_token, end_token):
            if 0 <= index < completion_token_count:
                mask[index] = False
    return mask


def _completion_char_span_to_token_span(
    *,
    tokenizer: AutoTokenizer,
    completion: str,
    completion_token_count: int,
    char_span: Tuple[int, int],
    force_nonempty: bool,
) -> Optional[Tuple[int, int]]:
    start_char, end_char = char_span
    start_char = max(0, min(len(completion), int(start_char)))
    end_char = max(start_char, min(len(completion), int(end_char)))
    if end_char <= start_char:
        return None
    start_token = len(tokenizer.encode(completion[:start_char], add_special_tokens=False))
    end_token = len(tokenizer.encode(completion[:end_char], add_special_tokens=False))
    start_token = max(0, min(completion_token_count, start_token))
    end_token = min(completion_token_count, end_token)
    if force_nonempty:
        end_token = max(start_token + 1, end_token)
    end_token = min(completion_token_count, end_token)
    if end_token <= start_token:
        return None
    return start_token, end_token


def _json_value_char_span(completion: str, key: str) -> Optional[Tuple[int, int]]:
    marker = json.dumps(str(key), separators=(",", ":")) + ":"
    start = completion.find(marker)
    if start < 0:
        return None
    value_start = start + len(marker)
    try:
        _, value_length = json.JSONDecoder().raw_decode(completion[value_start:])
    except json.JSONDecodeError:
        return None
    return value_start, value_start + value_length


FORMAT_CONFIG_VALUE_KEYS = (
    "reasoning",
    "notes",
    "rationale",
    "expected_avg_delta_from_current",
    "expected_avg_delta_from_init",
    "expected_final_delta_from_init",
    "expected_target_delta",
    "expected_frontier_delta",
    "left",
    "right",
    "guard",
    "target",
)


def _json_value_content_char_span(completion: str, key: str) -> Optional[Tuple[int, int]]:
    span = _json_value_char_span(completion, key)
    if span is None:
        return None
    start, end = span
    if end - start >= 2 and completion[start] == '"' and completion[end - 1] == '"':
        start += 1
        end -= 1
    if end <= start:
        return None
    return start, end


def _proposal_format_value_char_spans(completion: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for key in FORMAT_CONFIG_VALUE_KEYS:
        span = _json_value_content_char_span(completion, key)
        if span is None or span in seen:
            continue
        seen.add(span)
        spans.append(span)
    return spans


def _tagged_action_json_span(completion: str) -> Optional[Tuple[int, int]]:
    block = _extract_tagged_block(completion, "action")
    if block is None:
        return None
    _, content_start, close_start, _, content = block
    raw_content = completion[content_start:close_start]
    leading_ws = len(raw_content) - len(raw_content.lstrip())
    trailing_ws = len(raw_content) - len(raw_content.rstrip())
    start = content_start + leading_ws
    end = close_start - trailing_ws
    if end <= start or not content:
        return None
    return start, end


def _proposal_policy_span(trace: ProposalGRPOTrace, *, span_mode: str) -> Optional[Tuple[int, int]]:
    if span_mode == "reasoning_action":
        return None
    if span_mode != "action_only":
        raise ValueError(f"Unsupported proposal_grpo_span={span_mode!r}.")
    tagged_span = _tagged_action_json_span(trace.completion)
    if tagged_span is not None:
        return tagged_span
    start = trace.completion.find("{")
    end = trace.completion.rfind("}")
    if start >= 0 and end > start:
        return start, end + 1
    return None


def _completion_loss(model: AutoModelForCausalLM, batch: Optional[Mapping[str, torch.Tensor]]) -> torch.Tensor:
    import torch

    if batch is None:
        device = next(model.parameters()).device
        return torch.zeros((), dtype=torch.float32, device=device)
    return -_proposal_completion_mean_logprobs(model, batch).mean()


def _completion_loss_values(
    model: AutoModelForCausalLM,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    return -_proposal_completion_mean_logprobs(model, batch)


def _trace_prompt_text(trace: Any, fallback: str) -> str:
    if hasattr(trace, "prompt"):
        return str(trace.prompt())
    if hasattr(trace, "prompt_text"):
        return str(trace.prompt_text)
    if isinstance(trace, Mapping):
        return str(trace.get("prompt") or trace.get("prompt_text") or fallback)
    return fallback


def _trace_completion(trace: Any) -> str:
    if hasattr(trace, "target"):
        return str(trace.target())
    if hasattr(trace, "completion"):
        return str(trace.completion)
    if isinstance(trace, Mapping):
        return str(trace.get("completion") or trace.get("target") or "")
    return ""


def _finite_metric_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _round_observation_float(value: Any, digits: int = 6) -> Optional[float]:
    numeric = _finite_metric_float(value)
    if numeric is None:
        return None
    return round(numeric, digits)


def _prediction_observation_error(
    *,
    prediction: Optional[Mapping[str, Any]],
    key: str,
    realized: Optional[float],
) -> Optional[float]:
    if prediction is None or realized is None:
        return None
    expected = _finite_metric_float(prediction.get(key))
    if expected is None and key == "expected_avg_delta_from_init":
        expected = _finite_metric_float(prediction.get("expected_final_delta_from_init"))
    if expected is None:
        return None
    return float(realized) - expected


def _compact_observation_map(payload: Any) -> JsonDict:
    if not isinstance(payload, Mapping):
        return {}
    compact: JsonDict = {}
    for key, value in sorted(payload.items(), key=lambda item: int(item[0])):
        numeric = _round_observation_float(value)
        if numeric is not None:
            compact[str(int(key))] = numeric
    return compact


def _proposal_from_metadata(payload: Any) -> Optional[ConfigProposal]:
    if not isinstance(payload, Mapping):
        return None
    try:
        left = int(payload["left"])
        right = int(payload["right"])
        target = int(payload.get("target", left + right))
    except (KeyError, TypeError, ValueError):
        return None
    return ConfigProposal(
        left=left,
        right=right,
        guard=str(payload.get("guard", "none")),
        target=target,
        notes=str(payload.get("notes") or ""),
    )


def _observation_trace_prompt(prompt_text: str, trace_completion: str) -> str:
    return (
        f"{prompt_text}\n\n"
        "Assistant trace:\n"
        f"{trace_completion}\n\n"
        "Environment observation:\n"
    )


def _realized_observation_payload(trace: ProposalGRPOTrace) -> Optional[JsonDict]:
    metadata = trace.metadata
    reward_source = str(metadata.get("reward_source") or "")
    payload: JsonDict = {
        "valid": bool(trace.valid),
        "trained": reward_source in {"outcome", "rank_outcome"},
        "validation_category": trace.validation_category,
    }
    if trace.validation_message and not trace.valid:
        payload["message"] = trace.validation_message[:160]
    proposal = _proposal_from_metadata(trace.metadata.get("parsed_proposal"))
    if proposal is not None:
        payload["target"] = int(proposal.target)
    realized_values: Dict[str, Optional[float]] = {}
    for output_key, metadata_key in (
        ("frontier_delta", "frontier_delta"),
        ("target_delta", "target_delta"),
        ("avg_delta_from_init", "final_accuracy_delta"),
        ("avg_delta_from_current", "final_accuracy_delta_from_current"),
        ("avg_accuracy", "final_accuracy"),
    ):
        value = _round_observation_float(metadata.get(metadata_key))
        realized_values[output_key] = value
        if value is not None:
            payload[output_key] = value
    prediction = metadata.get("proposal_prediction") or metadata.get("parsed_prediction")
    if isinstance(prediction, Mapping):
        prediction_payload = dict(prediction)
        payload["prediction"] = sanitize_json_value(prediction_payload)
        for output_key, prediction_key in (
            ("frontier_delta_error", "expected_frontier_delta"),
            ("target_delta_error", "expected_target_delta"),
            ("avg_delta_from_current_error", "expected_avg_delta_from_current"),
            ("avg_delta_from_init_error", "expected_avg_delta_from_init"),
        ):
            realized_key = output_key.removesuffix("_error")
            error = _round_observation_float(
                _prediction_observation_error(
                    prediction=prediction_payload,
                    key=prediction_key,
                    realized=realized_values.get(realized_key),
                )
            )
            if error is not None:
                payload[output_key] = error
    per_size_delta = _compact_observation_map(metadata.get("per_size_delta"))
    if per_size_delta:
        payload["delta_per_size"] = per_size_delta
    per_size_accuracy = _compact_observation_map(metadata.get("per_size_accuracy"))
    if per_size_accuracy:
        payload["accuracy_per_size"] = per_size_accuracy
    failure_reason = metadata.get("failure_reason")
    if failure_reason:
        payload["failure"] = str(failure_reason)[:160]
    return sanitize_json_value(payload)


def _realized_observation_completion(trace: ProposalGRPOTrace) -> Optional[str]:
    payload = _realized_observation_payload(trace)
    if payload is None:
        return None
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _build_format_replay_rows(
    *,
    traces: Sequence[ProposalGRPOTrace],
    proposal_trace_buffer: Sequence[Any],
    fallback_prompt: str,
    max_examples: int,
) -> List[JsonDict]:
    if max_examples <= 0:
        return []
    seen_completions: set[str] = set()
    current_rows: List[JsonDict] = []
    for trace in traces:
        if not trace.valid or not trace.completion or trace.completion in seen_completions:
            continue
        seen_completions.add(trace.completion)
        current_rows.append({"prompt": trace.prompt_text, "completion": trace.completion, "source": "current_valid"})
    remaining = max(0, int(max_examples) - len(current_rows))
    history_rows: List[JsonDict] = []
    if remaining > 0:
        for trace in list(proposal_trace_buffer)[-remaining:]:
            completion = _trace_completion(trace)
            if completion and completion not in seen_completions:
                seen_completions.add(completion)
                history_rows.append(
                    {
                        "prompt": _trace_prompt_text(trace, fallback_prompt),
                        "completion": completion,
                        "source": "selected_trace_buffer",
                    }
                )
    return (current_rows + history_rows)[: int(max_examples)]


def _collate_optional_proposal_samples(
    *,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    device: torch.device,
) -> Optional[JsonDict]:
    if not samples:
        return None
    return _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=samples, device=device)


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


def _proposal_sample_batches(samples: Sequence[JsonDict], microbatch_size: int) -> Sequence[Sequence[JsonDict]]:
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive.")
    return [samples[index : index + microbatch_size] for index in range(0, len(samples), microbatch_size)]


def _proposal_completion_logprobs(
    model: AutoModelForCausalLM,
    batch: Mapping[str, torch.Tensor],
    *,
    normalize_by_length: bool,
) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    mask = batch["completion_mask"][:, : labels.shape[1]]
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    token_logprobs = -token_losses
    masked_logprobs = token_logprobs * mask.float()
    summed_logprobs = masked_logprobs.sum(dim=1)
    if not normalize_by_length:
        return summed_logprobs
    token_counts = mask.sum(dim=1).clamp_min(1).float()
    return summed_logprobs / token_counts


def _proposal_completion_mean_logprobs(model: AutoModelForCausalLM, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return _proposal_completion_logprobs(model, batch, normalize_by_length=True)


def _proposal_completion_mean_logprobs_for_samples(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    device: torch.device,
    microbatch_size: int,
    normalize_by_length: bool = True,
) -> torch.Tensor:
    import torch

    if not samples:
        return torch.empty(0, dtype=torch.float32, device=device)
    chunks: List[torch.Tensor] = []
    for sample_batch in _proposal_sample_batches(samples, microbatch_size):
        batch = _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=sample_batch, device=device)
        chunks.append(_proposal_completion_logprobs(model, batch, normalize_by_length=normalize_by_length))
    return torch.cat(chunks, dim=0) if chunks else torch.empty(0, dtype=torch.float32, device=device)


def _backward_policy_microbatches(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    advantages: torch.Tensor,
    old_logprobs: torch.Tensor,
    anchor_logprobs: Optional[torch.Tensor],
    device: torch.device,
    microbatch_size: int,
    kl_coef: float,
    anchor_kl_coef: float,
    normalize_by_length: bool,
) -> JsonDict:
    import torch

    denominator = max(1, len(samples))
    policy_sum = torch.zeros((), dtype=torch.float32, device=device)
    kl_sum = torch.zeros((), dtype=torch.float32, device=device)
    anchor_kl_sum = torch.zeros((), dtype=torch.float32, device=device)
    mean_after_sum = torch.zeros((), dtype=torch.float32, device=device)
    offset = 0
    for sample_batch in _proposal_sample_batches(samples, microbatch_size):
        batch_size = len(sample_batch)
        batch = _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=sample_batch, device=device)
        new_logprobs = _proposal_completion_logprobs(
            model,
            batch,
            normalize_by_length=normalize_by_length,
        )
        batch_advantages = advantages[offset : offset + batch_size]
        batch_old_logprobs = old_logprobs[offset : offset + batch_size]
        policy_terms = -(batch_advantages * new_logprobs)
        kl_terms = (new_logprobs - batch_old_logprobs) ** 2
        loss_terms = policy_terms.sum() + float(kl_coef) * kl_terms.sum()
        if anchor_logprobs is not None and float(anchor_kl_coef) > 0.0:
            batch_anchor_logprobs = anchor_logprobs[offset : offset + batch_size]
            anchor_kl_terms = (new_logprobs - batch_anchor_logprobs) ** 2
            loss_terms = loss_terms + float(anchor_kl_coef) * anchor_kl_terms.sum()
            anchor_kl_sum = anchor_kl_sum + anchor_kl_terms.detach().sum()
        (loss_terms / denominator).backward()
        policy_sum = policy_sum + policy_terms.detach().sum()
        kl_sum = kl_sum + kl_terms.detach().sum()
        mean_after_sum = mean_after_sum + new_logprobs.detach().sum()
        offset += batch_size
    return {
        "policy_loss": float((policy_sum / denominator).detach().cpu()),
        "kl_proxy": float((kl_sum / denominator).detach().cpu()),
        "anchor_kl_proxy": float((anchor_kl_sum / denominator).detach().cpu()),
        "mean_logprob_after": float((mean_after_sum / denominator).detach().cpu()),
    }


def _backward_completion_loss_microbatches(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: Sequence[JsonDict],
    device: torch.device,
    microbatch_size: int,
    loss_weight: float,
) -> float:
    import torch

    if not samples:
        return 0.0
    denominator = len(samples)
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    for sample_batch in _proposal_sample_batches(samples, microbatch_size):
        batch = _collate_proposal_grpo_samples(tokenizer=tokenizer, samples=sample_batch, device=device)
        loss_values = _completion_loss_values(model, batch)
        loss_values.sum().mul(float(loss_weight) / denominator).backward()
        loss_sum = loss_sum + loss_values.detach().sum()
    return float((loss_sum / denominator).detach().cpu())


def apply_proposal_grpo_update(
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    proposal_kl_reference_checkpoint: Optional[str] = None,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[Any],
    seed: int,
    proposal_trace_buffer: Sequence[Any] = (),
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
        "objective": str(getattr(args, "proposal_grpo_objective", "grpo")),
        "learning_rate": float(args.proposal_grpo_learning_rate),
        "kl_coef": float(args.proposal_grpo_kl_coef),
        "anchor_kl_coef": float(getattr(args, "proposal_grpo_anchor_kl_coef", 0.0)),
        "anchor_kl_reference": str(getattr(args, "proposal_grpo_anchor_kl_reference", "none")),
        "anchor_kl_reference_checkpoint": proposal_kl_reference_checkpoint,
        "anchor_kl_enabled": False,
        "anchor_kl_skip_reason": None,
        "grad_clip": float(args.proposal_grpo_grad_clip),
        "zero_variance": args.proposal_grpo_zero_variance,
        "fixed_baseline": float(args.proposal_grpo_fixed_baseline),
        "reward_mode": args.proposal_grpo_reward_mode,
        "outcome_scale": float(args.proposal_grpo_outcome_scale),
        "novelty_bonus_beta": float(getattr(args, "proposal_grpo_novelty_bonus_beta", 0.0)),
        "proposal_grpo_span": str(getattr(args, "proposal_grpo_span", "reasoning_action")),
        "candidate_metric_count": len(candidate_metrics),
        "loss_mode": getattr(args, "proposal_update_loss_mode", "legacy_grpo"),
        "observation_loss_weight": float(getattr(args, "proposal_observation_loss_weight", 0.0)),
        "format_loss_weight": float(getattr(args, "proposal_format_loss_weight", 0.0)),
        "format_replay_max_examples": int(getattr(args, "proposal_format_replay_max_examples", 0)),
        "format_mask_config_values": bool(getattr(args, "proposal_format_mask_config_values", True)),
        "microbatch_size": int(getattr(args, "proposal_update_microbatch_size", 8)),
        "proposal_trace_buffer_count": len(proposal_trace_buffer),
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
        action_history=proposal_trace_buffer,
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
        loss_mode = str(getattr(args, "proposal_update_loss_mode", "legacy_grpo"))
        span_mode = str(getattr(args, "proposal_grpo_span", "reasoning_action"))
        objective = str(getattr(args, "proposal_grpo_objective", "grpo"))
        if objective not in PROPOSAL_GRPO_OBJECTIVES:
            raise ValueError(f"Unsupported proposal_grpo_objective={objective!r}.")
        normalize_policy_logprobs = objective != "dr_grpo"
        metrics["policy_logprob_normalization"] = "mean" if normalize_policy_logprobs else "sum"
        encoded_samples: List[JsonDict] = []
        encoded_traces: List[ProposalGRPOTrace] = []
        for trace in traces:
            sample = _encode_proposal_grpo_sample(
                tokenizer=tokenizer,
                prompt_text=trace.prompt_text,
                completion=trace.completion,
                completion_char_span=(
                    _proposal_policy_span(trace, span_mode=span_mode) if loss_mode == "merged_agent" else None
                ),
            )
            if sample is None:
                continue
            encoded_samples.append(sample)
            encoded_traces.append(trace)
        metrics["trace_count"] = len(traces)
        metrics["trainable_trace_count"] = len(encoded_samples)
        metrics["completion_token_counts"] = [int(sample["completion_tokens"]) for sample in encoded_samples]
        metrics["total_completion_token_counts"] = [
            int(sample["total_completion_tokens"]) for sample in encoded_samples
        ]
        if not encoded_samples:
            metrics["skip_reason"] = "no_tokenizable_completions"
            write_json(output_dir / "proposal_grpo_metrics.json", metrics)
            return source_checkpoint, metrics

        observation_rows: List[JsonDict] = []
        observation_samples: List[JsonDict] = []
        format_rows: List[JsonDict] = []
        format_samples: List[JsonDict] = []
        if loss_mode == "merged_agent":
            format_loss_weight = float(getattr(args, "proposal_format_loss_weight", 0.0))
            for trace in traces:
                completion = _realized_observation_completion(trace)
                if not completion:
                    continue
                observation_prompt = _observation_trace_prompt(trace.prompt_text, trace.completion)
                sample = _encode_proposal_grpo_sample(
                    tokenizer=tokenizer,
                    prompt_text=observation_prompt,
                    completion=completion,
                    completion_char_span=None,
                )
                if sample is None:
                    continue
                observation_rows.append(
                    {
                        "proposal_index": trace.proposal_index,
                        "prompt": observation_prompt,
                        "trace_completion": trace.completion,
                        "completion": completion,
                        "policy_reward": trace.reward,
                    }
                )
                observation_samples.append(sample)
            if format_loss_weight > 0.0:
                format_rows = _build_format_replay_rows(
                    traces=traces,
                    proposal_trace_buffer=proposal_trace_buffer,
                    fallback_prompt=prompt.text(),
                    max_examples=int(getattr(args, "proposal_format_replay_max_examples", 0)),
                )
                mask_format_values = bool(getattr(args, "proposal_format_mask_config_values", True))
                for row in format_rows:
                    completion = str(row["completion"])
                    exclude_spans = _proposal_format_value_char_spans(completion) if mask_format_values else ()
                    sample = _encode_proposal_grpo_sample(
                        tokenizer=tokenizer,
                        prompt_text=str(row["prompt"]),
                        completion=completion,
                        completion_char_span=None,
                        completion_char_exclude_spans=exclude_spans,
                    )
                    if sample is not None:
                        sample["masked_value_tokens"] = int(sample["total_completion_tokens"]) - int(
                            sample["completion_tokens"]
                        )
                        format_samples.append(sample)
        if observation_rows:
            write_trace_jsonl(output_dir / "proposal_observation_targets.jsonl", observation_rows)
        if format_rows:
            write_trace_jsonl(output_dir / "proposal_format_targets.jsonl", format_rows)
        metrics["observation_trace_count"] = len(observation_rows)
        metrics["trainable_observation_trace_count"] = len(observation_samples)
        metrics["format_trace_count"] = len(format_rows)
        metrics["trainable_format_trace_count"] = len(format_samples)
        metrics["format_masked_value_token_counts"] = [
            int(sample.get("masked_value_tokens", 0)) for sample in format_samples
        ]
        metrics["format_completion_token_counts"] = [int(sample["completion_tokens"]) for sample in format_samples]
        metrics["format_total_completion_token_counts"] = [
            int(sample["total_completion_tokens"]) for sample in format_samples
        ]
        microbatch_size = int(getattr(args, "proposal_update_microbatch_size", 8))
        metrics["microbatch_size"] = microbatch_size
        metrics["policy_microbatch_count"] = len(_proposal_sample_batches(encoded_samples, microbatch_size))
        metrics["observation_microbatch_count"] = len(_proposal_sample_batches(observation_samples, microbatch_size))
        metrics["format_microbatch_count"] = len(_proposal_sample_batches(format_samples, microbatch_size))
        advantages = torch.tensor(
            [trace.advantage for trace in encoded_traces],
            dtype=torch.float32,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            old_logprobs = _proposal_completion_mean_logprobs_for_samples(
                model=model,
                tokenizer=tokenizer,
                samples=encoded_samples,
                device=device,
                microbatch_size=microbatch_size,
                normalize_by_length=normalize_policy_logprobs,
            ).detach()
        anchor_logprobs: Optional[torch.Tensor] = None
        anchor_reference_mode = str(getattr(args, "proposal_grpo_anchor_kl_reference", "none"))
        anchor_kl_coef = float(getattr(args, "proposal_grpo_anchor_kl_coef", 0.0))
        if anchor_kl_coef > 0.0 and anchor_reference_mode != "none":
            if not proposal_kl_reference_checkpoint:
                metrics["anchor_kl_skip_reason"] = "missing_reference_checkpoint"
            elif str(proposal_kl_reference_checkpoint) == str(source_checkpoint):
                anchor_logprobs = old_logprobs
                metrics["anchor_kl_enabled"] = True
                metrics["anchor_kl_skip_reason"] = None
                metrics["anchor_kl_reused_old_logprobs"] = True
                metrics["anchor_mean_logprob_reference"] = float(old_logprobs.mean().detach().cpu())
            else:
                anchor_model = None
                anchor_tokenizer = None
                try:
                    anchor_checkpoint_path = Path(str(proposal_kl_reference_checkpoint))
                    anchor_checkpoint_for_load = (
                        str(anchor_checkpoint_path.resolve())
                        if anchor_checkpoint_path.exists()
                        else str(proposal_kl_reference_checkpoint)
                    )
                    metrics["anchor_kl_reference_checkpoint_resolved"] = anchor_checkpoint_for_load
                    anchor_model, anchor_tokenizer = instantiate_model_and_tokenizer(
                        anchor_checkpoint_for_load,
                        bf16=args.bf16,
                        fp16=args.fp16,
                        init_from_scratch=False,
                        tokenizer_mode=args.tokenizer_mode,
                        recipe=args.recipe,
                    )
                    anchor_model.eval()
                    with torch.no_grad():
                        anchor_logprobs = _proposal_completion_mean_logprobs_for_samples(
                            model=anchor_model,
                            tokenizer=tokenizer,
                            samples=encoded_samples,
                            device=next(anchor_model.parameters()).device,
                            microbatch_size=microbatch_size,
                            normalize_by_length=normalize_policy_logprobs,
                        ).detach().to(device)
                    metrics["anchor_kl_enabled"] = True
                    metrics["anchor_kl_skip_reason"] = None
                    metrics["anchor_kl_reused_old_logprobs"] = False
                    metrics["anchor_mean_logprob_reference"] = float(anchor_logprobs.mean().detach().cpu())
                finally:
                    if anchor_model is not None:
                        del anchor_model
                    if anchor_tokenizer is not None:
                        del anchor_tokenizer
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.proposal_grpo_learning_rate))
        loss_history: List[JsonDict] = []
        for step_index in range(int(args.proposal_grpo_steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            policy_metrics = _backward_policy_microbatches(
                model=model,
                tokenizer=tokenizer,
                samples=encoded_samples,
                advantages=advantages,
                old_logprobs=old_logprobs,
                anchor_logprobs=anchor_logprobs,
                device=device,
                microbatch_size=microbatch_size,
                kl_coef=float(args.proposal_grpo_kl_coef),
                anchor_kl_coef=anchor_kl_coef,
                normalize_by_length=normalize_policy_logprobs,
            )
            policy_loss = float(policy_metrics["policy_loss"])
            kl_proxy = float(policy_metrics["kl_proxy"])
            anchor_kl_proxy = float(policy_metrics["anchor_kl_proxy"])
            mean_logprob_after = float(policy_metrics["mean_logprob_after"])
            observation_loss = 0.0
            format_loss = 0.0
            if loss_mode != "merged_agent":
                observation_samples_for_loss: Sequence[JsonDict] = []
                format_samples_for_loss: Sequence[JsonDict] = []
            else:
                observation_samples_for_loss = observation_samples
                format_samples_for_loss = format_samples
            if observation_samples_for_loss:
                observation_loss = _backward_completion_loss_microbatches(
                    model=model,
                    tokenizer=tokenizer,
                    samples=observation_samples_for_loss,
                    device=device,
                    microbatch_size=microbatch_size,
                    loss_weight=float(getattr(args, "proposal_observation_loss_weight", 0.0)),
                )
            if format_samples_for_loss:
                format_loss = _backward_completion_loss_microbatches(
                    model=model,
                    tokenizer=tokenizer,
                    samples=format_samples_for_loss,
                    device=device,
                    microbatch_size=microbatch_size,
                    loss_weight=float(getattr(args, "proposal_format_loss_weight", 0.0)),
                )
            loss = (
                policy_loss
                + float(args.proposal_grpo_kl_coef) * kl_proxy
                + anchor_kl_coef * anchor_kl_proxy
                + float(getattr(args, "proposal_observation_loss_weight", 0.0)) * observation_loss
                + float(getattr(args, "proposal_format_loss_weight", 0.0)) * format_loss
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.proposal_grpo_grad_clip))
            optimizer.step()
            loss_history.append(
                sanitize_json_value(
                    {
                        "step": step_index + 1,
                        "loss": float(loss),
                        "policy_loss": policy_loss,
                        "observation_loss": observation_loss,
                        "format_loss": format_loss,
                        "kl_proxy": kl_proxy,
                        "anchor_kl_proxy": anchor_kl_proxy,
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                        "mean_logprob_before": float(old_logprobs.mean().detach().cpu()),
                        "anchor_mean_logprob_reference": (
                            float(anchor_logprobs.mean().detach().cpu()) if anchor_logprobs is not None else None
                        ),
                        "mean_logprob_after": mean_logprob_after,
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
    proposal_kl_reference_checkpoint: Optional[str] = None,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[CandidateMetrics],
    seed: int,
    deps: ProposalGrpoDispatchDeps,
    proposal_trace_buffer: Sequence[Any] = (),
) -> tuple[str, JsonDict]:
    if args.controller_execution_mode != "slurm":
        return deps.apply_proposal_grpo_update(
            args=args,
            source_checkpoint=source_checkpoint,
            proposal_kl_reference_checkpoint=proposal_kl_reference_checkpoint,
            output_dir=output_dir,
            prompt=prompt,
            proposal_results=proposal_results,
            candidate_metrics=candidate_metrics,
            proposal_trace_buffer=proposal_trace_buffer,
            seed=seed,
        )
    deps.ensure_dir(output_dir)
    prompt_path = output_dir / "proposal_prompt.json"
    proposal_results_path = output_dir / "proposal_results.json"
    candidate_metrics_path = output_dir / "candidate_metrics.json"
    proposal_trace_buffer_path = output_dir / "proposal_trace_buffer.json"
    deps.write_json(prompt_path, {"system": prompt.system, "user": prompt.user})
    deps.write_json(proposal_results_path, proposal_results)
    deps.write_json(candidate_metrics_path, [metric.to_json_dict() for metric in candidate_metrics])
    deps.write_json(
        proposal_trace_buffer_path,
        [
            trace.to_json_dict() if hasattr(trace, "to_json_dict") else dict(trace)
            for trace in proposal_trace_buffer
        ],
    )
    worker_output = deps.run_controller_worker_slurm(
        args=args,
        worker_dir=output_dir / "controller_worker",
        phase=PHASE_PROPOSAL_GRPO,
        payload={
            "source_checkpoint": source_checkpoint,
            "proposal_kl_reference_checkpoint": proposal_kl_reference_checkpoint,
            "proposal_grpo_dir": str(output_dir),
            "prompt_path": str(prompt_path),
            "proposal_results_path": str(proposal_results_path),
            "candidate_metrics_path": str(candidate_metrics_path),
            "proposal_trace_buffer_path": str(proposal_trace_buffer_path),
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
    row_current_final_accuracy = row.get("current_final_accuracy")
    current_final_accuracy_value = (
        None if row_current_final_accuracy is None else float(row_current_final_accuracy)
    )
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
    final_accuracy_delta_from_current = float(
        row.get(
            "final_accuracy_delta_from_current",
            (
                final_accuracy - current_final_accuracy_value
                if current_final_accuracy_value is not None
                else final_accuracy_delta
            ),
        )
    )
    if "reward" in row:
        reward = float(row["reward"])
    else:
        reward = final_accuracy_delta_from_current
    return {
        "frontier_delta": frontier_delta,
        "final_accuracy": final_accuracy,
        "current_final_accuracy": current_final_accuracy_value,
        "init_final_accuracy": init_final_accuracy_value,
        "final_accuracy_delta": final_accuracy_delta,
        "final_accuracy_delta_from_current": final_accuracy_delta_from_current,
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
    "PROPOSAL_GRPO_OBJECTIVES",
    "PROPOSAL_GRPO_SPAN_MODES",
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
    "write_trace_jsonl",
]

#!/usr/bin/env python3
"""Proposal prompts, schemas, and trace helpers for adaptive composition."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]
PROPOSAL_OUTPUT_SCHEMAS = ("plain", "action_prediction")


@dataclass(frozen=True)
class PromptBundle:
    system: str
    user: str

    def text(self) -> str:
        return f"System:\n{self.system}\n\nUser:\n{self.user}"


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
class ProgramProposal:
    proposal_type: str
    task: str
    code: str

    def to_json_dict(self) -> JsonDict:
        return asdict(self)

    def to_completion(self) -> str:
        return self.code


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
    proposal: "ConfigProposal",
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


def proposal_payload_for_completion(proposal: "ConfigProposal") -> JsonDict:
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
    proposal: "ConfigProposal",
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

"""Config proposal schemas, parsing, and normalization helpers."""

from __future__ import annotations

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

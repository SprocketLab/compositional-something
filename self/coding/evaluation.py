"""Behavioral evaluators for atomic coding tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from self.coding.atomic_data import AtomicExample, apply_scalar_patch


@dataclass(frozen=True)
class ExampleEvaluation:
    exact: bool
    format_valid: bool
    behavior_valid: bool
    parsed_prediction: Any
    error: Optional[str]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exact": self.exact,
            "format_valid": self.format_valid,
            "behavior_valid": self.behavior_valid,
            "parsed_prediction": self.parsed_prediction,
            "error": self.error,
            "details": self.details,
        }


def parse_strict_json_array(text: str) -> Tuple[Optional[List[Any]], Optional[str]]:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return None, "response is not a bare JSON array"
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc.msg}"
    if not isinstance(parsed, list):
        return None, "response is not a JSON array"
    return parsed, None


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _value_matches_schema(value: Any, schema: Mapping[str, Any]) -> bool:
    expected = schema.get("type")
    if isinstance(expected, list):
        return any(_value_matches_schema(value, {**schema, "type": member}) for member in expected)
    if expected in (None, "any"):
        return True
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        return item_schema is None or all(_value_matches_schema(item, item_schema) for item in value)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _bfcl_call_schema_valid(call: Any, functions: Mapping[str, Mapping[str, Any]]) -> bool:
    if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
        return False
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, dict) or name not in functions:
        return False
    parameters = functions[name].get("parameters", {})
    properties = parameters.get("properties", {})
    required = parameters.get("required", [])
    if any(argument not in arguments for argument in required):
        return False
    if any(argument not in properties for argument in arguments):
        return False
    return all(_value_matches_schema(value, properties.get(argument, {})) for argument, value in arguments.items())


def _bfcl_call_matches(predicted: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    if predicted.get("name") != reference.get("name"):
        return False
    predicted_arguments = predicted.get("arguments")
    accepted_arguments = reference.get("arguments")
    if not isinstance(predicted_arguments, dict) or not isinstance(accepted_arguments, dict):
        return False
    if any(key not in accepted_arguments for key in predicted_arguments):
        return False
    for key, options in accepted_arguments.items():
        option_list = options if isinstance(options, list) else [options]
        if key not in predicted_arguments:
            if "" not in option_list:
                return False
            continue
        if not any(option != "" and _json_equal(predicted_arguments[key], option) for option in option_list):
            return False
    return True


def _calls_match_unordered(predicted: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]) -> bool:
    if len(predicted) != len(references):
        return False
    used = [False] * len(references)

    def search(index: int) -> bool:
        if index == len(predicted):
            return True
        for reference_index, reference in enumerate(references):
            if not used[reference_index] and _bfcl_call_matches(predicted[index], reference):
                used[reference_index] = True
                if search(index + 1):
                    return True
                used[reference_index] = False
        return False

    return search(0)


def evaluate_bfcl(example: AtomicExample, prediction: str) -> ExampleEvaluation:
    parsed, error = parse_strict_json_array(prediction)
    if parsed is None:
        return ExampleEvaluation(False, False, False, None, error, {"schema_valid": False})
    functions = {
        str(function["name"]): function
        for function in example.evaluator.get("functions", [])
    }
    schema_valid = all(_bfcl_call_schema_valid(call, functions) for call in parsed)
    exact = schema_valid and _calls_match_unordered(parsed, example.evaluator.get("accepted_calls", []))
    return ExampleEvaluation(
        exact=exact,
        format_valid=True,
        behavior_valid=schema_valid,
        parsed_prediction=parsed,
        error=None if exact else ("accepted-set mismatch" if schema_valid else "schema mismatch"),
        details={"schema_valid": schema_valid, "predicted_call_count": len(parsed)},
    )


def evaluate_commitpack(example: AtomicExample, prediction: str) -> ExampleEvaluation:
    parsed, error = parse_strict_json_array(prediction)
    if parsed is None:
        return ExampleEvaluation(False, False, False, None, error, {"patch_applicable": False})
    structurally_valid = True
    for operation in parsed:
        if not isinstance(operation, dict):
            structurally_valid = False
            break
        op = operation.get("op")
        required = {"op", "path"} if op == "remove" else {"op", "path", "value"}
        if op not in {"add", "remove", "replace"} or set(operation) != required or not isinstance(operation.get("path"), str):
            structurally_valid = False
            break
    if not structurally_valid:
        return ExampleEvaluation(
            False,
            True,
            False,
            parsed,
            "invalid JSON Patch structure",
            {"patch_applicable": False},
        )
    try:
        result = apply_scalar_patch(example.evaluator["old_document"], parsed)
    except Exception as exc:
        return ExampleEvaluation(
            False,
            True,
            False,
            parsed,
            str(exc),
            {"patch_applicable": False},
        )
    exact = result == example.evaluator["intended_document"]
    return ExampleEvaluation(
        exact=exact,
        format_valid=True,
        behavior_valid=True,
        parsed_prediction=parsed,
        error=None if exact else "patched document differs from intended state",
        details={"patch_applicable": True, "operation_count": len(parsed)},
    )


def evaluate_example(example: AtomicExample, prediction: str) -> ExampleEvaluation:
    if example.task == "bfcl":
        return evaluate_bfcl(example, prediction)
    if example.task == "commitpack":
        return evaluate_commitpack(example, prediction)
    raise ValueError(f"Unsupported coding task: {example.task!r}")


def evaluate_predictions(
    examples: Sequence[AtomicExample],
    predictions: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal lengths")
    rows: List[Dict[str, Any]] = []
    evaluations: List[ExampleEvaluation] = []
    for example, prediction in zip(examples, predictions):
        evaluation = evaluate_example(example, prediction)
        evaluations.append(evaluation)
        rows.append(
            {
                "source_id": example.source_id,
                "source_group_id": example.source_group_id,
                "component_count": example.component_count,
                "prediction": prediction,
                "target": example.target,
                **evaluation.to_dict(),
            }
        )
    count = len(evaluations)
    denominator = max(count, 1)
    summary = {
        "count": count,
        "exact_accuracy": sum(item.exact for item in evaluations) / denominator,
        "format_accuracy": sum(item.format_valid for item in evaluations) / denominator,
        "behavior_valid_accuracy": sum(item.behavior_valid for item in evaluations) / denominator,
        "error_counts": {},
    }
    error_counts: Dict[str, int] = {}
    for item in evaluations:
        if item.error:
            error_counts[item.error] = error_counts.get(item.error, 0) + 1
    summary["error_counts"] = dict(sorted(error_counts.items(), key=lambda pair: (-pair[1], pair[0])))
    return summary, rows

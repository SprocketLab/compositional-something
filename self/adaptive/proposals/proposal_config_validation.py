"""Config proposal row validation."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Mapping, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.proposals.proposal_config_schema import (
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

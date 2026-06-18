"""Dry-run proposal processing for the adaptive proposal pilot."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.core.proposal_config_schema import DEFAULT_CONFIG_SEARCH_SPACES, parse_config_proposal
from self.core.proposal_io import build_trace_row
from self.core.proposal_prompts import (
    PromptBundle,
    render_config_prompt,
    render_program_prompt,
    render_program_repair_prompt,
)
from self.core.program_sandbox import validate_program_with_repair
from self.core.program_sandbox_cases import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.core.program_sandbox_models import ProgramValidationResult


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


#!/usr/bin/env python3
"""Pilot controller for SEAL-style adaptive composition proposals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from self.core.frontier import (
    load_diagnostics_payload,
    proposal_quality_metrics,
    select_frontier,
)
from self.core.proposal_config_schema import DEFAULT_CONFIG_SEARCH_SPACES, parse_config_proposal
from self.core.proposals import (
    PromptBundle,
    render_config_prompt,
    render_program_prompt,
    render_program_repair_prompt,
)
from self.core.proposal_io import build_trace_row, load_fixture_proposals, write_trace_jsonl
from self.core.program_sandbox import validate_program_with_repair
from self.core.program_sandbox_cases import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.core.program_sandbox_models import (
    ProgramValidationResult,
)
from self.core.data_io import ensure_dir, sanitize_json_value


JsonDict = Dict[str, Any]

TASK_CHOICES = ("addition", "run_length")
CONDITION_CHOICES = ("config", "program")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive composition proposal pilot controller.")
    parser.add_argument("--task", choices=TASK_CHOICES, required=True)
    parser.add_argument("--condition", choices=CONDITION_CHOICES, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs/adaptive_self_improvement_pilot"))
    parser.add_argument("--proposal-fixture-jsonl", type=Path, required=True)
    parser.add_argument(
        "--dry-run-proposals",
        action="store_true",
        help="Validate/rank fixture proposals and write artifacts without launching model generation or training.",
    )
    parser.add_argument("--round-index", type=int, default=1)
    parser.add_argument(
        "--current-frontier-min",
        type=int,
        default=0,
        help="Deprecated alias for --source-min-allowed.",
    )
    parser.add_argument(
        "--current-frontier-max",
        type=int,
        default=0,
        help="Deprecated alias for --source-max-allowed.",
    )
    parser.add_argument("--source-min-allowed", type=int, default=None)
    parser.add_argument("--source-max-allowed", type=int, default=None)
    parser.add_argument("--frontier-min-allowed", type=int, default=1)
    parser.add_argument("--frontier-max-allowed", type=int, default=32)
    parser.add_argument(
        "--frontier-policy",
        choices=("fixed", "weak_regime"),
        default="fixed",
        help="fixed uses the passed allowed range; weak_regime selects a low-accuracy size/slice from diagnostics.",
    )
    parser.add_argument(
        "--frontier-diagnostics-json",
        type=str,
        default=None,
        help="Aggregate diagnostics JSON used by --frontier-policy weak_regime.",
    )
    parser.add_argument(
        "--frontier-diagnostics-path",
        type=Path,
        default=None,
        help="Path to a summary JSON/object/list used by --frontier-policy weak_regime.",
    )
    parser.add_argument("--frontier-min-count", type=int, default=1)
    parser.add_argument("--frontier-max-accuracy", type=float, default=0.85)
    parser.add_argument("--frontier-max-width", type=int, default=1)
    parser.add_argument("--frontier-prefer-larger-weight", type=float, default=0.01)
    parser.add_argument(
        "--enforce-selected-frontier",
        action="store_true",
        help="Constrain config proposals to the selected weak-regime frontier min/max.",
    )
    parser.add_argument("--min-examples-per-size", type=int, default=1)
    parser.add_argument("--max-examples-per-size", type=int, default=5000)
    parser.add_argument("--lambda-final", type=float, default=0.1)
    parser.add_argument(
        "--init-final-accuracy",
        type=float,
        default=None,
        help=(
            "Init-time final/global accuracy baseline. Reward uses "
            "final_accuracy - init_final_accuracy instead of absolute final_accuracy."
        ),
    )
    parser.add_argument("--max-traces", type=int, default=2)
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--program-timeout-seconds", type=float, default=1.0)
    parser.add_argument(
        "--aggregate-metrics-json",
        type=str,
        default="{}",
        help="Aggregate diagnostics included in the proposal prompt. Must not contain per-example oracle labels.",
    )
    parser.add_argument(
        "--plan-log-path",
        type=Path,
        default=Path("plan/260603-self-improvement-init.md"),
        help="Plan document to append implementation/pilot logs to.",
    )
    return parser


def _load_json_arg(raw: str, *, name: str) -> JsonDict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _resolve_init_final_accuracy(
    args: argparse.Namespace,
    aggregate_metrics: Mapping[str, Any],
) -> Optional[float]:
    if args.init_final_accuracy is not None:
        return float(args.init_final_accuracy)
    for key in (
        "init_final_accuracy",
        "initial_final_accuracy",
        "init_time_final_accuracy",
        "baseline_final_accuracy",
    ):
        value = aggregate_metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _reward(
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


def _target_format(task: str) -> str:
    if task == "run_length":
        return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
    if task == "addition":
        return "integer result string formed from component prediction strings"
    raise ValueError(f"Unsupported task={task!r}")


def _component_prediction_examples(task: str) -> List[str]:
    if task == "run_length":
        return ["3|0|2|2|3", "5|1|5|1|5"]
    if task == "addition":
        return ["46", "064", "1002"]
    raise ValueError(f"Unsupported task={task!r}")


def _program_cases(task: str):
    if task == "run_length":
        return build_run_length_program_cases(random_seed=0, random_count=8)
    if task == "addition":
        return build_addition_program_cases()
    raise ValueError(f"Unsupported task={task!r}")


def _raw_output(row: Mapping[str, Any]) -> Any:
    if "code_lines" in row:
        code_lines = row["code_lines"]
        if not isinstance(code_lines, list):
            raise ValueError("code_lines must be a list of strings")
        return "\n".join(str(line) for line in code_lines)
    for key in ("raw_output", "output", "completion", "proposal", "code"):
        if key in row:
            return row[key]
    return row


def _completion_for_trace(parsed: Any, raw: Any) -> str:
    if hasattr(parsed, "to_completion"):
        return parsed.to_completion()
    if isinstance(raw, str):
        return raw
    return json.dumps(raw, sort_keys=True)


def _source_bounds(args: argparse.Namespace) -> tuple[int, int]:
    source_min = args.source_min_allowed
    if source_min is None:
        source_min = args.current_frontier_min if args.current_frontier_min > 0 else 1
    source_max = args.source_max_allowed
    if source_max is None:
        source_max = args.current_frontier_max if args.current_frontier_max > 0 else args.frontier_max_allowed
    return int(source_min), int(source_max)


def _render_prompt(args: argparse.Namespace, aggregate_metrics: Mapping[str, Any]) -> PromptBundle:
    if args.condition == "config":
        space = DEFAULT_CONFIG_SEARCH_SPACES[args.task]
        source_min, source_max = _source_bounds(args)
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
        target_format=_target_format(args.task),
        component_prediction_examples=_component_prediction_examples(args.task),
    )


def _validate_config_row(row: Mapping[str, Any], args: argparse.Namespace) -> JsonDict:
    space = DEFAULT_CONFIG_SEARCH_SPACES[args.task]
    source_min, source_max = _source_bounds(args)
    validation = parse_config_proposal(
        _raw_output(row),
        task_name=args.task,
        source_min_allowed=source_min,
        source_max_allowed=source_max,
        frontier_min_allowed=args.frontier_min_allowed,
        frontier_max_allowed=args.frontier_max_allowed,
        guards=space["guards"],
    )
    reward = _reward(
        row,
        lambda_final=args.lambda_final,
        init_final_accuracy=args.init_final_accuracy,
    )
    completion = _completion_for_trace(validation.proposal, _raw_output(row)) if validation.valid else ""
    return sanitize_json_value(
        {
            "id": row.get("id"),
            "condition": "config",
            "raw_output": _raw_output(row),
            "valid": validation.valid,
            "validation_category": validation.category,
            "validation_message": validation.message,
            "parsed_proposal": validation.proposal.to_json_dict() if validation.valid else None,
            "completion": completion,
            **reward,
        }
    )


def _validate_program_row(row: Mapping[str, Any], args: argparse.Namespace) -> JsonDict:
    code = str(_raw_output(row))
    repair_prompt_text: Optional[str] = None

    def repair_callback(category: str, message: str, previous_program: str) -> Optional[str]:
        nonlocal repair_prompt_text
        repair_prompt = render_program_repair_prompt(
            task_name=args.task,
            target_format=_target_format(args.task),
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
        cases=_program_cases(args.task),
        timeout_seconds=args.program_timeout_seconds,
        repair_attempts=args.repair_attempts,
    )
    reward = _reward(
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


def _process_rows(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[JsonDict]:
    results: List[JsonDict] = []
    seen_completions: set[str] = set()
    for index, row in enumerate(rows):
        row_condition = row.get("condition", args.condition)
        if row_condition != args.condition:
            continue
        result = _validate_config_row(row, args) if args.condition == "config" else _validate_program_row(row, args)
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


def _select_best(results: Sequence[Mapping[str, Any]]) -> Optional[JsonDict]:
    eligible = [dict(result) for result in results if result.get("selection_eligible")]
    if not eligible:
        return None
    eligible.sort(key=lambda result: (float(result["reward"]), float(result["frontier_delta"])), reverse=True)
    selected = eligible[0]
    selected["selected"] = True
    return selected


def _trace_rows(
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)


def _load_frontier_diagnostics(args: argparse.Namespace) -> JsonDict:
    diagnostics: JsonDict = {}
    if args.frontier_diagnostics_path is not None:
        diagnostics.update(load_diagnostics_payload(args.frontier_diagnostics_path))
    if args.frontier_diagnostics_json:
        diagnostics.update(_load_json_arg(args.frontier_diagnostics_json, name="--frontier-diagnostics-json"))
    return diagnostics


def _apply_frontier_policy(args: argparse.Namespace, aggregate_metrics: JsonDict) -> JsonDict:
    diagnostics = _load_frontier_diagnostics(args)
    if not diagnostics:
        diagnostics = dict(aggregate_metrics)
    selection = select_frontier(
        diagnostics,
        task=args.task,
        allowed_min=args.frontier_min_allowed,
        allowed_max=args.frontier_max_allowed,
        policy=args.frontier_policy,
        min_count=args.frontier_min_count,
        max_accuracy=args.frontier_max_accuracy,
        max_width=args.frontier_max_width,
        prefer_larger_weight=args.frontier_prefer_larger_weight,
    )
    if args.enforce_selected_frontier and selection.selected is not None:
        args.frontier_min_allowed = selection.frontier_min()
        args.frontier_max_allowed = selection.frontier_max()
    return selection.to_json_dict()


def _append_plan_log(plan_path: Path, summary: Mapping[str, Any]) -> None:
    if not plan_path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "",
        f"### Implementation Log: {timestamp}",
        "",
        "- Added adaptive proposal pilot artifacts for this run.",
        f"- Task/condition: `{summary['task']}` / `{summary['condition']}`.",
        f"- Output directory: `{summary['output_dir']}`.",
        f"- Valid proposals: {summary['valid_count']} / {summary['proposal_count']}.",
        f"- Trace rows written: {summary['trace_count']}.",
    ]
    if summary.get("selected_id") is not None:
        lines.append(
            f"- Selected proposal: `{summary['selected_id']}` with reward {summary['selected_reward']:.6f}."
        )
    else:
        lines.append("- Selected proposal: none; no valid positive-reward proposal was available.")
    with plan_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def run(args: argparse.Namespace) -> JsonDict:
    if not args.dry_run_proposals:
        raise NotImplementedError(
            "This pilot entrypoint currently validates/ranks generated or fixture proposals only. "
            "Pass --dry-run-proposals for the implemented pilot path."
        )
    aggregate_metrics = _load_json_arg(args.aggregate_metrics_json, name="--aggregate-metrics-json")
    args.init_final_accuracy = _resolve_init_final_accuracy(args, aggregate_metrics)
    if args.init_final_accuracy is not None:
        aggregate_metrics = {
            **aggregate_metrics,
            "init_final_accuracy": args.init_final_accuracy,
            "reward_formula": "frontier_delta + lambda_final * (final_accuracy - init_final_accuracy)",
        }
    frontier_selection = _apply_frontier_policy(args, aggregate_metrics)
    aggregate_metrics = {
        **aggregate_metrics,
        "adaptive_frontier_selection": frontier_selection,
    }
    output_dir = args.output_dir
    ensure_dir(output_dir)

    prompt = _render_prompt(args, aggregate_metrics)
    rows = load_fixture_proposals(args.proposal_fixture_jsonl)
    results = _process_rows(rows, args)
    selected = _select_best(results)
    for result in results:
        result["selected"] = selected is not None and result.get("proposal_index") == selected.get("proposal_index")
    trace_rows = _trace_rows(results=results, prompt=prompt.text(), args=args)
    quality_metrics = proposal_quality_metrics(
        results,
        selected_id=selected.get("id") if selected else None,
    )

    _write_json(output_dir / "proposal_prompt.json", {"system": prompt.system, "user": prompt.user})
    _write_json(output_dir / "frontier_selection.json", frontier_selection)
    _write_json(output_dir / "proposal_results.json", results)
    _write_json(output_dir / "selected_proposal.json", selected)
    _write_json(output_dir / "proposal_quality_metrics.json", quality_metrics)
    write_trace_jsonl(output_dir / "trace_examples.jsonl", trace_rows)

    summary: JsonDict = {
        "task": args.task,
        "condition": args.condition,
        "model_name": args.model_name,
        "proposal_count": len(results),
        "valid_count": sum(1 for result in results if result.get("valid")),
        "positive_count": sum(1 for result in results if result.get("trace_include")),
        "trace_count": len(trace_rows),
        "selected_id": selected.get("id") if selected else None,
        "selected_reward": float(selected["reward"]) if selected else None,
        "selected_proposal_index": selected.get("proposal_index") if selected else None,
        "output_dir": str(output_dir),
        "dry_run_proposals": args.dry_run_proposals,
        "frontier_policy": args.frontier_policy,
        "frontier_selection": frontier_selection,
        "init_final_accuracy": args.init_final_accuracy,
        "reward_formula": "frontier_delta + lambda_final * final_accuracy_delta",
        "proposal_quality_metrics": quality_metrics,
    }
    _write_json(output_dir / "summary.json", summary)
    _append_plan_log(args.plan_log_path, summary)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(sanitize_json_value(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

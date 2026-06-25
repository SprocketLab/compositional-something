#!/usr/bin/env python3
"""Pilot controller for SEAL-style adaptive composition proposals."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from self.adaptive.frontier import (
    load_diagnostics_payload,
    proposal_quality_metrics,
    select_frontier,
)
from self.adaptive.proposal import (
    build_pilot_trace_rows as _trace_rows,
    completion_for_pilot_trace as _completion_for_trace,
    component_prediction_examples_for_pilot_task as _component_prediction_examples,
    pilot_reward as _reward,
    pilot_source_bounds as _source_bounds,
    process_pilot_rows as _process_rows,
    program_cases_for_pilot_task as _program_cases,
    raw_pilot_output as _raw_output,
    render_pilot_prompt as _render_prompt,
    select_best_pilot_result as _select_best,
    target_format_for_pilot_task as _target_format,
    validate_config_pilot_row as _validate_config_row,
    validate_program_pilot_row as _validate_program_row,
)
from self.adaptive.proposal import load_fixture_proposals, write_trace_jsonl
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
        help="Deprecated init-time final/global accuracy baseline retained for old fixture compatibility.",
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
            "reward_formula": "final_accuracy - current_final_accuracy",
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
        "reward_formula": "final_accuracy - current_final_accuracy",
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

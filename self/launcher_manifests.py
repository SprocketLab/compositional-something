#!/usr/bin/env python3
"""Manifest builders used by launcher submitter scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

JsonDict = dict[str, Any]


def _split_words(raw: str) -> list[str]:
    return raw.split()


def _split_int_words(raw: str) -> list[int]:
    return [int(value) for value in raw.split()]


def build_adaptive_candidate_submission_manifest(
    *,
    out_root: str,
    tasks: str,
    conditions: str,
    outcome_trace_target_modes: str,
    proposal_grpo_zero_variance_modes: str,
    num_candidates_list: str,
    adaptive_config_files: str,
    job_fields: Sequence[str],
) -> JsonDict:
    if len(job_fields) % 7 != 0:
        raise ValueError("adaptive candidate job_fields must be groups of 7 values.")

    jobs: JsonDict = {}
    for index in range(0, len(job_fields), 7):
        task, condition, outcome_mode, zero_variance, num_candidates, job_id, output_dir = job_fields[
            index : index + 7
        ]
        jobs[f"{task}-{condition}-{outcome_mode}-n{num_candidates}-grpo-{zero_variance}"] = {
            "task": task,
            "condition": condition,
            "outcome_trace_target_mode": outcome_mode,
            "proposal_grpo_zero_variance": zero_variance,
            "num_candidates": int(num_candidates),
            "job_id": job_id,
            "output_dir": output_dir,
            "status": "submitted",
        }

    return {
        "out_root": out_root,
        "tasks": _split_words(tasks),
        "conditions": _split_words(conditions),
        "outcome_trace_target_modes": _split_words(outcome_trace_target_modes),
        "proposal_grpo_zero_variance_modes": _split_words(proposal_grpo_zero_variance_modes),
        "num_candidates_list": _split_int_words(num_candidates_list),
        "adaptive_config_files": adaptive_config_files,
        "jobs": jobs,
    }


def build_adaptive_condition_submission_manifest(
    *,
    out_root: str,
    partition: str,
    gres: str,
    time_limit: str,
    cpus_per_task: str,
    mem: str,
    frontier_policy: str,
    frontier_min_count: str,
    frontier_max_accuracy: str,
    frontier_max_width: str,
    frontier_prefer_larger_weight: str,
    enforce_selected_frontier: str,
    frontier_diagnostics_path: str,
    addition_config_job_id: str,
    addition_program_job_id: str,
    run_length_config_job_id: str,
    run_length_program_job_id: str,
) -> JsonDict:
    return {
        "out_root": out_root,
        "slurm": {
            "partition": partition,
            "gres": gres,
            "time": time_limit,
            "cpus_per_task": cpus_per_task,
            "mem": mem,
            "frontier_policy": frontier_policy,
            "frontier_min_count": frontier_min_count,
            "frontier_max_accuracy": frontier_max_accuracy,
            "frontier_max_width": frontier_max_width,
            "frontier_prefer_larger_weight": frontier_prefer_larger_weight,
            "enforce_selected_frontier": enforce_selected_frontier,
            "frontier_diagnostics_path": frontier_diagnostics_path or None,
        },
        "jobs": {
            "addition_config": {
                "job_id": addition_config_job_id,
                "task": "addition",
                "condition": "config",
                "output_dir": f"{out_root}/addition-config",
            },
            "addition_program": {
                "job_id": addition_program_job_id,
                "task": "addition",
                "condition": "program",
                "output_dir": f"{out_root}/addition-program",
            },
            "run_length_config": {
                "job_id": run_length_config_job_id,
                "task": "run_length",
                "condition": "config",
                "output_dir": f"{out_root}/run-length-config",
            },
            "run_length_program": {
                "job_id": run_length_program_job_id,
                "task": "run_length",
                "condition": "program",
                "output_dir": f"{out_root}/run-length-program",
            },
        },
        "scope_note": (
            "These are split adaptive proposal/preflight condition jobs. "
            "They do not yet run temporary LoRA self-edit training/evaluation loops."
        ),
    }


def write_manifest(path: Path, payload: JsonDict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write launcher submission manifests.")
    subparsers = parser.add_subparsers(dest="manifest_type", required=True)

    candidate = subparsers.add_parser("adaptive-candidate")
    candidate.add_argument("--manifest", type=Path, required=True)
    candidate.add_argument("--out-root", required=True)
    candidate.add_argument("--tasks", required=True)
    candidate.add_argument("--conditions", required=True)
    candidate.add_argument("--outcome-trace-target-modes", required=True)
    candidate.add_argument("--proposal-grpo-zero-variance-modes", required=True)
    candidate.add_argument("--num-candidates-list", required=True)
    candidate.add_argument("--adaptive-config-files", default="")
    candidate.add_argument("--job-fields", nargs="*", default=[])

    condition = subparsers.add_parser("adaptive-condition")
    condition.add_argument("--manifest", type=Path, required=True)
    condition.add_argument("--out-root", required=True)
    condition.add_argument("--partition", required=True)
    condition.add_argument("--gres", required=True)
    condition.add_argument("--time-limit", required=True)
    condition.add_argument("--cpus-per-task", required=True)
    condition.add_argument("--mem", required=True)
    condition.add_argument("--frontier-policy", required=True)
    condition.add_argument("--frontier-min-count", required=True)
    condition.add_argument("--frontier-max-accuracy", required=True)
    condition.add_argument("--frontier-max-width", required=True)
    condition.add_argument("--frontier-prefer-larger-weight", required=True)
    condition.add_argument("--enforce-selected-frontier", required=True)
    condition.add_argument("--frontier-diagnostics-path", default="")
    condition.add_argument("--addition-config-job-id", required=True)
    condition.add_argument("--addition-program-job-id", required=True)
    condition.add_argument("--run-length-config-job-id", required=True)
    condition.add_argument("--run-length-program-job-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.manifest_type == "adaptive-candidate":
        payload = build_adaptive_candidate_submission_manifest(
            out_root=args.out_root,
            tasks=args.tasks,
            conditions=args.conditions,
            outcome_trace_target_modes=args.outcome_trace_target_modes,
            proposal_grpo_zero_variance_modes=args.proposal_grpo_zero_variance_modes,
            num_candidates_list=args.num_candidates_list,
            adaptive_config_files=args.adaptive_config_files,
            job_fields=args.job_fields,
        )
    elif args.manifest_type == "adaptive-condition":
        payload = build_adaptive_condition_submission_manifest(
            out_root=args.out_root,
            partition=args.partition,
            gres=args.gres,
            time_limit=args.time_limit,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            frontier_policy=args.frontier_policy,
            frontier_min_count=args.frontier_min_count,
            frontier_max_accuracy=args.frontier_max_accuracy,
            frontier_max_width=args.frontier_max_width,
            frontier_prefer_larger_weight=args.frontier_prefer_larger_weight,
            enforce_selected_frontier=args.enforce_selected_frontier,
            frontier_diagnostics_path=args.frontier_diagnostics_path,
            addition_config_job_id=args.addition_config_job_id,
            addition_program_job_id=args.addition_program_job_id,
            run_length_config_job_id=args.run_length_config_job_id,
            run_length_program_job_id=args.run_length_program_job_id,
        )
    else:
        raise AssertionError(f"Unhandled manifest_type={args.manifest_type!r}.")
    write_manifest(args.manifest, payload)


if __name__ == "__main__":
    main()

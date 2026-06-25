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
    model_name: str,
    proposal_model_name: str,
    outcome_trace_target_modes: str,
    proposal_grpo_reward_modes: str,
    proposal_grpo_zero_variance_modes: str,
    num_candidates_list: str,
    proposal_grpo_learning_rates: str,
    proposal_grpo_kl_coef: str,
    synthetic_proposal_sft_examples_list: str,
    synthetic_proposal_sft_seed_mix: str,
    synthetic_proposal_sft_num_epochs: str,
    synthetic_proposal_sft_learning_rate: str,
    synthetic_proposal_sft_top_k: str,
    synthetic_proposal_sft_temperature: str,
    adaptive_config_files: str,
    job_fields: Sequence[str],
) -> JsonDict:
    if len(job_fields) % 12 != 0:
        raise ValueError("adaptive candidate job_fields must be groups of 12 values.")

    jobs: JsonDict = {}
    for index in range(0, len(job_fields), 12):
        (
            task,
            condition,
            outcome_mode,
            reward_mode,
            zero_variance,
            num_candidates,
            proposal_lr,
            proposal_kl,
            synthetic_examples,
            synthetic_seed_mix,
            job_id,
            output_dir,
        ) = job_fields[
            index : index + 12
        ]
        synthetic_suffix = (
            f"seedmix-syn{synthetic_examples}"
            if str(synthetic_seed_mix) == "1"
            else f"syn{synthetic_examples}"
        )
        jobs[
            (
                f"{task}-{condition}-{outcome_mode}-n{num_candidates}-reward-{reward_mode}"
                f"-grpo-{zero_variance}-lr-{proposal_lr}-{synthetic_suffix}"
            )
        ] = {
            "task": task,
            "condition": condition,
            "outcome_trace_target_mode": outcome_mode,
            "proposal_grpo_reward_mode": reward_mode,
            "proposal_grpo_zero_variance": zero_variance,
            "num_candidates": int(num_candidates),
            "proposal_grpo_learning_rate": proposal_lr,
            "proposal_grpo_kl_coef": proposal_kl,
            "synthetic_proposal_sft_examples": int(synthetic_examples),
            "synthetic_proposal_sft_seed_mix": str(synthetic_seed_mix) == "1",
            "job_id": job_id,
            "output_dir": output_dir,
            "status": "submitted",
        }

    return {
        "out_root": out_root,
        "tasks": _split_words(tasks),
        "conditions": _split_words(conditions),
        "model_name": model_name,
        "proposal_model_name": proposal_model_name,
        "outcome_trace_target_modes": _split_words(outcome_trace_target_modes),
        "proposal_grpo_reward_modes": _split_words(proposal_grpo_reward_modes),
        "proposal_grpo_zero_variance_modes": _split_words(proposal_grpo_zero_variance_modes),
        "num_candidates_list": _split_int_words(num_candidates_list),
        "proposal_grpo_learning_rates": _split_words(proposal_grpo_learning_rates),
        "proposal_grpo_kl_coef": proposal_grpo_kl_coef,
        "synthetic_proposal_sft_examples_list": _split_int_words(synthetic_proposal_sft_examples_list),
        "synthetic_proposal_sft_seed_mix": str(synthetic_proposal_sft_seed_mix) == "1",
        "synthetic_proposal_sft_num_epochs": int(synthetic_proposal_sft_num_epochs),
        "synthetic_proposal_sft_learning_rate": synthetic_proposal_sft_learning_rate,
        "synthetic_proposal_sft_top_k": int(synthetic_proposal_sft_top_k),
        "synthetic_proposal_sft_temperature": float(synthetic_proposal_sft_temperature),
        "adaptive_config_files": adaptive_config_files,
        "jobs": jobs,
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
    candidate.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    candidate.add_argument("--proposal-model-name", default="current")
    candidate.add_argument("--outcome-trace-target-modes", required=True)
    candidate.add_argument("--proposal-grpo-reward-modes", required=True)
    candidate.add_argument("--proposal-grpo-zero-variance-modes", required=True)
    candidate.add_argument("--num-candidates-list", required=True)
    candidate.add_argument("--proposal-grpo-learning-rates", required=True)
    candidate.add_argument("--proposal-grpo-kl-coef", required=True)
    candidate.add_argument("--synthetic-proposal-sft-examples-list", default="0")
    candidate.add_argument("--synthetic-proposal-sft-seed-mix", default="0")
    candidate.add_argument("--synthetic-proposal-sft-num-epochs", default="1")
    candidate.add_argument("--synthetic-proposal-sft-learning-rate", default="1e-6")
    candidate.add_argument("--synthetic-proposal-sft-top-k", default="4")
    candidate.add_argument("--synthetic-proposal-sft-temperature", default="0.7")
    candidate.add_argument("--adaptive-config-files", default="")
    candidate.add_argument("--job-fields", nargs="*", default=[])

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.manifest_type == "adaptive-candidate":
        payload = build_adaptive_candidate_submission_manifest(
            out_root=args.out_root,
            tasks=args.tasks,
            conditions=args.conditions,
            model_name=args.model_name,
            proposal_model_name=args.proposal_model_name,
            outcome_trace_target_modes=args.outcome_trace_target_modes,
            proposal_grpo_reward_modes=args.proposal_grpo_reward_modes,
            proposal_grpo_zero_variance_modes=args.proposal_grpo_zero_variance_modes,
            num_candidates_list=args.num_candidates_list,
            proposal_grpo_learning_rates=args.proposal_grpo_learning_rates,
            proposal_grpo_kl_coef=args.proposal_grpo_kl_coef,
            synthetic_proposal_sft_examples_list=args.synthetic_proposal_sft_examples_list,
            synthetic_proposal_sft_seed_mix=args.synthetic_proposal_sft_seed_mix,
            synthetic_proposal_sft_num_epochs=args.synthetic_proposal_sft_num_epochs,
            synthetic_proposal_sft_learning_rate=args.synthetic_proposal_sft_learning_rate,
            synthetic_proposal_sft_top_k=args.synthetic_proposal_sft_top_k,
            synthetic_proposal_sft_temperature=args.synthetic_proposal_sft_temperature,
            adaptive_config_files=args.adaptive_config_files,
            job_fields=args.job_fields,
        )
    else:
        raise AssertionError(f"Unhandled manifest_type={args.manifest_type!r}.")
    write_manifest(args.manifest, payload)


if __name__ == "__main__":
    main()

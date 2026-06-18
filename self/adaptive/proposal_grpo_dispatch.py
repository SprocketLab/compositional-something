"""Proposal-GRPO update dispatch helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.adaptive.phases import PHASE_PROPOSAL_GRPO
from self.core.models import CandidateMetrics
from self.adaptive.proposal_prompts import PromptBundle


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
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[CandidateMetrics],
    seed: int,
    deps: ProposalGrpoDispatchDeps,
) -> tuple[str, JsonDict]:
    if args.controller_execution_mode != "slurm":
        return deps.apply_proposal_grpo_update(
            args=args,
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            prompt=prompt,
            proposal_results=proposal_results,
            candidate_metrics=candidate_metrics,
            seed=seed,
        )
    deps.ensure_dir(output_dir)
    prompt_path = output_dir / "proposal_prompt.json"
    proposal_results_path = output_dir / "proposal_results.json"
    candidate_metrics_path = output_dir / "candidate_metrics.json"
    deps.write_json(prompt_path, {"system": prompt.system, "user": prompt.user})
    deps.write_json(proposal_results_path, proposal_results)
    deps.write_json(candidate_metrics_path, [metric.to_json_dict() for metric in candidate_metrics])
    worker_output = deps.run_controller_worker_slurm(
        args=args,
        worker_dir=output_dir / "controller_worker",
        phase=PHASE_PROPOSAL_GRPO,
        payload={
            "source_checkpoint": source_checkpoint,
            "proposal_grpo_dir": str(output_dir),
            "prompt_path": str(prompt_path),
            "proposal_results_path": str(proposal_results_path),
            "candidate_metrics_path": str(candidate_metrics_path),
            "seed": seed,
        },
    )
    return str(worker_output["next_checkpoint"]), dict(worker_output["proposal_grpo_metrics"])

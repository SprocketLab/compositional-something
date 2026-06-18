"""Driver-binding bridge for proposal-GRPO update dispatch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from self.core.models import CandidateMetrics
from self.core.proposal_grpo_dispatch import (
    ProposalGrpoDispatchDeps,
    apply_or_dispatch_proposal_grpo_update as _apply_or_dispatch_proposal_grpo_update_impl,
)
from self.core.proposals import PromptBundle


JsonDict = Dict[str, Any]


def proposal_grpo_dispatch_deps(bindings: Any) -> ProposalGrpoDispatchDeps:
    return ProposalGrpoDispatchDeps(
        apply_proposal_grpo_update=bindings.apply_proposal_grpo_update,
        run_controller_worker_slurm=bindings._run_controller_worker_slurm,
        ensure_dir=bindings.ensure_dir,
        write_json=bindings.write_json,
    )


def apply_or_dispatch_proposal_grpo_update(
    bindings: Any,
    *,
    args: argparse.Namespace,
    source_checkpoint: str,
    output_dir: Path,
    prompt: PromptBundle,
    proposal_results: Sequence[Mapping[str, Any]],
    candidate_metrics: Sequence[CandidateMetrics],
    seed: int,
) -> tuple[str, JsonDict]:
    return _apply_or_dispatch_proposal_grpo_update_impl(
        args=args,
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        prompt=prompt,
        proposal_results=proposal_results,
        candidate_metrics=candidate_metrics,
        seed=seed,
        deps=proposal_grpo_dispatch_deps(bindings),
    )

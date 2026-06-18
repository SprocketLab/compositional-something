"""Checkpoint retention and cleanup helpers for adaptive training."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from self.core.models import CandidateMetrics


@dataclass(frozen=True)
class CheckpointManager:
    output_dir: Path
    keep_candidate_models: bool = False
    keep_proposal_grpo_checkpoints: bool = False

    def cleanup_unselected_candidates(
        self,
        *,
        metrics: Sequence[CandidateMetrics],
        selected: Optional[CandidateMetrics],
    ) -> None:
        if self.keep_candidate_models:
            return
        selected_dir = selected.model_dir if selected is not None else None
        for metric in metrics:
            model_dir = metric.model_dir
            if model_dir is None or model_dir == selected_dir:
                continue
            parent = model_dir.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)

    def cleanup_replaced_checkpoint(
        self,
        *,
        old_checkpoint: str,
        new_checkpoint: str,
    ) -> List[str]:
        if old_checkpoint == new_checkpoint:
            return []
        old_model_dir = Path(old_checkpoint)
        new_model_dir = Path(new_checkpoint)
        if old_model_dir.name != "model":
            return []
        if not old_model_dir.exists() or not new_model_dir.exists():
            return []
        try:
            old_model_dir.resolve().relative_to(self.output_dir.resolve())
        except ValueError:
            return []
        except OSError:
            return []
        if old_model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in old_model_dir.parts and self.keep_candidate_models:
            return []
        shutil.rmtree(old_model_dir, ignore_errors=True)
        return [str(old_model_dir)]


def cleanup_unselected_models(
    *,
    metrics: Sequence[CandidateMetrics],
    selected: Optional[CandidateMetrics],
    keep_all: bool,
) -> None:
    CheckpointManager(output_dir=Path("."), keep_candidate_models=keep_all).cleanup_unselected_candidates(
        metrics=metrics,
        selected=selected,
    )


def cleanup_replaced_model_checkpoint(
    *,
    old_checkpoint: str,
    new_checkpoint: str,
    output_dir: Path,
    keep_candidate_models: bool,
    keep_proposal_grpo_checkpoints: bool,
) -> List[str]:
    return CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=keep_candidate_models,
        keep_proposal_grpo_checkpoints=keep_proposal_grpo_checkpoints,
    ).cleanup_replaced_checkpoint(
        old_checkpoint=old_checkpoint,
        new_checkpoint=new_checkpoint,
    )

#!/usr/bin/env python3
"""Controller phase contracts for adaptive candidate training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


PHASE_SEED = "seed"
PHASE_ROUND_MODEL = "round_model"
PHASE_PROPOSAL_GRPO = "proposal_grpo"


@dataclass(frozen=True)
class SeedPhaseResult:
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Dict[int, float]
    init_final_accuracy: float
    model_dir: Optional[Path]


@dataclass(frozen=True)
class RoundModelPhaseResult:
    current_final_accuracy: float
    current_per_size_accuracy: Dict[int, float]
    prompt: Any
    proposal_results: List[dict[str, Any]]
    work_items: List[Any]

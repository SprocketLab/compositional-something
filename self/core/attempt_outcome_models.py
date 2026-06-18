"""Shared attempt-outcome dependency and result containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AttemptOutcomeDeps:
    build_round_outcome_trace_examples: Callable[..., list[Any]]
    build_selected_proposal_trace_example: Callable[..., Any]
    apply_or_dispatch_proposal_grpo_update: Callable[..., tuple[str, JsonDict]]
    write_json: Callable[[Path, Any], None]
    write_trace_jsonl: Callable[[Path, Sequence[Mapping[str, Any]]], None]
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]


@dataclass(frozen=True)
class AttemptOutcomeResult:
    selected_rounds: int
    consecutive_no_selection: int
    current_checkpoint: str
    current_final_accuracy: float
    current_per_size_accuracy: Mapping[int, float]
    proposal_grpo_update_count: int
    should_break: bool = False


__all__ = [
    "AttemptOutcomeDeps",
    "AttemptOutcomeResult",
    "JsonDict",
]

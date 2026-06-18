"""Per-round setup helpers for non-adaptive self-improvement runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence


@dataclass(frozen=True)
class NonAdaptiveRoundPlan:
    round_idx: int
    max_size: int
    round_dir: Path
    save_model_this_round: bool
    should_skip_completed_round: bool


@dataclass(frozen=True)
class NonAdaptiveRoundTrainingData:
    train_examples: List[Any]
    pseudo_used_count: int


def prepare_nonadaptive_round_plan(
    *,
    base_output_dir: Path,
    round_idx: int,
    size_schedule: Any,
    save_model_policy: str,
    num_expand_rounds: int,
    resume_requested: bool,
    resume_round: int,
    ensure_dir_fn: Callable[[Path], Path | None],
) -> NonAdaptiveRoundPlan:
    """Resolve round-local paths, size metadata, save policy, and resume skip state."""
    max_size = size_schedule.round_max_size_for_index(round_idx)
    round_dir = base_output_dir / f"round_{round_idx:02d}"
    ensure_dir_fn(round_dir)
    save_model_this_round = save_model_policy == "all_rounds" or (
        save_model_policy == "final_only" and round_idx == num_expand_rounds
    )
    return NonAdaptiveRoundPlan(
        round_idx=round_idx,
        max_size=max_size,
        round_dir=round_dir,
        save_model_this_round=save_model_this_round,
        should_skip_completed_round=bool(resume_requested and round_idx < resume_round),
    )


def prepare_nonadaptive_round_training_data(
    *,
    round_dir: Path,
    base_train_examples: Sequence[Any],
    pseudo_examples: Sequence[Any],
    task: Any,
    save_examples_fn: Callable[[Path, Iterable[Any], Callable[[Any], Any]], None],
) -> NonAdaptiveRoundTrainingData:
    """Build and persist the training examples used by one non-adaptive round."""
    train_examples = list(base_train_examples)
    train_examples.extend(pseudo_examples)
    pseudo_used_count = len(pseudo_examples)

    save_examples_fn(round_dir / "train_examples.jsonl", train_examples, task.serialize_example)
    save_examples_fn(round_dir / "pseudo_examples_used.jsonl", pseudo_examples, task.serialize_example)

    return NonAdaptiveRoundTrainingData(
        train_examples=train_examples,
        pseudo_used_count=pseudo_used_count,
    )

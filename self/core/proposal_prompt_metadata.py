"""Task-specific proposal prompt metadata and default source-pair policy."""

from __future__ import annotations

import argparse
from typing import List

from self.core.program_sandbox_cases import (
    build_addition_program_cases,
    build_run_length_program_cases,
)
from self.core.program_sandbox_models import SandboxCase
from self.core.proposal_config_schema import ConfigProposal
from self.tasks.bit_common import normalize_bit_target_mode
from self.tasks.bit_parsing import RUN_LENGTH_TARGET_RUN_STATE


def target_format_for_task(task_name: str, args: argparse.Namespace) -> str:
    if task_name == "addition":
        return "a non-negative integer string formed only from component prediction strings"
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return "max_run|prefix_symbol|prefix_run|suffix_symbol|suffix_run"
        return "max_run|prefix_run|suffix_run"
    raise ValueError(f"Unsupported task={task_name!r}.")


def component_prediction_examples_for_task(task_name: str, args: argparse.Namespace) -> List[str]:
    if task_name == "addition":
        return ["46", "064", "1002"]
    if task_name == "run_length":
        if normalize_bit_target_mode(args) == RUN_LENGTH_TARGET_RUN_STATE:
            return ["3|0|2|1|1", "5|1|5|1|5"]
        return ["3|2|1", "5|5|5"]
    raise ValueError(f"Unsupported task={task_name!r}.")


def program_validation_cases(task_name: str, args: argparse.Namespace) -> List[SandboxCase]:
    if task_name == "addition":
        return build_addition_program_cases()
    if task_name == "run_length":
        if normalize_bit_target_mode(args) != RUN_LENGTH_TARGET_RUN_STATE:
            return []
        return build_run_length_program_cases(random_seed=args.seed, random_count=8)
    raise ValueError(f"Unsupported task={task_name!r}.")


def choose_default_program_pair(
    *,
    source_sizes: set[int],
    frontier_min: int,
    frontier_max: int,
    allow_repeat_targets: bool,
) -> ConfigProposal:
    for target in range(frontier_min, frontier_max + 1):
        if not allow_repeat_targets and target in source_sizes:
            continue
        pairs = [(left, target - left) for left in sorted(source_sizes) if target - left in source_sizes]
        if not pairs:
            continue
        left, right = sorted(pairs, key=lambda pair: (abs(pair[0] - pair[1]), pair[0]))[0]
        return ConfigProposal(left=left, right=right, guard="none", target=target)
    raise ValueError(
        "No driver-selected program pair is available from the current source pool "
        f"for frontier={frontier_min}..{frontier_max}."
    )


__all__ = [
    "choose_default_program_pair",
    "component_prediction_examples_for_task",
    "program_validation_cases",
    "target_format_for_task",
]

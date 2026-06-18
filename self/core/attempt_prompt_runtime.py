"""Attempt-level proposal prompt construction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from self.core.proposal_config_schema import DEFAULT_CONFIG_SEARCH_SPACES, ConfigProposal
from self.core.proposal_prompts import (
    choose_default_program_pair,
    PromptBundle,
    render_config_prompt,
    render_program_candidate_prompt,
)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AttemptPromptDeps:
    choose_default_program_pair: Callable[..., ConfigProposal]
    render_config_prompt: Callable[..., PromptBundle]
    render_program_candidate_prompt: Callable[..., PromptBundle]


@dataclass(frozen=True)
class AttemptPromptResult:
    prompt: PromptBundle
    default_program_pair: Optional[ConfigProposal]
    aggregate_metrics: JsonDict
    current_source: JsonDict
    allowed_target_frontier: JsonDict


DEFAULT_ATTEMPT_PROMPT_DEPS = AttemptPromptDeps(
    choose_default_program_pair=choose_default_program_pair,
    render_config_prompt=render_config_prompt,
    render_program_candidate_prompt=render_program_candidate_prompt,
)


def build_attempt_prompt(
    *,
    args: argparse.Namespace,
    current_checkpoint: str,
    current_final_accuracy: float,
    init_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    source_sizes: set[int],
    selected_round_for_prompt: int,
    attempt_index: int,
    selected_rounds: int,
    consecutive_no_selection: int,
    extra_aggregate_metrics: Optional[Mapping[str, Any]] = None,
    deps: AttemptPromptDeps = DEFAULT_ATTEMPT_PROMPT_DEPS,
) -> AttemptPromptResult:
    frontier_min = args.frontier_min_size
    frontier_max = args.frontier_max_size
    aggregate_metrics: JsonDict = {
        "current_final_accuracy": current_final_accuracy,
        "init_final_accuracy": init_final_accuracy,
        "per_size_accuracy": {str(size): score for size, score in current_per_size_accuracy.items()},
        "source_sizes": sorted(source_sizes),
        "attempt_index": attempt_index,
        "selected_rounds_completed": selected_rounds,
        "target_selected_rounds": args.num_rounds,
        "consecutive_no_selection": consecutive_no_selection,
        "reward_formula": "frontier_delta + lambda_final * (candidate_final_accuracy - init_final_accuracy)",
    }
    if extra_aggregate_metrics:
        aggregate_metrics.update(dict(extra_aggregate_metrics))

    current_source_payload: JsonDict = {
        "sizes": sorted(source_sizes),
        "min": min(source_sizes),
        "max": max(source_sizes),
    }
    allowed_frontier_payload: JsonDict = {
        "min": frontier_min,
        "max": frontier_max,
        "already_in_source": sorted(size for size in source_sizes if size >= frontier_min),
    }
    default_program_pair: Optional[ConfigProposal] = None
    if args.condition == "program":
        default_program_pair = deps.choose_default_program_pair(
            source_sizes=source_sizes,
            frontier_min=frontier_min,
            frontier_max=frontier_max,
            allow_repeat_targets=args.allow_repeat_targets,
        )
        aggregate_metrics["driver_selected_program_pair"] = default_program_pair.to_json_dict()

    if args.condition == "config":
        prompt = deps.render_config_prompt(
            task_name=args.task,
            round_index=selected_round_for_prompt,
            current_source=current_source_payload,
            allowed_target_frontier=allowed_frontier_payload,
            aggregate_metrics=aggregate_metrics,
            guard_choices=DEFAULT_CONFIG_SEARCH_SPACES[args.task]["guards"],
            model_name=current_checkpoint,
            proposal_output_schema=args.proposal_output_schema,
        )
    else:
        prompt = deps.render_program_candidate_prompt(
            args=args,
            round_index=selected_round_for_prompt,
            current_checkpoint=current_checkpoint,
            current_source=current_source_payload,
            allowed_target_frontier=allowed_frontier_payload,
            aggregate_metrics=aggregate_metrics,
            default_pair=default_program_pair,
        )

    return AttemptPromptResult(
        prompt=prompt,
        default_program_pair=default_program_pair,
        aggregate_metrics=aggregate_metrics,
        current_source=current_source_payload,
        allowed_target_frontier=allowed_frontier_payload,
    )

"""Round-iteration helper for the non-adaptive self-improvement loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from self.core.nonadaptive_round_models import (
    NonAdaptiveRoundRuntimeContext,
    NonAdaptiveRoundRuntimeState,
)


@dataclass(frozen=True)
class NonAdaptiveRoundLoopResult:
    round_dirs: list[Path]
    completed_rounds: int
    stopped_early: bool


def run_nonadaptive_round_loop(
    *,
    context: NonAdaptiveRoundRuntimeContext,
    state: NonAdaptiveRoundRuntimeState,
    num_rounds: int,
    run_round_fn: Callable[..., Any],
    round_runtime_kwargs: Mapping[str, Any],
) -> NonAdaptiveRoundLoopResult:
    """Run non-adaptive rounds until completion or a round asks to stop."""

    round_dirs: list[Path] = []
    stopped_early = False
    for round_idx in range(num_rounds):
        round_result = run_round_fn(
            context=context,
            state=state,
            round_idx=round_idx,
            **round_runtime_kwargs,
        )
        round_dirs.append(round_result.round_dir)
        if round_result.should_break:
            stopped_early = True
            break

    return NonAdaptiveRoundLoopResult(
        round_dirs=round_dirs,
        completed_rounds=len(round_dirs),
        stopped_early=stopped_early,
    )

"""Run-finalization helpers for non-adaptive self-improvement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True)
class NonAdaptiveFinalizationResult:
    checkpoints_cleaned: bool


def finalize_nonadaptive_run(
    *,
    keep_checkpoints: bool,
    save_model_policy: str,
    round_dirs: Sequence[Path],
    results_path: Path,
    cleanup_round_checkpoints_fn: Callable[[Sequence[Path]], None],
    print_fn: Callable[..., None] = print,
) -> NonAdaptiveFinalizationResult:
    """Clean transient checkpoints when requested and report the result path."""
    checkpoints_cleaned = False
    if not keep_checkpoints and save_model_policy != "none":
        cleanup_round_checkpoints_fn(round_dirs)
        checkpoints_cleaned = True

    print_fn(f"[INFO] Saved round summaries to {results_path}", flush=True)
    return NonAdaptiveFinalizationResult(checkpoints_cleaned=checkpoints_cleaned)

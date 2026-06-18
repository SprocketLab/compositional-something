"""Dry-run attempt handling for adaptive candidate training."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from self.core.proposals import ConfigProposal


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class DryRunAttemptDeps:
    load_fixture_proposals: Callable[[Path], list[Mapping[str, Any]]]
    rows_for_round: Callable[..., list[Mapping[str, Any]]]
    validate_proposal_rows: Callable[..., list[JsonDict]]
    build_candidate_work_items: Callable[..., list[Any]]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class DryRunAttemptResult:
    selected_rounds: int
    consecutive_no_selection: int
    should_stop: bool


def run_dry_attempt(
    *,
    args: argparse.Namespace,
    task: Any,
    round_dir: Path,
    output_dir: Path,
    selected_round_for_prompt: int,
    attempt_index: int,
    source_sizes: set[int],
    default_program_pair: Optional[ConfigProposal],
    source_examples: Sequence[Any],
    exclude_keys: set[Any],
    rng: random.Random,
    summary_records: list[JsonDict],
    selected_rounds: int,
    consecutive_no_selection: int,
    deps: DryRunAttemptDeps,
) -> DryRunAttemptResult:
    rows = (
        deps.rows_for_round(
            deps.load_fixture_proposals(args.proposal_fixture_jsonl),
            selected_round_for_prompt,
            attempt_index=attempt_index,
        )
        if args.proposal_fixture_jsonl
        else []
    )
    proposal_results = deps.validate_proposal_rows(
        rows=rows[: args.num_candidates],
        args=args,
        source_sizes=source_sizes,
        frontier_min=args.frontier_min_size,
        frontier_max=args.frontier_max_size,
        default_pair=default_program_pair,
    )
    deps.write_json(round_dir / "proposal_results.json", proposal_results)
    work_items = deps.build_candidate_work_items(
        args=args,
        task=task,
        round_dir=round_dir,
        proposal_results=proposal_results,
        source_examples=source_examples,
        exclude_keys=exclude_keys,
        rng=rng,
    )
    deps.write_json(
        round_dir / "dry_run_summary.json",
        {
            "work_items": len(work_items),
            "attempt_index": attempt_index,
            "selected_round": selected_round_for_prompt,
        },
    )
    summary_records.append(
        {
            "attempt": attempt_index,
            "selected_round": selected_round_for_prompt if work_items else None,
            "selected": None,
            "dry_run_data_only": True,
            "work_items": len(work_items),
            "source_sizes": sorted(source_sizes),
        }
    )
    deps.write_json(output_dir / "adaptive_candidate_training_results.json", summary_records)
    if work_items:
        return DryRunAttemptResult(
            selected_rounds=selected_rounds + 1,
            consecutive_no_selection=0,
            should_stop=False,
        )

    next_consecutive_no_selection = consecutive_no_selection + 1
    should_stop = next_consecutive_no_selection >= args.no_selection_patience
    if should_stop:
        print(
            f"[WARN] Reached no_selection_patience={args.no_selection_patience}; stopping dry run.",
            flush=True,
        )
    return DryRunAttemptResult(
        selected_rounds=selected_rounds,
        consecutive_no_selection=next_consecutive_no_selection,
        should_stop=should_stop,
    )

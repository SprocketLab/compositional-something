#!/usr/bin/env python3
"""Experience trace builders and replay samplers for adaptive self-improvement."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

from self.core.experience_outcome_traces import build_outcome_trace_example, build_round_outcome_trace_examples
from self.core.experience_trace_models import (
    OutcomeTraceExample,
    ProposalTraceExample,
    build_post_task_proposal_rehearsal_examples,
    outcome_trace_from_json,
    proposal_trace_from_json,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.core.proposals import PromptBundle, build_trace_row, write_trace_jsonl


JsonDict = Dict[str, Any]


def write_round_trace(
    *,
    args: argparse.Namespace,
    task_name: str,
    round_index: int,
    prompt: PromptBundle,
    work_items: Sequence[Any],
    metrics: Sequence[Any],
    path: Path,
) -> List[JsonDict]:
    by_index = {item.index: item for item in work_items}
    positive = [metric for metric in metrics if metric.valid and metric.reward > 0.0]
    positive.sort(key=lambda metric: metric.reward, reverse=True)
    rows: List[JsonDict] = []
    for metric in positive[: max(0, args.max_traces_per_round)]:
        item = by_index.get(metric.index)
        if item is None:
            continue
        rows.append(
            build_trace_row(
                round_index=round_index,
                task=task_name,
                condition=args.condition,
                reward=metric.reward,
                frontier_delta=metric.frontier_delta,
                final_accuracy=metric.final_accuracy,
                prompt=prompt.text(),
                completion=item.completion,
                extra_metadata=proposal_trace_metadata(metric),
            )
        )
    write_trace_jsonl(path, rows)
    return rows


def proposal_trace_metadata(metric: Any) -> JsonDict:
    return {
        "proposal_index": metric.index,
        "proposal_id": metric.row_id,
        "target": metric.proposal.target,
        "left": metric.proposal.left,
        "right": metric.proposal.right,
        "guard": metric.proposal.guard,
        "proposal_prediction": metric.proposal_prediction,
        "target_accuracy": metric.target_accuracy,
        "current_target_accuracy": metric.current_target_accuracy,
        "target_delta": metric.target_delta,
        "frontier_accuracy": metric.frontier_accuracy,
        "current_frontier_accuracy": metric.current_frontier_accuracy,
        "frontier_delta": metric.frontier_delta,
        "init_final_accuracy": metric.init_final_accuracy,
        "final_accuracy_delta": metric.final_accuracy_delta,
        "final_accuracy_delta_from_current": metric.final_accuracy_delta_from_current,
        "pseudo_count": metric.pseudo_count,
    }


def build_selected_proposal_trace_example(
    *,
    task_name: str,
    condition: str,
    round_index: int,
    prompt: PromptBundle,
    selected_item: Any,
    selected: Any,
) -> ProposalTraceExample:
    return ProposalTraceExample(
        prompt_text=prompt.text(),
        completion=selected_item.completion,
        task=task_name,
        condition=condition,
        round_index=round_index,
        reward=selected.reward,
        metadata=proposal_trace_metadata(selected),
    )


def build_candidate_proposal_trace_example(
    *,
    task_name: str,
    condition: str,
    round_index: int,
    prompt: PromptBundle,
    item: Any,
) -> ProposalTraceExample:
    proposal = item.proposal
    return ProposalTraceExample(
        prompt_text=prompt.text(),
        completion=item.completion,
        task=task_name,
        condition=condition,
        round_index=round_index,
        reward=math.nan,
        metadata={
            "proposal_index": item.index,
            "proposal_id": item.row_id,
            "target": proposal.target,
            "left": proposal.left,
            "right": proposal.right,
            "guard": proposal.guard,
            "proposal_prediction": item.proposal_prediction,
            "candidate_local_trace": True,
        },
    )

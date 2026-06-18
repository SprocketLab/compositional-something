"""Candidate reward and metric construction helpers."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

from self.core.models import CandidateMetrics, CandidateWorkItem


def static_frontier_sizes(args: argparse.Namespace) -> list[int]:
    return list(range(int(args.frontier_min_size), int(args.frontier_max_size) + 1))


def mean_accuracy_for_sizes(per_size_accuracy: Mapping[int, float], sizes: Sequence[int]) -> float:
    if not sizes:
        return math.nan
    total = 0.0
    for size in sizes:
        value = per_size_accuracy.get(int(size), 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        total += numeric
    return total / len(sizes)


def build_no_pseudo_candidate_metrics(
    *,
    args: argparse.Namespace,
    item: CandidateWorkItem,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    current_target_accuracy = float(current_per_size_accuracy.get(item.proposal.target, 0.0))
    current_frontier_accuracy = mean_accuracy_for_sizes(current_per_size_accuracy, static_frontier_sizes(args))
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=False,
        reward=float("-inf"),
        frontier_delta=float("-inf"),
        target_accuracy=math.nan,
        current_target_accuracy=current_target_accuracy,
        final_accuracy=math.nan,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=math.nan,
        per_size_accuracy={},
        pseudo_count=0,
        model_dir=None,
        failure_reason="no pseudo labels retained",
        proposal_trace_replay_count=0,
        candidate_proposal_trace_count=0,
        outcome_trace_replay_count=0,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=math.nan,
        target_delta=math.nan,
        frontier_accuracy=math.nan,
        current_frontier_accuracy=current_frontier_accuracy,
        proposal_prediction=dict(item.proposal_prediction),
    )


def build_trained_candidate_metrics(
    *,
    args: argparse.Namespace,
    item: CandidateWorkItem,
    final_accuracy: float,
    per_size_accuracy: Mapping[int, float],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    model_dir: Path,
    proposal_trace_replay_count: int,
    candidate_proposal_trace_count: int,
    post_task_proposal_rehearsal_count: int,
    outcome_trace_replay_count: int,
) -> CandidateMetrics:
    target_accuracy = float(per_size_accuracy.get(item.proposal.target, 0.0))
    current_target_accuracy = float(current_per_size_accuracy.get(item.proposal.target, 0.0))
    target_delta = target_accuracy - current_target_accuracy
    frontier_sizes = static_frontier_sizes(args)
    frontier_accuracy = mean_accuracy_for_sizes(per_size_accuracy, frontier_sizes)
    current_frontier_accuracy = mean_accuracy_for_sizes(current_per_size_accuracy, frontier_sizes)
    frontier_delta = frontier_accuracy - current_frontier_accuracy
    final_accuracy_delta = final_accuracy - init_final_accuracy
    final_accuracy_delta_from_current = final_accuracy - current_final_accuracy
    reward = frontier_delta + args.lambda_final * final_accuracy_delta
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=True,
        reward=reward,
        frontier_delta=frontier_delta,
        frontier_accuracy=frontier_accuracy,
        current_frontier_accuracy=current_frontier_accuracy,
        target_accuracy=target_accuracy,
        current_target_accuracy=current_target_accuracy,
        target_delta=target_delta,
        final_accuracy=final_accuracy,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=final_accuracy_delta,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=final_accuracy_delta_from_current,
        per_size_accuracy={int(size): float(value) for size, value in per_size_accuracy.items()},
        pseudo_count=len(item.pseudo_examples),
        model_dir=model_dir,
        proposal_trace_replay_count=proposal_trace_replay_count,
        candidate_proposal_trace_count=candidate_proposal_trace_count,
        post_task_proposal_rehearsal_count=post_task_proposal_rehearsal_count,
        outcome_trace_replay_count=outcome_trace_replay_count,
        proposal_prediction=dict(item.proposal_prediction),
    )

"""Candidate selection policy for adaptive self-improvement."""

from __future__ import annotations

from typing import Optional, Sequence

from self.core.models import CandidateMetrics


def select_candidate(metrics: Sequence[CandidateMetrics], min_reward: float) -> Optional[CandidateMetrics]:
    eligible = [metric for metric in metrics if metric.valid and metric.reward >= min_reward]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda metric: (
            metric.reward,
            metric.frontier_delta,
            metric.target_delta,
            metric.final_accuracy_delta_from_current,
        ),
        reverse=True,
    )[0]

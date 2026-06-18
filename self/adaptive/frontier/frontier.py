#!/usr/bin/env python3
"""Adaptive frontier selection public API and proposal-quality metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value
from self.adaptive.frontier.frontier_candidates import build_frontier_candidates
from self.adaptive.frontier.frontier_models import FrontierCandidate, FrontierSelection


JsonDict = Dict[str, Any]

__all__ = [
    "FrontierCandidate",
    "FrontierSelection",
    "build_frontier_candidates",
    "load_diagnostics_payload",
    "proposal_quality_metrics",
    "select_frontier",
]


def load_diagnostics_payload(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if not payload:
            return {}
        last = payload[-1]
        if not isinstance(last, dict):
            raise ValueError(f"Last diagnostics entry in {path} is not an object.")
        return dict(last)
    if not isinstance(payload, dict):
        raise ValueError(f"Diagnostics payload at {path} must be a JSON object or non-empty list of objects.")
    return dict(payload)


def select_frontier(
    diagnostics: Mapping[str, Any],
    *,
    task: str,
    allowed_min: int,
    allowed_max: int,
    policy: str = "weak_regime",
    min_count: int = 1,
    max_accuracy: float = 0.85,
    max_width: int = 1,
    prefer_larger_weight: float = 0.01,
) -> FrontierSelection:
    if policy not in {"fixed", "weak_regime"}:
        raise ValueError(f"Unsupported frontier policy {policy!r}.")
    if allowed_max < allowed_min:
        raise ValueError("allowed_max must be >= allowed_min.")
    if policy == "fixed":
        return FrontierSelection(
            policy=policy,
            selected=None,
            candidates=[],
            fallback_frontier_min=allowed_min,
            fallback_frontier_max=allowed_max,
        )
    candidates = build_frontier_candidates(
        diagnostics,
        task=task,
        allowed_min=allowed_min,
        allowed_max=allowed_max,
        min_count=min_count,
        max_accuracy=max_accuracy,
        max_width=max_width,
        prefer_larger_weight=prefer_larger_weight,
    )
    return FrontierSelection(
        policy=policy,
        selected=candidates[0] if candidates else None,
        candidates=candidates,
        fallback_frontier_min=allowed_min,
        fallback_frontier_max=allowed_max,
    )


def proposal_quality_metrics(
    results: Sequence[Mapping[str, Any]],
    *,
    selected_id: Optional[str] = None,
) -> JsonDict:
    total = len(results)
    valid = [row for row in results if bool(row.get("valid"))]
    invalid = [row for row in results if not bool(row.get("valid"))]
    duplicates = [row for row in results if bool(row.get("duplicate"))]
    positives = [row for row in results if bool(row.get("trace_include"))]
    eligible = [row for row in results if bool(row.get("selection_eligible"))]
    repairs_attempted = [row for row in results if bool(row.get("repair_attempted"))]
    repaired = [row for row in results if bool(row.get("repaired"))]
    rewards = [float(row.get("reward", 0.0)) for row in valid]
    eligible_rewards = [float(row.get("reward", 0.0)) for row in eligible]
    by_category: Dict[str, int] = {}
    for row in invalid:
        category = str(row.get("validation_category") or "invalid")
        by_category[category] = by_category.get(category, 0) + 1

    def rate(count: int) -> Optional[float]:
        return count / total if total else None

    def mean(values: Sequence[float]) -> Optional[float]:
        return sum(values) / len(values) if values else None

    payload: JsonDict = {
        "proposal_count": total,
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "valid_rate": rate(len(valid)),
        "duplicate_count": len(duplicates),
        "duplicate_rate": rate(len(duplicates)),
        "positive_count": len(positives),
        "positive_rate": rate(len(positives)),
        "selection_eligible_count": len(eligible),
        "selection_eligible_rate": rate(len(eligible)),
        "repair_attempted_count": len(repairs_attempted),
        "repair_attempted_rate": rate(len(repairs_attempted)),
        "repaired_count": len(repaired),
        "repair_success_rate": (len(repaired) / len(repairs_attempted) if repairs_attempted else None),
        "best_reward": max(rewards) if rewards else None,
        "mean_valid_reward": mean(rewards),
        "best_eligible_reward": max(eligible_rewards) if eligible_rewards else None,
        "mean_eligible_reward": mean(eligible_rewards),
        "selected_id": selected_id,
        "invalid_by_category": by_category,
    }
    return sanitize_json_value(payload)

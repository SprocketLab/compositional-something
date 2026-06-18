#!/usr/bin/env python3
"""Adaptive frontier selection and proposal-quality metrics."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class FrontierCandidate:
    task: str
    size_min: int
    size_max: int
    slice_name: str
    accuracy: float
    count: int
    score: float
    source: str
    reason: str

    def to_json_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class FrontierSelection:
    policy: str
    selected: Optional[FrontierCandidate]
    candidates: List[FrontierCandidate]
    fallback_frontier_min: int
    fallback_frontier_max: int

    def frontier_min(self) -> int:
        if self.selected is None:
            return self.fallback_frontier_min
        return self.selected.size_min

    def frontier_max(self) -> int:
        if self.selected is None:
            return self.fallback_frontier_max
        return self.selected.size_max

    def to_json_dict(self) -> JsonDict:
        return {
            "policy": self.policy,
            "selected": self.selected.to_json_dict() if self.selected else None,
            "candidates": [candidate.to_json_dict() for candidate in self.candidates],
            "fallback_frontier_min": self.fallback_frontier_min,
            "fallback_frontier_max": self.fallback_frontier_max,
            "frontier_min": self.frontier_min(),
            "frontier_max": self.frontier_max(),
        }


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


def _coerce_accuracy(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return max(0.0, min(1.0, result))


def _coerce_count(value: Any, default: int = 1) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, result)


def _size_accuracy_items(payload: Mapping[str, Any]) -> Iterable[tuple[int, float]]:
    for key in ("per_size_accuracy", "per_digit_accuracy", "per_bit_accuracy"):
        raw = payload.get(key)
        if not isinstance(raw, Mapping):
            continue
        for size_text, accuracy_raw in raw.items():
            try:
                size_value = int(size_text)
            except (TypeError, ValueError):
                continue
            accuracy = _coerce_accuracy(accuracy_raw)
            if accuracy is not None:
                yield size_value, accuracy


def _candidate_score(
    *,
    accuracy: float,
    count: int,
    size_min: int,
    allowed_min: int,
    prefer_larger_weight: float,
) -> float:
    error = 1.0 - accuracy
    count_weight = math.log1p(max(1, count))
    size_weight = prefer_larger_weight * max(0, size_min - allowed_min)
    return error * count_weight + size_weight


def _add_candidate(
    candidates: List[FrontierCandidate],
    *,
    task: str,
    size_min: int,
    size_max: int,
    slice_name: str,
    accuracy: float,
    count: int,
    source: str,
    reason: str,
    allowed_min: int,
    allowed_max: int,
    min_count: int,
    max_accuracy: float,
    prefer_larger_weight: float,
) -> None:
    if size_max < allowed_min or size_min > allowed_max:
        return
    clipped_min = max(size_min, allowed_min)
    clipped_max = min(size_max, allowed_max)
    if clipped_max < clipped_min:
        return
    if count < min_count or accuracy > max_accuracy:
        return
    score = _candidate_score(
        accuracy=accuracy,
        count=count,
        size_min=clipped_min,
        allowed_min=allowed_min,
        prefer_larger_weight=prefer_larger_weight,
    )
    candidates.append(
        FrontierCandidate(
            task=task,
            size_min=clipped_min,
            size_max=clipped_max,
            slice_name=slice_name,
            accuracy=accuracy,
            count=count,
            score=score,
            source=source,
            reason=reason,
        )
    )


def build_frontier_candidates(
    diagnostics: Mapping[str, Any],
    *,
    task: str,
    allowed_min: int,
    allowed_max: int,
    min_count: int = 1,
    max_accuracy: float = 0.85,
    max_width: int = 1,
    prefer_larger_weight: float = 0.01,
) -> List[FrontierCandidate]:
    """Rank exact-size and slice-level weak regimes from aggregate diagnostics.

    The selector treats the frontier as "where the current model is weak and
    still inside the experiment's safe search envelope", not automatically as
    the next larger size.
    """
    candidates: List[FrontierCandidate] = []
    bounded_width = max(1, int(max_width))

    for size_value, accuracy in _size_accuracy_items(diagnostics):
        _add_candidate(
            candidates,
            task=task,
            size_min=size_value,
            size_max=min(allowed_max, size_value + bounded_width - 1),
            slice_name="all",
            accuracy=accuracy,
            count=_coerce_count(diagnostics.get("eval_count"), default=1),
            source="per_size_accuracy",
            reason=f"size {size_value} accuracy {accuracy:.4f} is below threshold",
            allowed_min=allowed_min,
            allowed_max=allowed_max,
            min_count=min_count,
            max_accuracy=max_accuracy,
            prefer_larger_weight=prefer_larger_weight,
        )

    composed_slices = diagnostics.get("composed_eval_slices")
    if isinstance(composed_slices, Mapping):
        for slice_name, metric_raw in composed_slices.items():
            if not isinstance(metric_raw, Mapping):
                continue
            slice_accuracy = _coerce_accuracy(metric_raw.get("accuracy"))
            slice_count = _coerce_count(metric_raw.get("count"), default=1)
            per_size_items = list(_size_accuracy_items(metric_raw))
            if per_size_items:
                for size_value, accuracy in per_size_items:
                    _add_candidate(
                        candidates,
                        task=task,
                        size_min=size_value,
                        size_max=min(allowed_max, size_value + bounded_width - 1),
                        slice_name=str(slice_name),
                        accuracy=accuracy,
                        count=slice_count,
                        source=f"composed_eval_slices.{slice_name}.per_size_accuracy",
                        reason=(
                            f"slice {slice_name!s} at size {size_value} accuracy "
                            f"{accuracy:.4f} is below threshold"
                        ),
                        allowed_min=allowed_min,
                        allowed_max=allowed_max,
                        min_count=min_count,
                        max_accuracy=max_accuracy,
                        prefer_larger_weight=prefer_larger_weight,
                    )
            elif slice_accuracy is not None:
                _add_candidate(
                    candidates,
                    task=task,
                    size_min=allowed_min,
                    size_max=min(allowed_max, allowed_min + bounded_width - 1),
                    slice_name=str(slice_name),
                    accuracy=slice_accuracy,
                    count=slice_count,
                    source=f"composed_eval_slices.{slice_name}",
                    reason=f"slice {slice_name!s} accuracy {slice_accuracy:.4f} is below threshold",
                    allowed_min=allowed_min,
                    allowed_max=allowed_max,
                    min_count=min_count,
                    max_accuracy=max_accuracy,
                    prefer_larger_weight=prefer_larger_weight,
                )

    generic_regimes = diagnostics.get("regime_metrics")
    if isinstance(generic_regimes, Sequence) and not isinstance(generic_regimes, (str, bytes)):
        for index, metric_raw in enumerate(generic_regimes):
            if not isinstance(metric_raw, Mapping):
                continue
            accuracy = _coerce_accuracy(metric_raw.get("accuracy"))
            if accuracy is None:
                continue
            size_min = _coerce_count(metric_raw.get("size_min", metric_raw.get("size")), default=allowed_min)
            size_max = _coerce_count(metric_raw.get("size_max", size_min), default=size_min)
            _add_candidate(
                candidates,
                task=task,
                size_min=size_min,
                size_max=size_max,
                slice_name=str(metric_raw.get("slice", metric_raw.get("slice_name", "all"))),
                accuracy=accuracy,
                count=_coerce_count(metric_raw.get("count"), default=1),
                source=f"regime_metrics[{index}]",
                reason=str(metric_raw.get("reason", "generic regime accuracy is below threshold")),
                allowed_min=allowed_min,
                allowed_max=allowed_max,
                min_count=min_count,
                max_accuracy=max_accuracy,
                prefer_larger_weight=prefer_larger_weight,
            )

    candidates.sort(key=lambda item: (item.score, item.size_max, item.count), reverse=True)
    return candidates


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
        "repair_success_rate": (
            len(repaired) / len(repairs_attempted) if repairs_attempted else None
        ),
        "best_reward": max(rewards) if rewards else None,
        "mean_valid_reward": mean(rewards),
        "best_eligible_reward": max(eligible_rewards) if eligible_rewards else None,
        "mean_eligible_reward": mean(eligible_rewards),
        "selected_id": selected_id,
        "invalid_by_category": by_category,
    }
    return sanitize_json_value(payload)

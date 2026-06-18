#!/usr/bin/env python3
"""Value objects for adaptive frontier selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


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

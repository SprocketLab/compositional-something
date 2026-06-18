"""Size-schedule helpers for the non-adaptive self-improvement loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class NonAdaptiveSizeSchedule:
    initial_max_size: int
    expand_num_size: int
    num_expand_rounds: int
    frontier_min_size: Optional[int]

    @property
    def final_max_size(self) -> int:
        if self.frontier_min_size is None:
            return self.initial_max_size + self.expand_num_size * self.num_expand_rounds
        if self.num_expand_rounds <= 0:
            return self.initial_max_size
        return self.frontier_min_size + self.expand_num_size * self.num_expand_rounds - 1

    @property
    def composed_min_size(self) -> int:
        if self.frontier_min_size is None:
            return self.initial_max_size + 1
        return self.frontier_min_size

    def round_max_size_for_index(self, round_idx: int) -> int:
        if self.frontier_min_size is None or round_idx <= 0:
            return self.initial_max_size + round_idx * self.expand_num_size
        return self.frontier_min_size + round_idx * self.expand_num_size - 1

    def target_max_size_for_round(self, round_idx: int) -> int:
        if self.frontier_min_size is None:
            return self.initial_max_size + (round_idx + 1) * self.expand_num_size
        return self.frontier_min_size + (round_idx + 1) * self.expand_num_size - 1


def normalize_frontier_min_size(args: Any) -> Optional[int]:
    frontier_min_size = getattr(args, "frontier_min_size", None)
    if frontier_min_size is None:
        return None
    frontier_min_size = int(frontier_min_size)
    if frontier_min_size <= args.initial_max_size:
        raise ValueError("frontier_min_size must be greater than initial_max_size.")
    return frontier_min_size


def build_nonadaptive_size_schedule(args: Any, frontier_min_size: Optional[int]) -> NonAdaptiveSizeSchedule:
    return NonAdaptiveSizeSchedule(
        initial_max_size=int(args.initial_max_size),
        expand_num_size=int(args.expand_num_size),
        num_expand_rounds=int(args.num_expand_rounds),
        frontier_min_size=frontier_min_size,
    )

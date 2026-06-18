"""Composition-size helpers shared by bit-string tasks."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence


BIT_COMPOSITION_PATH_RANDOM = "random"
BIT_COMPOSITION_PATH_FIXED_BINARY = "fixed_binary"
BIT_COMPOSITION_PATH_MODES = {BIT_COMPOSITION_PATH_RANDOM, BIT_COMPOSITION_PATH_FIXED_BINARY}


def choose_component_sizes(
    target_size: int,
    sizes: Sequence[int],
    rng: random.Random,
    *,
    min_parts: int = 2,
    compose_arity: str = "at_least2",
    bit_composition_path_mode: str = BIT_COMPOSITION_PATH_RANDOM,
) -> Optional[List[int]]:
    unique_sizes = sorted({size for size in sizes if size > 0 and size <= target_size})
    if not unique_sizes:
        return None

    if bit_composition_path_mode == BIT_COMPOSITION_PATH_FIXED_BINARY:
        left = target_size // 2
        right = target_size - left
        if left not in unique_sizes or right not in unique_sizes:
            raise ValueError(
                f"Unable to fixed-binary compose size {target_size}: requires component sizes "
                f"{left}+{right}, available={unique_sizes}."
            )
        return [left, right]

    if compose_arity == "exact2":
        pairs = [
            [left, right]
            for left in unique_sizes
            for right in unique_sizes
            if left + right == target_size
        ]
        if not pairs:
            return None
        return rng.choice(pairs)

    memo: Dict[tuple[int, int], Optional[List[int]]] = {}

    def helper(remaining: int, parts_needed: int) -> Optional[List[int]]:
        key = (remaining, parts_needed)
        if key in memo:
            return memo[key]
        candidates = list(unique_sizes)
        rng.shuffle(candidates)
        for size in candidates:
            if size > remaining:
                continue
            next_remaining = remaining - size
            next_parts_needed = max(0, parts_needed - 1)
            if next_remaining == 0:
                if next_parts_needed == 0:
                    memo[key] = [size]
                    return memo[key]
                continue
            tail = helper(next_remaining, next_parts_needed)
            if tail is not None:
                memo[key] = [size, *tail]
                return memo[key]
        memo[key] = None
        return None

    return helper(target_size, min_parts)


def exact2_reachable_sizes_from_examples(
    examples: Sequence[Any],
    *,
    size_getter: Callable[[Any], int],
    min_size: int,
    max_size: int,
) -> List[int]:
    available_sizes = sorted({size_getter(example) for example in examples})
    if not available_sizes:
        return []
    reachable = {
        left + right
        for left in available_sizes
        for right in available_sizes
        if min_size <= left + right <= max_size
    }
    return sorted(reachable)


def fixed_binary_reachable_sizes_from_examples(
    examples: Sequence[Any],
    *,
    size_getter: Callable[[Any], int],
    min_size: int,
    max_size: int,
) -> List[int]:
    available_sizes = {size_getter(example) for example in examples}
    if not available_sizes:
        return []
    reachable = []
    for target_size in range(min_size, max_size + 1):
        left = target_size // 2
        right = target_size - left
        if left in available_sizes and right in available_sizes:
            reachable.append(target_size)
    return reachable


def bit_composed_target_sizes_from_examples(
    examples: Sequence[Any],
    *,
    size_getter: Callable[[Any], int],
    min_size: int,
    max_size: int,
    compose_arity: str,
    bit_composition_path_mode: str,
) -> Optional[List[int]]:
    if bit_composition_path_mode == BIT_COMPOSITION_PATH_FIXED_BINARY:
        return fixed_binary_reachable_sizes_from_examples(
            examples,
            size_getter=size_getter,
            min_size=min_size,
            max_size=max_size,
        )
    if compose_arity == "exact2":
        return exact2_reachable_sizes_from_examples(
            examples,
            size_getter=size_getter,
            min_size=min_size,
            max_size=max_size,
        )
    return None

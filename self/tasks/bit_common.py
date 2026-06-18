#!/usr/bin/env python3
"""Shared helpers for bit-string self-improvement tasks."""

from __future__ import annotations

import itertools
import random
from typing import Any, List, Tuple

from self.tasks.bit_composition import (
    BIT_COMPOSITION_PATH_FIXED_BINARY,
    BIT_COMPOSITION_PATH_MODES,
    BIT_COMPOSITION_PATH_RANDOM,
    bit_composed_target_sizes_from_examples,
    choose_component_sizes,
    exact2_reachable_sizes_from_examples,
    fixed_binary_reachable_sizes_from_examples,
)
from self.tasks.bit_parsing import (
    INTEGER_PATTERN,
    MULTIPLICATION_FORMATS,
    RUN_LENGTH_ALPHABET_SYMBOLS,
    RUN_LENGTH_FORMATS,
    RUN_LENGTH_RUN_STATE_PATTERN,
    RUN_LENGTH_TARGET_RUN_STATE,
    SYMBOL_RUN_PAIR_PATTERN,
    format_multiplication_target,
    parse_multiplication_prediction,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
    parse_run_length_symbol_pair_prediction,
)
from self.tasks.bit_pseudolabels import (
    build_direct_pseudo_examples,
    build_guarded_bit_pseudo_examples,
    count_examples_by_size,
    format_size_count_map,
    guard_slice_partition,
    run_length_guard_accepts_true_components,
)


BIT_TARGET_MODES = {"default", "plain_output", "symbol_run_pair", RUN_LENGTH_TARGET_RUN_STATE}
BIT_COMPOSE_ARITIES = {"at_least2", "exact2"}
BIT_GUARDED_COMPOSE_RULES = {
    "none",
    "run_length_no_boundary_continue",
    "run_length_unfiltered_pair",
}


def normalize_task_format_version(args: Any, default: str = "legacy") -> str:
    return str(getattr(args, "format_version", default))


def normalize_bit_target_mode(args: Any, default: str = "default") -> str:
    return str(getattr(args, "target_mode", default))


def normalize_compose_arity(args: Any, default: str = "at_least2") -> str:
    return str(getattr(args, "compose_arity", default))


def normalize_bit_composition_path_mode(args: Any, default: str = BIT_COMPOSITION_PATH_RANDOM) -> str:
    return str(getattr(args, "bit_composition_path_mode", default))


def normalize_guarded_compose_rule(args: Any, default: str = "none") -> str:
    return str(getattr(args, "guarded_compose_rule", default))


def normalize_symbol_alphabet_size(args: Any, default: int = 2) -> int:
    return int(getattr(args, "symbol_alphabet_size", default))


def sample_unique_bitstrings(
    bits: int,
    count: int,
    rng: random.Random,
    occupied: set[Tuple[int, str]],
    *,
    alphabet: str = "01",
    max_attempts: int = 10_000,
) -> List[str]:
    if count <= 0:
        return []
    available = (len(alphabet) ** bits) - sum(1 for key in occupied if key[0] == bits)
    if count > available:
        raise ValueError(f"Requested {count} unique bitstrings for bits={bits}, but only {available} remain.")

    if available <= 131_072 and count * 4 >= available:
        pool = [
            "".join(symbols)
            for symbols in itertools.product(alphabet, repeat=bits)
            if (bits, "".join(symbols)) not in occupied
        ]
        rng.shuffle(pool)
        return pool[:count]

    sampled: List[str] = []
    attempts = 0
    while len(sampled) < count:
        attempts += 1
        bitstring = "".join(rng.choice(alphabet) for _ in range(bits))
        key = (bits, bitstring)
        if key in occupied or bitstring in sampled:
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Unable to sample {count} unique bitstrings for bits={bits} after {max_attempts} attempts."
                )
            continue
        sampled.append(bitstring)
        attempts = 0
    return sampled

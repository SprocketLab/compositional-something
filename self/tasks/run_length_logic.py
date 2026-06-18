#!/usr/bin/env python3
"""Pure run-length target/state helpers."""

from __future__ import annotations

from typing import Optional, Tuple

from self.tasks.bit_common import RUN_LENGTH_TARGET_RUN_STATE


def compute_run_stats(bitstring: str) -> Tuple[int, int, int]:
    if not bitstring:
        return 0, 0, 0
    max_run = 1
    current = 1
    for previous, current_symbol in zip(bitstring, bitstring[1:]):
        if current_symbol == previous:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    prefix_symbol = bitstring[0]
    prefix = 0
    for ch in bitstring:
        if ch == prefix_symbol:
            prefix += 1
        else:
            break
    suffix_symbol = bitstring[-1]
    suffix = 0
    for ch in reversed(bitstring):
        if ch == suffix_symbol:
            suffix += 1
        else:
            break
    return max_run, prefix, suffix


def compute_run_state(bitstring: str) -> Tuple[int, str, int, str, int]:
    max_run, prefix, suffix = compute_run_stats(bitstring)
    if not bitstring:
        return max_run, "", prefix, "", suffix
    return max_run, bitstring[0], prefix, bitstring[-1], suffix


def format_run_length_run_state(state: Tuple[int, str, int, str, int]) -> str:
    max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = state
    return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"


def merge_run_state(
    left: Tuple[int, int, str, int, str, int],
    right: Tuple[int, int, str, int, str, int],
) -> Tuple[int, int, str, int, str, int]:
    left_bits, left_max, left_prefix_symbol, left_prefix_run, left_suffix_symbol, left_suffix_run = left
    right_bits, right_max, right_prefix_symbol, right_prefix_run, right_suffix_symbol, right_suffix_run = right
    bits = left_bits + right_bits
    boundary = left_suffix_run + right_prefix_run if left_suffix_symbol == right_prefix_symbol else 0
    max_run = max(left_max, right_max, boundary)
    prefix_symbol = left_prefix_symbol
    prefix_run = left_prefix_run
    if left_prefix_run == left_bits and left_prefix_symbol == right_prefix_symbol:
        prefix_run = left_bits + right_prefix_run
    suffix_symbol = right_suffix_symbol
    suffix_run = right_suffix_run
    if right_suffix_run == right_bits and left_suffix_symbol == right_suffix_symbol:
        suffix_run = right_bits + left_suffix_run
    return bits, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run


def leftmost_max_run_pair(bitstring: str) -> Tuple[str, int]:
    if not bitstring:
        return "", 0
    best_symbol = bitstring[0]
    best_length = 1
    current_symbol = bitstring[0]
    current_length = 1
    for ch in bitstring[1:]:
        if ch == current_symbol:
            current_length += 1
        else:
            current_symbol = ch
            current_length = 1
        if current_length > best_length:
            best_symbol = current_symbol
            best_length = current_length
    return best_symbol, best_length


def format_run_length_target(
    max_run: int,
    prefix: int,
    suffix: int,
    format_version: str,
    target_mode: str = "default",
    *,
    bitstring: Optional[str] = None,
) -> str:
    del format_version
    if target_mode == "plain_output":
        return str(max_run)
    if target_mode == "symbol_run_pair":
        if bitstring is None:
            raise ValueError("symbol_run_pair run-length targets require bitstring context.")
        symbol, run_length = leftmost_max_run_pair(bitstring)
        return f"{symbol}|{run_length}"
    if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
        if bitstring is None:
            raise ValueError("run_state run-length targets require bitstring context.")
        state = compute_run_state(bitstring)
        return format_run_length_run_state(state)
    return f"{max_run}|{prefix}|{suffix}"

#!/usr/bin/env python3
"""Bit-string parsing, composition, and pseudolabel helpers."""

from __future__ import annotations

# --- from bit.py ---
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


# --- from bit.py ---
import re
from typing import Any, Optional

from self.core.evaluation import extract_numeric_answer


INTEGER_PATTERN = re.compile(r"[-+]?\d+")
RUN_LENGTH_FORMATS = {"legacy", "symbolic_v1"}
MULTIPLICATION_FORMATS = {"legacy", "symbolic_v1"}
RUN_LENGTH_TARGET_RUN_STATE = "run_state"
RUN_LENGTH_ALPHABET_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYMBOL_RUN_PAIR_PATTERN = re.compile(
    rf"([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])\s*(?:\||,|:|\s+)\s*([-+]?\d+)"
)

RUN_LENGTH_RUN_STATE_PATTERN = re.compile(
    rf"([-+]?\d+)\s*(?:\||,|:|\s+)\s*([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])"
    rf"\s*(?:\||,|:|\s+)\s*([-+]?\d+)\s*(?:\||,|:|\s+)\s*"
    rf"([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])\s*(?:\||,|:|\s+)\s*([-+]?\d+)"
)


def format_multiplication_target(value: int, digits: int, format_version: str) -> str:
    if format_version == "symbolic_v1":
        return f"{value:0{digits * 2}d}"
    return str(value)


def parse_multiplication_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    value = extract_numeric_answer(text)
    if value is None:
        return None
    if example is None or getattr(example, "format_version", "legacy") != "symbolic_v1":
        return value
    return format_multiplication_target(int(value), int(example.digits), str(example.format_version))


def parse_run_length_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    matches = INTEGER_PATTERN.findall(text)
    target_mode = getattr(example, "target_mode", "default") if example is not None else "default"
    if target_mode == "plain_output":
        if not matches:
            return None
        value = int(matches[-1])
        if value < 0:
            return None
        return str(value)
    if target_mode == "symbol_run_pair":
        return parse_run_length_symbol_pair_prediction(text, example)
    if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
        return parse_run_length_run_state_prediction(text, example)
    if len(matches) < 3:
        return None
    max_run = int(matches[0])
    prefix = int(matches[1])
    suffix = int(matches[2])
    return f"{max_run}|{prefix}|{suffix}"


def parse_run_length_symbol_pair_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    allowed_symbols = set(RUN_LENGTH_ALPHABET_SYMBOLS)
    if example is not None and getattr(example, "bitstring", None):
        allowed_symbols = set(str(example.bitstring))
    for match in SYMBOL_RUN_PAIR_PATTERN.finditer(text):
        symbol = match.group(1)
        if symbol not in allowed_symbols:
            continue
        value = int(match.group(2))
        if value < 0:
            continue
        return f"{symbol}|{value}"
    return None


def parse_run_length_run_state_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    allowed_symbols = set(RUN_LENGTH_ALPHABET_SYMBOLS)
    bits = None
    if example is not None:
        if getattr(example, "bitstring", None):
            allowed_symbols = set(str(example.bitstring))
        bits = int(getattr(example, "bits", 0) or 0)
    for match in RUN_LENGTH_RUN_STATE_PATTERN.finditer(text):
        max_run = int(match.group(1))
        prefix_symbol = match.group(2)
        prefix_run = int(match.group(3))
        suffix_symbol = match.group(4)
        suffix_run = int(match.group(5))
        if prefix_symbol not in allowed_symbols or suffix_symbol not in allowed_symbols:
            continue
        if max_run < 0 or prefix_run < 0 or suffix_run < 0:
            continue
        if bits and (max_run > bits or prefix_run > bits or suffix_run > bits):
            continue
        return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"
    return None


# --- from bit.py ---
import math
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map
from self.core.task_protocols import JsonDict

_DEFAULT_GENERATE_PREDICTION_MAP = generate_prediction_map


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.tasks")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def _generate_prediction_map(**kwargs: Any) -> Any:
    return _compat_symbol("generate_prediction_map", _DEFAULT_GENERATE_PREDICTION_MAP)(**kwargs)


def build_direct_pseudo_examples(
    candidate_examples: Sequence[Any],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    decode_max_new_tokens: int,
    key_getter: Callable[[Any], Any],
    prediction_parser: Callable[[str], Optional[str]],
    clone_builder: Callable[[Any, Optional[str]], Any],
    mode: str,
) -> Tuple[List[Any], int, JsonDict]:
    prediction_map = _generate_prediction_map(
        model=model,
        tokenizer=tokenizer,
        examples=candidate_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=key_getter,
        prediction_parser=prediction_parser,
    )
    pseudo_examples: List[Any] = []
    missing_total = 0
    for example in candidate_examples:
        override = prediction_map.get(key_getter(example))
        if override is None:
            missing_total += 1
            continue
        pseudo_examples.append(clone_builder(example, override))
    diagnostics: JsonDict = {
        "mode": mode,
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
    }
    return pseudo_examples, missing_total, diagnostics


def count_examples_by_size(examples: Sequence[Any], size_getter: Callable[[Any], int]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for example in examples:
        counts[size_getter(example)] += 1
    return dict(counts)


def format_size_count_map(values: Dict[int, int]) -> str:
    return ", ".join(f"{size}:{count}" for size, count in sorted(values.items()))


def run_length_guard_accepts_true_components(
    component_keys: Sequence[Tuple[int, str]],
) -> Optional[bool]:
    if len(component_keys) != 2:
        return None
    left_bitstring = component_keys[0][1]
    right_bitstring = component_keys[1][1]
    if not left_bitstring or not right_bitstring:
        return None
    return left_bitstring[-1] != right_bitstring[0]


def guard_slice_partition(
    examples: Sequence[Any],
    component_map: Dict[Any, List[Any]],
    *,
    key_getter: Callable[[Any], Any],
    guard_fn: Callable[[Sequence[Any]], Optional[bool]],
) -> Dict[str, List[Any]]:
    accepted: List[Any] = []
    rejected: List[Any] = []
    for example in examples:
        component_keys = component_map.get(key_getter(example), [])
        status = guard_fn(component_keys)
        if status is True:
            accepted.append(example)
        elif status is False:
            rejected.append(example)
    return {
        "accepted_by_guard": accepted,
        "rejected_by_guard": rejected,
        "all": list(examples),
    }


def build_guarded_bit_pseudo_examples(
    candidate_examples: Sequence[Any],
    initial_component_map: Dict[Any, List[Any]],
    *,
    target_max_size: int,
    requested_per_size: int,
    size_getter: Callable[[Any], int],
    key_getter: Callable[[Any], Any],
    clone_builder: Callable[[Any, Optional[str]], Any],
    evaluate_candidate: Callable[[Any, Optional[Sequence[Any]]], Tuple[str, Optional[str]]],
    refill_builder: Callable[[int, int, set[Any]], Tuple[List[Any], Dict[Any, List[Any]]]],
    mode: str,
    max_refill_rounds: int = 32,
) -> Tuple[List[Any], int, JsonDict]:
    active_candidates = [example for example in candidate_examples if size_getter(example) <= target_max_size]
    target_sizes = sorted({size_getter(example) for example in active_candidates})
    requested_counts = {size: requested_per_size for size in target_sizes}
    requested_total = sum(requested_counts.values())

    candidate_total_by_size: Dict[int, int] = defaultdict(int)
    retained_total_by_size: Dict[int, int] = defaultdict(int)
    missing_total_by_size: Dict[int, int] = defaultdict(int)
    rejected_total_by_size: Dict[int, int] = defaultdict(int)
    pseudo_examples: List[Any] = []
    missing_total = 0
    rejected_total = 0
    occupied_keys = {key_getter(example) for example in active_candidates}

    def process_batch(
        examples: Sequence[Any],
        component_map: Dict[Any, List[Any]],
    ) -> None:
        nonlocal missing_total, rejected_total
        for example in examples:
            key = key_getter(example)
            size = size_getter(example)
            candidate_total_by_size[size] += 1
            status, override = evaluate_candidate(example, component_map.get(key))
            if status == "accepted" and override is not None:
                pseudo_examples.append(clone_builder(example, override))
                retained_total_by_size[size] += 1
            elif status == "missing":
                missing_total += 1
                missing_total_by_size[size] += 1
            else:
                rejected_total += 1
                rejected_total_by_size[size] += 1

    process_batch(active_candidates, initial_component_map)

    refill_rounds = 0
    while True:
        deficits = {
            size: max(0, requested_counts[size] - retained_total_by_size.get(size, 0))
            for size in requested_counts
        }
        deficits = {size: count for size, count in deficits.items() if count > 0}
        if not deficits:
            break
        if refill_rounds >= max_refill_rounds:
            raise RuntimeError(
                "Unable to retain the requested guarded pseudo examples after refill attempts. "
                f"Missing per-size counts: {format_size_count_map(deficits)}"
            )
        refill_rounds += 1
        progress_made = False
        for size, need in sorted(deficits.items()):
            refill_examples, refill_component_map = refill_builder(size, need, occupied_keys)
            if not refill_examples:
                continue
            progress_made = True
            occupied_keys.update(key_getter(example) for example in refill_examples)
            process_batch(refill_examples, refill_component_map)
        if not progress_made:
            raise RuntimeError(
                "Unable to retain the requested guarded pseudo examples after refill attempts. "
                f"Missing per-size counts: {format_size_count_map(deficits)}"
            )

    diagnostics: JsonDict = {
        "mode": mode,
        "target_max_bits": int(target_max_size),
        "requested_per_size": requested_per_size,
        "requested_total": requested_total,
        "candidate_total": sum(candidate_total_by_size.values()),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "rejected_total": rejected_total,
        "retained_fraction": len(pseudo_examples) / sum(candidate_total_by_size.values())
        if candidate_total_by_size
        else math.nan,
        "per_size_candidate_total": dict(sorted(candidate_total_by_size.items())),
        "per_size_retained_total": dict(sorted(retained_total_by_size.items())),
        "per_size_missing_total": dict(sorted(missing_total_by_size.items())),
        "per_size_rejected_total": dict(sorted(rejected_total_by_size.items())),
        "refill_rounds": refill_rounds,
    }
    return pseudo_examples, missing_total, diagnostics


# --- from bit.py ---
import itertools
import random
from typing import Any, List, Tuple



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

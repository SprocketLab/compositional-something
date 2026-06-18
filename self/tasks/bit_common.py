#!/usr/bin/env python3
"""Shared helpers for bit-string self-improvement tasks."""

from __future__ import annotations

import math
import itertools
import random
import sys
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map
from self.core.task_protocols import JsonDict
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


BIT_TARGET_MODES = {"default", "plain_output", "symbol_run_pair", RUN_LENGTH_TARGET_RUN_STATE}
BIT_COMPOSE_ARITIES = {"at_least2", "exact2"}
BIT_GUARDED_COMPOSE_RULES = {
    "none",
    "run_length_no_boundary_continue",
    "run_length_unfiltered_pair",
}

_DEFAULT_GENERATE_PREDICTION_MAP = generate_prediction_map


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.self_improvement_tasks")
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

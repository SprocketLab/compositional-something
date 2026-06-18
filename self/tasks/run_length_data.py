#!/usr/bin/env python3
"""Run-length example containers and dataset construction helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from self.tasks.bit_common import BIT_COMPOSITION_PATH_RANDOM, choose_component_sizes, sample_unique_bitstrings
from self.tasks.run_length_logic import compute_run_stats, format_run_length_target


@dataclass(frozen=True)
class RunLengthExample:
    bitstring: str
    bits: int
    max_run: int
    prefix_run: int
    suffix_run: int
    format_version: str = "legacy"
    target_mode: str = "default"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"runlen({self.bitstring})="
        return f"Q: runlen({self.bitstring}) = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_run_length_target(
            self.max_run,
            self.prefix_run,
            self.suffix_run,
            self.format_version,
            self.target_mode,
            bitstring=self.bitstring,
        )

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def run_length_key(example: RunLengthExample) -> Tuple[int, str]:
    return example.bits, example.bitstring


def encode_run_length_key(key: Tuple[int, str]) -> str:
    return f"{key[0]}|{key[1]}"


def decode_run_length_key(value: str) -> Tuple[int, str]:
    bits, bitstring = value.split("|", 1)
    return int(bits), bitstring


def generate_run_length_example(
    num_bits: int,
    rng: random.Random,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
) -> RunLengthExample:
    bitstring = "".join(rng.choice(alphabet) for _ in range(num_bits))
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=num_bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=format_version,
        target_mode=target_mode,
    )


def merge_run_length(left: RunLengthExample, right: RunLengthExample) -> RunLengthExample:
    bitstring = left.bitstring + right.bitstring
    bits = left.bits + right.bits
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=left.format_version,
        target_mode=left.target_mode,
    )


def compose_run_length_examples(*examples: RunLengthExample) -> RunLengthExample:
    if len(examples) < 2:
        raise ValueError("Need at least two run-length examples to compose a longer instance.")
    merged = examples[0]
    for nxt in examples[1:]:
        merged = merge_run_length(merged, nxt)
    return merged


def compose_run_length_to_length(
    buckets: Dict[int, List[RunLengthExample]],
    target_bits: int,
    rng: random.Random,
    *,
    compose_arity: str = "at_least2",
    bit_composition_path_mode: str = BIT_COMPOSITION_PATH_RANDOM,
) -> Tuple[RunLengthExample, List[RunLengthExample]]:
    sizes = list(buckets.keys())
    chosen_sizes = choose_component_sizes(
        target_bits,
        sizes,
        rng,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    if not chosen_sizes:
        raise ValueError(
            f"Unable to compose a run-length example of {target_bits} bits from base bucket sizes {sorted(sizes)}."
        )
    chosen = [rng.choice(buckets[size]) for size in chosen_sizes]
    return compose_run_length_examples(*chosen), chosen


def build_run_length_length_bucket_dataset(
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
    split_order: Sequence[str] = ("train", "validation", "test"),
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for bits in range(min_bits, max_bits + 1):
        per_bit_per_split = {split: per_bit_counts.get(split, 0) for split in split_order}
        total_requested = sum(per_bit_per_split.values())
        if total_requested == 0:
            continue
        available_unique = max(0, (len(alphabet) ** bits) - used_counts.get(bits, 0))
        if available_unique < total_requested:
            print(
                f"[WARN] Requested {total_requested} examples for bits={bits} exceeds available unique strings ({available_unique}); capping counts.",
                flush=True,
            )
            remaining = available_unique
            for split in split_order:
                requested = per_bit_per_split[split]
                if requested > remaining:
                    per_bit_per_split[split] = remaining
                    remaining = 0
                else:
                    remaining -= requested
            total_requested = sum(per_bit_per_split.values())
            if total_requested == 0:
                continue

        generated: List[Tuple[RunLengthExample, Tuple[int, str], bool]] = []
        for bitstring in sample_unique_bitstrings(
            bits,
            total_requested,
            rng,
            occupied,
            alphabet=alphabet,
            max_attempts=max_attempts,
        ):
            max_run, prefix, suffix = compute_run_stats(bitstring)
            example = RunLengthExample(
                bitstring=bitstring,
                bits=bits,
                max_run=max_run,
                prefix_run=prefix,
                suffix_run=suffix,
                format_version=format_version,
                target_mode=target_mode,
            )
            key = run_length_key(example)
            occupied.add(key)
            used_counts[bits] += 1
            generated.append((example, key, False))

        index = 0
        for split in split_order:
            count = per_bit_per_split.get(split, 0)
            if count <= 0:
                continue
            chunk = generated[index : index + count]
            index += count
            splits[split].extend(example for example, _, _ in chunk)
            if record_keys and split in record_keys:
                for _, key, is_duplicate in chunk:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def bucket_run_length_by_bits(examples: Sequence[RunLengthExample]) -> Dict[int, List[RunLengthExample]]:
    buckets: Dict[int, List[RunLengthExample]] = defaultdict(list)
    for example in examples:
        buckets[example.bits].append(example)
    return buckets


def build_run_length_composed_dataset(
    base_splits: Dict[str, List[RunLengthExample]],
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    record_components: Optional[Dict[str, Dict[Tuple[int, str], List[Tuple[int, str]]]]] = None,
    compose_arity: str = "at_least2",
    bit_composition_path_mode: str = BIT_COMPOSITION_PATH_RANDOM,
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for split in ("train", "validation", "test"):
        requested_per_bit = per_bit_counts.get(split, 0)
        if requested_per_bit <= 0:
            continue
        buckets = bucket_run_length_by_bits(base_splits.get(split, []))
        component_map = None
        if record_components is not None:
            component_map = record_components.setdefault(split, {})
        for bits in range(min_bits, max_bits + 1):
            available_unique = max(0, (2**bits) - used_counts.get(bits, 0))
            effective_target = min(requested_per_bit, available_unique)
            if effective_target < requested_per_bit:
                print(
                    f"[WARN] Requested {requested_per_bit} composed examples for bits={bits} split='{split}' exceeds available unique strings ({available_unique}); capping.",
                    flush=True,
                )
            if effective_target <= 0:
                continue
            generated: List[Tuple[RunLengthExample, Tuple[int, str], bool, List[Tuple[int, str]]]] = []
            attempts = 0
            duplicates_allowed = False
            while len(generated) < effective_target:
                attempts += 1
                example, components = compose_run_length_to_length(
                    buckets,
                    bits,
                    rng,
                    compose_arity=compose_arity,
                    bit_composition_path_mode=bit_composition_path_mode,
                )
                component_keys = [run_length_key(component) for component in components]
                key = run_length_key(example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique composed sampling for bits={bits} split='{split}' (progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        generated.append((example, key, True, component_keys))
                        attempts = 0
                    continue
                occupied.add(key)
                used_counts[bits] += 1
                generated.append((example, key, False, component_keys))
                attempts = 0
            splits[split].extend(example for example, _, _, _ in generated)
            if record_keys and split in record_keys:
                for _, key, is_duplicate, _ in generated:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if component_map is not None:
                for _, key, _, component_keys in generated:
                    component_map[key] = component_keys
            if progress_name:
                print(
                    f"[INFO] Generated {len(generated)}/{effective_target} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )
        rng.shuffle(splits[split])
    return splits


def clone_run_length_with_override(example: RunLengthExample, override: Optional[str]) -> RunLengthExample:
    if override is None:
        return example
    return RunLengthExample(
        bitstring=example.bitstring,
        bits=example.bits,
        max_run=example.max_run,
        prefix_run=example.prefix_run,
        suffix_run=example.suffix_run,
        format_version=example.format_version,
        target_mode=example.target_mode,
        target_override=override,
    )

#!/usr/bin/env python3
"""Multiplication sampling, blocked-component payloads, and slice helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from self.tasks.multiplication_data import MultiplicationExample, MultiplicationKey


def _example_cls() -> type["MultiplicationExample"]:
    from self.tasks.multiplication_data import MultiplicationExample

    return MultiplicationExample


def _multiplication_key(example: "MultiplicationExample") -> "MultiplicationKey":
    from self.tasks.multiplication_data import multiplication_key

    return multiplication_key(example)


def random_int_with_exact_digits(num_digits: int, rng: random.Random) -> int:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    if num_digits == 1:
        return rng.randint(0, 9)
    low = 10 ** (num_digits - 1)
    high = (10**num_digits) - 1
    return rng.randint(low, high)


def generate_multiplication_seed_example(
    block_size: int,
    rng: random.Random,
    format_version: str = "legacy",
) -> "MultiplicationExample":
    upper = (10**block_size) - 1
    a = rng.randint(0, upper)
    b = rng.randint(0, upper)
    return _example_cls()(
        a=a,
        b=b,
        digits=block_size,
        result=a * b,
        operand_width=block_size,
        format_version=format_version,
    )


def generate_long_multiplication_example(
    digits: int,
    rng: random.Random,
    format_version: str = "legacy",
) -> "MultiplicationExample":
    a = random_int_with_exact_digits(digits, rng)
    b = random_int_with_exact_digits(digits, rng)
    return _example_cls()(
        a=a,
        b=b,
        digits=digits,
        result=a * b,
        operand_width=digits,
        format_version=format_version,
    )


def iter_multiplication_sizes(min_digits: int, max_digits: int, block_size: int) -> List[int]:
    del block_size
    if max_digits < min_digits:
        return []
    return list(range(min_digits, max_digits + 1))


def split_value_into_blocks(value: int, total_digits: int, block_size: int) -> List[int]:
    text = f"{value:0{total_digits}d}"
    blocks: List[int] = []
    for end in range(len(text), 0, -block_size):
        start = max(0, end - block_size)
        blocks.append(int(text[start:end]))
    return blocks


def analyze_partial_products(partials: Sequence[Dict[str, int]]) -> Tuple[int, int]:
    digit_fan_in: Dict[int, int] = defaultdict(int)
    column_sums: Dict[int, int] = defaultdict(int)
    max_position = 0
    for partial in partials:
        product = partial["a"] * partial["b"]
        shift = partial["shift"]
        text = str(product)
        for offset, digit_char in enumerate(reversed(text)):
            position = shift + offset
            digit_fan_in[position] += 1
            column_sums[position] += int(digit_char)
            max_position = max(max_position, position)

    max_overlap = max(digit_fan_in.values()) if digit_fan_in else 0

    carry_count = 0
    carry = 0
    position = 0
    while position <= max_position or carry > 0:
        total = column_sums.get(position, 0) + carry
        if total >= 10:
            carry_count += 1
        carry = total // 10
        position += 1
    return max_overlap, carry_count


def build_multiplication_component_payload(
    example: "MultiplicationExample",
    block_size: int,
) -> Dict[str, Any]:
    a_blocks = split_value_into_blocks(example.a, example.digits, block_size)
    b_blocks = split_value_into_blocks(example.b, example.digits, block_size)
    partials: List[Dict[str, int]] = []
    for i, block_a in enumerate(a_blocks):
        for j, block_b in enumerate(b_blocks):
            partials.append(
                {
                    "a": block_a,
                    "b": block_b,
                    "shift": (i + j) * block_size,
                }
            )
    max_overlap, carry_count = analyze_partial_products(partials)
    return {
        "partials": partials,
        "max_overlap": int(max_overlap),
        "carry_count": int(carry_count),
        "block_size": int(block_size),
    }


def build_multiplication_seed_dataset(
    *,
    block_size: int,
    per_split_counts: Dict[str, int],
    rng: random.Random,
    exclude_keys: Optional[set["MultiplicationKey"]] = None,
    record_keys: Optional[Dict[str, set["MultiplicationKey"]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
) -> Dict[str, List["MultiplicationExample"]]:
    splits: Dict[str, List["MultiplicationExample"]] = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    requested_total = sum(per_split_counts.get(split, 0) for split in ("train", "validation", "test"))
    universe = 10 ** (2 * block_size)
    available_unique = max(0, universe - len(occupied))
    if requested_total > available_unique:
        print(
            f"[WARN] Requested {requested_total} multiplication seed examples exceeds available unique pairs ({available_unique}); capping.",
            flush=True,
        )
    remaining_total = available_unique
    adjusted_counts = {}
    for split in ("train", "validation", "test"):
        requested = per_split_counts.get(split, 0)
        adjusted = min(requested, remaining_total)
        adjusted_counts[split] = adjusted
        remaining_total -= adjusted

    generated: List[Tuple["MultiplicationExample", "MultiplicationKey", bool]] = []
    attempts = 0
    duplicates_allowed = False
    total_needed = sum(adjusted_counts.values())
    while len(generated) < total_needed:
        attempts += 1
        example = generate_multiplication_seed_example(block_size, rng, format_version=format_version)
        key = _multiplication_key(example)
        if key in occupied:
            if attempts >= max_attempts:
                if not duplicates_allowed:
                    print(
                        f"[WARN] Exhausted unique multiplication seed sampling (progress={progress_name}); allowing duplicates.",
                        flush=True,
                    )
                    duplicates_allowed = True
                generated.append((example, key, True))
                attempts = 0
            continue
        occupied.add(key)
        generated.append((example, key, False))
        attempts = 0

    index = 0
    for split in ("train", "validation", "test"):
        count = adjusted_counts[split]
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
                f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' digits={block_size}",
                flush=True,
            )
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def build_multiplication_long_dataset(
    *,
    min_digits: int,
    max_digits: int,
    per_digit_counts: Dict[str, int],
    rng: random.Random,
    block_size: int,
    exclude_keys: Optional[set["MultiplicationKey"]] = None,
    record_keys: Optional[Dict[str, set["MultiplicationKey"]]] = None,
    progress_name: Optional[str] = None,
    record_components: Optional[Dict[str, Dict["MultiplicationKey", Dict[str, Any]]]] = None,
    max_attempts: int = 50_000,
    format_version: str = "legacy",
) -> Dict[str, List["MultiplicationExample"]]:
    splits: Dict[str, List["MultiplicationExample"]] = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    sizes = iter_multiplication_sizes(min_digits, max_digits, block_size)
    per_split_counts = {key: int(per_digit_counts.get(key, 0)) for key in ("train", "validation", "test")}

    for digits in sizes:
        count_per_size = {split: per_split_counts.get(split, 0) for split in ("train", "validation", "test")}
        requested_total = sum(count_per_size.values())
        if requested_total <= 0:
            continue

        value_count = 10 if digits == 1 else 9 * (10 ** (digits - 1))
        total_unique = value_count * value_count
        already_used = sum(1 for key in occupied if key[0] == digits)
        available_unique = max(0, total_unique - already_used)
        if requested_total > available_unique:
            print(
                f"[WARN] Requested {requested_total} multiplication examples exceeds available unique pairs "
                f"({available_unique}) for digits={digits}; capping.",
                flush=True,
            )
        remaining_total = available_unique
        adjusted_counts: Dict[str, int] = {}
        for split in ("validation", "test", "train"):
            requested = count_per_size.get(split, 0)
            adjusted = min(requested, remaining_total)
            adjusted_counts[split] = adjusted
            remaining_total -= adjusted

        generated: List[Tuple["MultiplicationExample", "MultiplicationKey", Dict[str, Any], bool]] = []
        attempts = 0
        duplicates_allowed = False
        total_needed = sum(adjusted_counts.values())
        while len(generated) < total_needed:
            attempts += 1
            example = generate_long_multiplication_example(digits, rng, format_version=format_version)
            key = _multiplication_key(example)
            if key in occupied:
                if attempts >= max_attempts:
                    if not duplicates_allowed:
                        print(
                            f"[WARN] Exhausted unique multiplication sampling (digits={digits}); allowing duplicates.",
                            flush=True,
                        )
                        duplicates_allowed = True
                    generated.append((example, key, {}, True))
                    attempts = 0
                continue
            payload = build_multiplication_component_payload(example, block_size)
            occupied.add(key)
            generated.append((example, key, payload, False))
            attempts = 0

        index = 0
        for split in ("train", "validation", "test"):
            count = adjusted_counts.get(split, 0)
            if count <= 0:
                continue
            chunk = generated[index : index + count]
            index += count
            splits[split].extend(example for example, _, _, _ in chunk)
            if record_keys and split in record_keys:
                for _, key, _, is_duplicate in chunk:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if record_components is not None:
                component_map = record_components.setdefault(split, {})
                for _, key, payload, is_duplicate in chunk:
                    if is_duplicate:
                        continue
                    component_map[key] = payload
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' digits={digits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def get_multiplication_slice_name(payload: Dict[str, Any], block_size: int) -> str:
    max_overlap = int(payload.get("max_overlap", 0))
    carry_count = int(payload.get("carry_count", 0))
    overlap_tag = "low_overlap" if max_overlap <= 2 else "high_overlap"
    carry_tag = "low_carry" if carry_count <= block_size else "high_carry"
    return f"{overlap_tag}_{carry_tag}"

"""Example, parsing, and sampled-data helpers for rectangular multiplication."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from self.tasks.rectangular_digits import (
    format_cot_reverse_prompt,
    format_cot_reverse_target,
    normalize_cot_reverse_prediction_for_training,
    parse_cot_reverse_final_value,
)
from self.tasks.rectangular_partitions import PartitionKey, partition_label


RectangularMultiplicationKey = Tuple[int, int, int, int]
RECTANGULAR_MULTIPLICATION_FORMATS = {"legacy", "symbolic_v1", "cot_reverse_v1"}

INTEGER_PATTERN = re.compile(r"[-+]?\d+")
SampleIntWithExactDigits = Callable[..., int]


def extract_numeric_answer(text: str) -> Optional[str]:
    matches = INTEGER_PATTERN.findall(text)
    if not matches:
        return None
    best: Optional[str] = None
    best_len = -1
    for token in matches:
        candidate = token.strip()
        length = len(candidate.lstrip("+-"))
        if length > best_len or (length == best_len and candidate != best):
            best = candidate
            best_len = length
    return best


def values_for_digits(num_digits: int, *, include_zero_single_digit: bool = True) -> range:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    if num_digits == 1:
        return range(0 if include_zero_single_digit else 1, 10)
    start = 10 ** (num_digits - 1)
    stop = 10**num_digits
    return range(start, stop)


def sample_int_with_exact_digits(
    num_digits: int,
    rng: random.Random,
    *,
    include_zero_single_digit: bool = False,
) -> int:
    if num_digits <= 0:
        raise ValueError("num_digits must be positive.")
    if num_digits == 1:
        low = 0 if include_zero_single_digit else 1
        return rng.randint(low, 9)
    low = 10 ** (num_digits - 1)
    high = (10**num_digits) - 1
    return rng.randint(low, high)


@dataclass(frozen=True, slots=True)
class RectangularMultiplicationExample:
    a: int
    b: int
    a_digits: int
    b_digits: int
    format_version: str = "legacy"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"{self.a:0{self.a_digits}d}×{self.b:0{self.b_digits}d}="
        if self.format_version == "cot_reverse_v1":
            return format_cot_reverse_prompt(self.a, self.b)
        return f"Q: {self.a:0{self.a_digits}d} * {self.b:0{self.b_digits}d} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        value = self.a * self.b
        if self.format_version == "symbolic_v1":
            return f"{value:0{self.a_digits + self.b_digits}d}"
        if self.format_version == "cot_reverse_v1":
            return format_cot_reverse_target(self.a, self.b)
        return str(value)

    def target_prefix(self) -> str:
        return "" if self.format_version in {"symbolic_v1", "cot_reverse_v1"} else " "

    @property
    def total_digits(self) -> int:
        return self.a_digits + self.b_digits


def rectangular_multiplication_key(example: RectangularMultiplicationExample) -> RectangularMultiplicationKey:
    return example.a_digits, example.b_digits, example.a, example.b


def normalize_rectangular_prediction_for_training(
    text: str,
    example: RectangularMultiplicationExample,
) -> Optional[str]:
    if example.format_version == "cot_reverse_v1":
        return normalize_cot_reverse_prediction_for_training(text)
    value = parse_rectangular_multiplication_final_value(text, example)
    if value is None:
        return None
    if example.format_version == "symbolic_v1":
        return f"{value:0{example.total_digits}d}"
    return str(value)


def parse_rectangular_multiplication_final_value(
    text: str,
    example: RectangularMultiplicationExample,
) -> Optional[int]:
    if example.format_version == "cot_reverse_v1":
        return parse_cot_reverse_final_value(text, example.total_digits)
    value = extract_numeric_answer(text)
    if value is None:
        return None
    return int(value)


def prediction_matches_example(text: str, example: RectangularMultiplicationExample) -> bool:
    value = parse_rectangular_multiplication_final_value(text, example)
    return value == (example.a * example.b)


def build_sampled_rectangular_dataset(
    *,
    partitions: Sequence[PartitionKey],
    per_partition_counts: Dict[str, int],
    rng: random.Random,
    format_version: str,
    exclude_keys: Optional[set[RectangularMultiplicationKey]] = None,
    record_keys: Optional[Dict[str, set[RectangularMultiplicationKey]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 50_000,
    include_zero_single_digit: bool = False,
    sample_int_fn: Optional[SampleIntWithExactDigits] = None,
) -> Dict[str, List[RectangularMultiplicationExample]]:
    splits = {name: [] for name in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    sampler = sample_int_fn or sample_int_with_exact_digits

    for partition in partitions:
        a_digits, b_digits = partition
        adjusted_counts = {
            split: int(per_partition_counts.get(split, 0))
            for split in ("train", "validation", "test")
        }
        generated: List[
            Tuple[RectangularMultiplicationExample, RectangularMultiplicationKey, bool]
        ] = []
        total_needed = sum(adjusted_counts.values())
        attempts = 0
        duplicates_allowed = False

        while len(generated) < total_needed:
            attempts += 1
            example = RectangularMultiplicationExample(
                a=sampler(
                    a_digits,
                    rng,
                    include_zero_single_digit=include_zero_single_digit,
                ),
                b=sampler(
                    b_digits,
                    rng,
                    include_zero_single_digit=include_zero_single_digit,
                ),
                a_digits=a_digits,
                b_digits=b_digits,
                format_version=format_version,
            )
            key = rectangular_multiplication_key(example)
            if key in occupied:
                if duplicates_allowed:
                    generated.append((example, key, True))
                    attempts = 0
                    continue
                if attempts >= max_attempts:
                    print(
                        f"[WARN] Exhausted unique rectangular multiplication sampling for partition "
                        f"{partition_label(partition)}; allowing duplicates.",
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
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for "
                    f"split='{split}' partition={partition_label(partition)}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits

#!/usr/bin/env python3
"""
Weak-to-strong generalization experiment for digit-wise multiplication.

This script generates synthetic multiplication datasets, constructs compositional
examples, fine-tunes Qwen models under three training regimes, and
evaluates exact-match accuracy on test sets.

Variants:
1. Weak model (Qwen3-0.6B) trained on short multiplications (<=5 digits)
2. Strong_Full (Qwen3-1.7B) trained on full coverage up to max digits
3. Strong_W2S_GT (Qwen3-1.7B) trained on weak data + compositional non-carry data

Results are logged to TensorBoard and printed as a table.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore[override]
        """Minimal no-op writer when tensorboard is unavailable."""
        def __init__(self, *args, **kwargs) -> None:
            pass
        def add_scalar(self, *args, **kwargs) -> None:
            pass
        def close(self) -> None:
            pass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
set_seed,
)

from datetime import datetime

try:
    from peft import LoraConfig as PeftLoraConfig, get_peft_model
except ImportError:
    PeftLoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

NUMERIC_PATTERN = re.compile(r"[-+]?\d+")


from .addition_pipeline import (
    example_key,
    # encode_key,
    # decode_key,
    # save_pseudo_cache,
    # load_pseudo_cache,
    # extract_numeric_answer
)


def encode_key(key: Tuple[Tuple[int, int], int, int]) -> str:
    return f"{key[0][0]}|{key[0][1]}|{key[1]}|{key[2]}"


def decode_key(value: str) -> Tuple[Tuple[int, int], int, int]:
    parts = value.split("|")
    if len(parts) != 4:
        raise ValueError(f"Invalid key encoding: {value}")
    return (int(parts[0]), int(parts[1])), int(parts[2]), int(parts[3])


def save_prediction_map(
    cache_path: Path,
    predictions: Dict[Tuple[Tuple[int, int], int, int], str],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a prediction map (from `generate_prediction_map`) to JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "predictions": {encode_key(key): value for key, value in predictions.items()},
    }
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_prediction_map(cache_path: Path) -> Dict[Tuple[Tuple[int, int], int, int], str]:
    """Load a prediction map saved by `save_prediction_map`."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Prediction cache file not found: {cache_path}")
    with cache_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    predictions = payload.get("predictions", {})
    return {decode_key(key): value for key, value in predictions.items()}


@dataclass(frozen=True)
class MultiplicationExample:

    a: int
    b: int
    result: int
    digits: Tuple[int, int]
    target_override: Optional[str] = None

    def prompt(self) -> str:
        return f"Q: {self.a} * {self.b} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return str(self.result)

    def target_w_component_map(self, component_map: Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]]) -> str:
        """Chain-of-thought style target that shows the component decomposition before the final answer.

        Produces a geometric sum `c_0 * 10^(k-1) + c_1 * 10^(k-2) + ... + c_{k-1} = result`
        where components are ordered high-to-low. CoT is only emitted for
        E_{n,m} with n>=2 and m>=2; examples with a 1-digit side fall back to
        the plain label.
        """
        if self.target_override is not None:
            return self.target_override
        key = example_key(self)
        components = component_map.get(key)
        if not components:
            return str(self.result)
        if self.digits[0] < 2 or self.digits[1] < 2:
            return str(self.result)
        k = len(components)
        terms = [
            str(components[i][1] * components[i][2] * 10**(k - 1 - i))
            for i in range(k)
        ]
        return " + ".join(terms) + f" = {self.result}"

    def target_w_base_predictions(
        self,
        component_map: Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]],
        base_predictions: Dict[Tuple[Tuple[int, int], int, int], str],
    ) -> str:
        """CoT target whose intermediate steps use base-model predictions.

        Components are in high-to-low order; the chain is
        `pred_0 * 10^(k-1) + pred_1 * 10^(k-2) + ... + pred_{k-1} = final`.
        The final answer after '=' uses `target_override` when set (pseudo-label
        flow) and `self.result` otherwise (ground-truth flow). Falls back to
        `target_w_component_map` if component predictions are missing or
        non-integer, and returns a plain label when no components are
        registered. CoT is only emitted for E_{n,m} with n>=2 and m>=2;
        examples with a 1-digit side fall back to the plain label.
        """
        key = example_key(self)
        components = component_map.get(key)
        final = self.target_override if self.target_override is not None else str(self.result)
        if not components:
            return final
        if self.digits[0] < 2 or self.digits[1] < 2:
            return final
        try:
            preds = [int(base_predictions[comp]) for comp in components]
        except (KeyError, ValueError):
            return self.target_w_component_map(component_map)
        k = len(preds)
        terms = [str(preds[i] * 10**(k - 1 - i)) for i in range(k)]
        return " + ".join(terms) + f" = {final}"

    def formatted_a(self) -> str:
        return str(self.a)

    def formatted_b(self) -> str:
        return str(self.b)


def example_key(example: MultiplicationExample) -> Tuple[Tuple[int, int], int, int]:
    """Stable key for deduplication across splits."""
    return (example.digits, example.a, example.b)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from a model response.

    Handles both target formats produced by `target()` and `target_w_component_map()`:
    - plain: "{result}"
    - CoT:   "{comp1} + {comp2} = {result}"

    For CoT-style outputs, take the first number after the last '='. Otherwise,
    take the first number in the text.
    """
    tail = text.rsplit("=", 1)[1] if "=" in text else text
    match = NUMERIC_PATTERN.search(tail)
    return match.group(0).strip() if match else None


def generate_multiplication_pair(
    digits_a: int,
    digits_b: int,
    rng: Optional[random.Random] = None,
) -> MultiplicationExample:
    """Return a random multiplication example with requested digit lengths.

    For 1-digit operands the lower bound is 0 (operand range [0, 9]) so the
    component model trained on (n, 1) / (1, n) shapes sees `x * 0 = 0`. The
    schoolbook decomposition of any (n, m) with m >= 2 queries `a * 0` whenever
    b has a zero digit (~27-47% of random b's depending on m), and excluding 0
    from training puts those queries off-distribution. For multi-digit operands
    the lower bound stays at 10^(d-1) to preserve the digit count (no leading
    zeros).
    """
    if digits_a <= 0 or digits_b <= 0:
        raise ValueError("digits_a and digits_b must be positive")
    rng = rng or random.Random()
    low_a = 0 if digits_a == 1 else 10 ** (digits_a - 1)
    low_b = 0 if digits_b == 1 else 10 ** (digits_b - 1)
    max_a, max_b = 10**digits_a - 1, 10**digits_b - 1
    a = rng.randint(low_a, max_a)
    b = rng.randint(low_b, max_b)
    return MultiplicationExample(a=a, b=b, result=a * b, digits=(digits_a, digits_b))


def generate_composable_multiplication_pairs(
    target_digits_n: int,
    target_digits_m: int,
    rng: Optional[random.Random],
) -> Tuple[MultiplicationExample, List[MultiplicationExample]]:
    """Attempt to compose a multiplication example E_{n,m} with target_digits=(n,m) from two components E_{n-1,m} and E_{1,m}"""
    if target_digits_n <= 1 or target_digits_m <= 0:
        raise ValueError("target_digits_n must be > 1 and target_digits_m must be > 0 for composable examples")
    rng = rng or random.Random()
    low_m, high_m = 10**(target_digits_m - 1), 10**target_digits_m - 1
    low_n_minus_1, high_n_minus_1 = 10**(target_digits_n - 2), 10**(target_digits_n - 1) - 1
    low_1, high_1 = 1, 9
    m = rng.randint(low_m, high_m)
    n_minus_1 = rng.randint(low_n_minus_1, high_n_minus_1)
    one = rng.randint(low_1, high_1)
    example_n_minus_1_m = MultiplicationExample(a=n_minus_1, b=m, result=n_minus_1 * m, digits=(target_digits_n - 1, target_digits_m))
    example_1_m = MultiplicationExample(a=one, b=m, result=one  * m, digits=(1, target_digits_m))
    example_n_m = MultiplicationExample(a=example_n_minus_1_m.a * 10 + example_1_m.a, b=m, result=(example_n_minus_1_m.a * 10 + example_1_m.a) * m, digits=(target_digits_n, target_digits_m))
    return example_n_m, [example_n_minus_1_m, example_1_m]


def generate_composable_multiplication_pairs_b_axis(
    target_digits_n: int,
    target_digits_m: int,
    rng: Optional[random.Random] = None,
) -> Tuple[MultiplicationExample, List[MultiplicationExample]]:
    """Compose E_{n,m} from E_{n,m-1} and E_{n,1} by decomposing b.

    Mirrors `generate_composable_multiplication_pairs` for shapes with n=1 where
    a-axis decomposition is impossible. Returns [high_part, low_part] so the
    usual `{pred0 * 10} + {pred1}` CoT format applies unchanged.
    """
    if target_digits_n <= 0 or target_digits_m <= 1:
        raise ValueError("target_digits_n must be > 0 and target_digits_m must be > 1 for b-axis composition")
    rng = rng or random.Random()
    low_n, high_n = 10**(target_digits_n - 1), 10**target_digits_n - 1
    low_m_minus_1, high_m_minus_1 = 10**(target_digits_m - 2), 10**(target_digits_m - 1) - 1
    low_1, high_1 = 1, 9
    n_val = rng.randint(low_n, high_n)
    m_minus_1_val = rng.randint(low_m_minus_1, high_m_minus_1)
    one_val = rng.randint(low_1, high_1)
    high_part = MultiplicationExample(
        a=n_val, b=m_minus_1_val,
        result=n_val * m_minus_1_val,
        digits=(target_digits_n, target_digits_m - 1),
    )
    low_part = MultiplicationExample(
        a=n_val, b=one_val,
        result=n_val * one_val,
        digits=(target_digits_n, 1),
    )
    composed_b = m_minus_1_val * 10 + one_val
    composed = MultiplicationExample(
        a=n_val, b=composed_b,
        result=n_val * composed_b,
        digits=(target_digits_n, target_digits_m),
    )
    return composed, [high_part, low_part]


def generate_schoolbook_multiplication_pairs(
    target_digits_n: int,
    target_digits_m: int,
    rng: Optional[random.Random] = None,
) -> Tuple[MultiplicationExample, List[MultiplicationExample]]:
    """Compose E_{n,m} via schoolbook long multiplication with geometric shifts.

    Decomposes along the shorter dimension:
    - n >= m (and m >= 2): splits b into its m digits, emitting m components of
      shape (n, 1), so `a * b = sum_i (a * b_i) * 10^(m-1-i)`.
    - n < m (and n >= 2): splits a into its n digits, emitting n components of
      shape (1, m).
    - m == 1 with n >= 2: splits a into n digits of shape (1, 1) (the short
      side is already 1 digit, so we split the only decomposable side).
    - n == 1 with m >= 2: splits b into m digits of shape (1, 1).
    - (1, 1): raises.

    Components are ordered high-to-low; component i contributes
    `pred_i * 10^(k-1-i)` with k = len(components).
    """
    if target_digits_n < 1 or target_digits_m < 1:
        raise ValueError("target_digits_n and target_digits_m must be >= 1")
    if target_digits_n == 1 and target_digits_m == 1:
        raise ValueError("Cannot decompose E_{1,1}.")
    rng = rng or random.Random()
    # 1-digit operands include 0 so the (n, 1) / (1, m) training distribution
    # covers `x * 0 = 0`; multi-digit operands keep the leading-digit floor.
    low_a = 0 if target_digits_n == 1 else 10 ** (target_digits_n - 1)
    low_b = 0 if target_digits_m == 1 else 10 ** (target_digits_m - 1)
    high_a, high_b = 10 ** target_digits_n - 1, 10 ** target_digits_m - 1
    a = rng.randint(low_a, high_a)
    b = rng.randint(low_b, high_b)
    composed = MultiplicationExample(
        a=a, b=b, result=a * b,
        digits=(target_digits_n, target_digits_m),
    )
    if target_digits_m == 1:
        # b is already 1-digit; split a.
        decompose_b = False
    elif target_digits_n == 1:
        # a is already 1-digit; split b.
        decompose_b = True
    else:
        # Both >= 2. Split the shorter side (ties -> b).
        decompose_b = target_digits_n >= target_digits_m
    if decompose_b:
        component_digits = (target_digits_n, 1)
        digits_b = [(b // 10 ** i) % 10 for i in range(target_digits_m - 1, -1, -1)]
        components = [
            MultiplicationExample(a=a, b=bi, result=a * bi, digits=component_digits)
            for bi in digits_b
        ]
    else:
        component_digits = (1, target_digits_m)
        digits_a = [(a // 10 ** i) % 10 for i in range(target_digits_n - 1, -1, -1)]
        components = [
            MultiplicationExample(a=ai, b=b, result=ai * b, digits=component_digits)
            for ai in digits_a
        ]
    return composed, components


def generate_composable_multiplication_pairs_auto(
    target_digits_n: int,
    target_digits_m: int,
    rng: Optional[random.Random] = None,
) -> Tuple[MultiplicationExample, List[MultiplicationExample]]:
    """Compose E_{n,m} via schoolbook decomposition (kept for backward compat)."""
    return generate_schoolbook_multiplication_pairs(target_digits_n, target_digits_m, rng)


def calculate_max_pairs(digits_a: int, digits_b: int) -> int:
    """Calculate the total number of valid (a, b) pairs for given digit lengths.

    1-digit operands span [0, 9] (10 values); multi-digit operands span
    [10^(d-1), 10^d - 1] (9 * 10^(d-1) values, no leading zeros). Mirrors the
    sampling bounds used by `generate_multiplication_pair` and
    `generate_schoolbook_multiplication_pairs`.
    """
    if digits_a <= 0 or digits_b <= 0:
        return 0
    count_a = 10 if digits_a == 1 else 9 * (10 ** (digits_a - 1))
    count_b = 10 if digits_b == 1 else 9 * (10 ** (digits_b - 1))
    return count_a * count_b


def calculate_max_pairs_total_digits(total_digits: int) -> int:
    """Calculate total valid pairs (a, b) where sum of digits is total_digits."""
    if total_digits < 2:
        return 0
    max_pairs = 0
    for digits_a in range(1, total_digits):
        digits_b = total_digits - digits_a
        max_pairs += calculate_max_pairs(digits_a, digits_b)
    return max_pairs


def bucket_by_digits(examples: Sequence[MultiplicationExample]) -> Dict[int, List[MultiplicationExample]]:
    buckets: Dict[int, List[MultiplicationExample]] = defaultdict(list)
    for example in examples:
        buckets[example.digits].append(example)
    return buckets


def build_length_bucket_dataset(
    min_digits: int, # represents the sum of the digits in the two numbers (e.g. 3 could be (1,2) or (2,1))
    max_digits: int,
    per_digit_counts: Dict[str, int], # e.g. {"train": 1000, "validation": 200, "test": 200}
    rng: random.Random,
    *,
    exclude_pairs: Optional[set[tuple[Tuple[int, int], int, int]]] = None,
    record_pairs: Optional[Dict[str, set[tuple[Tuple[int, int], int, int]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
) -> Dict[str, List[MultiplicationExample]]:
    splits = {key: [] for key in ["train", "validation", "test"]}
    occupied = set(exclude_pairs) if exclude_pairs else set()
    used_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1
    split_order = ("train", "validation", "test")
    for digits in range(min_digits, max_digits + 1):
        # count how many examples we need for this digit bucket, and how many unique pairs we have available
        per_digit_per_split = {split: per_digit_counts.get(split, 0) for split in split_order}
        total_requested = sum(per_digit_per_split.values()) * (digits - 1)
        if total_requested == 0:
            continue
        max_pairs = calculate_max_pairs_total_digits(digits)
        available_unique = max(0, max_pairs - used_counts.get(digits, 0))
        # logic for fairly reducing counts across splits when we don't have enough unique pairs to satisfy the requested counts
        if available_unique < total_requested:
            print(
                f"[WARN] Requested {total_requested} examples for digits={digits} exceeds available unique pairs ({available_unique}); capping counts.",
                flush=True,
            )
            remaining = available_unique
            for split in split_order:
                requested = per_digit_per_split[split] * (digits - 1)
                if requested > remaining:
                    per_digit_per_split[split] = remaining
                    remaining = 0
                else:
                    remaining -= requested
            total_requested = sum(per_digit_per_split.values())
        # start the generation process for this digit bucket
        digit_examples: List[Tuple[MultiplicationExample, Tuple[Tuple[int, int], int, int], bool]] = []
        attempts = 0
        duplicates_allowed = False
        all_possible_digit_allocations = [(a, digits - a) for a in range(1, digits)]
        while len(digit_examples) < total_requested:
            attempts += 1
            example_digit_allocation = all_possible_digit_allocations[len(digit_examples) % len(all_possible_digit_allocations)]
            example = generate_multiplication_pair(*example_digit_allocation, rng=rng)
            key = example_key(example)
            if key in occupied:
                if attempts >= max_attempts:
                    if not duplicates_allowed:
                        print(
                            f"[WARN] Exhausted unique sampling for digits={digits} (progress={progress_name}); "
                            "allowing duplicates.",
                            flush=True,
                        )
                        duplicates_allowed = True
                    digit_examples.append((example, key, True))
                    attempts = 0
                continue
            occupied.add(key)
            used_counts[digits] += 1
            digit_examples.append((example, key, False))
            attempts = 0
        # allocate examples to splits according to the per-digit counts, and record pairs if requested
        idx = 0
        for split in split_order:
            count = per_digit_per_split.get(split, 0)
            if count <= 0:
                continue
            chunk = digit_examples[idx : idx + count]
            idx += count
            splits[split].extend(ex for ex, _, _ in chunk)
            if record_pairs and split in record_pairs:
                for _, key, is_dup in chunk:
                    if not is_dup:
                        record_pairs[split].add(key)
            if progress_name:
                print(
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' digits={digits}",
                    flush=True,
                )
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def build_composed_datasets(
    # base_splits: Dict[str, List[MultiplicationExample]], # we don't compose examples within the initial buckets, but instead compose new examples from E_{n-1,m} and E_{1,m} extension sets.
    min_digits: int,
    max_digits: int,
    per_digit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_pairs: Optional[set[tuple[Tuple[int, int], int, int]]] = None,
    record_pairs: Optional[Dict[str, set[tuple[Tuple[int, int], int, int]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    record_components: Optional[Dict[str, Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]]]] = None,
) -> Dict[str, List[MultiplicationExample]]:
    """Construct compositional datasets from the learned E_{n,m} sets"""
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_pairs) if exclude_pairs else set()
    used_counts: Dict[Tuple[Tuple[int, int], int, int], int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1
    for split in ("train", "validation", "test"):
        requested_per_digit = per_digit_counts.get(split, 0)
        if requested_per_digit == 0:
            continue
        component_map = None
        if record_components is not None:
            component_map = record_components.setdefault(split, {})
        # generated: (example, key, is_duplicate, component_keys)
        generated: List[Tuple[MultiplicationExample, Tuple[int, int, int], bool, List[Tuple[int, int, int]]]] = []
        per_digit_targets: Dict[int, int] = {}
        digit_schedule: List[int] = []
        for digits in range(min_digits, max_digits + 1):
            _requested_per_digit = requested_per_digit * (digits - 1)  # each composed example with total digits n has totally E_{i,j} i+j=n different sets.
            max_pairs = calculate_max_pairs_total_digits(digits)
            available_unique = max(0, max_pairs - used_counts.get(digits, 0))
            effective_target = min(_requested_per_digit, available_unique)
            if effective_target < _requested_per_digit:
                print(
                    f"[WARN] Requested {_requested_per_digit} composed examples for digits={digits} split='{split}' exceeds available unique pairs ({available_unique}); capping.",
                    flush=True,
                )
            if effective_target <= 0:
                continue
            per_digit_targets[digits] = effective_target
            digit_schedule.extend([digits] * effective_target)
        if not digit_schedule:
            continue
        rng.shuffle(digit_schedule)
        duplicates_allowed: Dict[int, bool] = defaultdict(bool)
        per_digit_generated: Dict[int, int] = defaultdict(int)
        for digits in digit_schedule:
            all_possible_digit_allocations = [(a, digits - a) for a in range(1, digits)]
            attempts = 0
            while True:
                attempts += 1
                component_list: List[MultiplicationExample] = []
                # draw_digits = rng.choice(all_possible_digit_allocations)
                draw_digits = all_possible_digit_allocations[per_digit_generated[digits] % len(all_possible_digit_allocations)] # round-robin
                is_reversed = draw_digits[0] < draw_digits[1]
                # don't do stitch from initial buckets anymore, but directly construct the composed example from available E_{n-1,m} and E_{1,m} examples, and check if the composed example is valid (i.e. not in occupied)
                draw_digits = sorted(draw_digits, reverse=True)  # ensure we always draw (n-1,m) and (1,m) in the same order for consistent composition
                composed_example, components_list = generate_composable_multiplication_pairs(*draw_digits, rng)
                if is_reversed:
                    composed_example = MultiplicationExample(
                        a=composed_example.b,
                        b=composed_example.a,
                        result=composed_example.result,
                        digits=(composed_example.digits[1], composed_example.digits[0]),
                    )
                    components_list = [
                        MultiplicationExample(
                            a=components_list[0].b,
                            b=components_list[0].a,
                            result=components_list[0].result,
                            digits=(components_list[0].digits[1], components_list[0].digits[0]),
                        ),
                        MultiplicationExample(
                            a=components_list[1].b,
                            b=components_list[1].a,
                            result=components_list[1].result,
                            digits=(components_list[1].digits[1], components_list[1].digits[0]),
                        ),
                    ]
                component_list = components_list
                key = example_key(composed_example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed[digits]:
                            print(
                                f"[WARN] Exhausted unique composed sampling for digits={digits} split='{split}' "
                                f"(progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed[digits] = True
                        generated.append((composed_example, key, True, [example_key(c) for c in component_list]))
                        per_digit_generated[digits] += 1
                        break
                    continue
                occupied.add(key)
                used_counts[digits] += 1
                generated.append((composed_example, key, False, [example_key(c) for c in component_list]))
                per_digit_generated[digits] += 1
                break
        if not generated:
            continue
        splits[split].extend(ex for ex, _, _, _ in generated)
        if record_pairs and split in record_pairs:
            for _, key, is_dup, _ in generated:
                if not is_dup:
                    record_pairs[split].add(key)
        if component_map is not None:
            for ex, key, _, component_keys in generated:
                component_map[key] = component_keys
        if progress_name:
            for digits, count in per_digit_generated.items():
                target = per_digit_targets.get(digits, count)
                print(
                    f"[INFO] Generated {count}/{target} {progress_name} examples for split='{split}' digits={digits} (dynamic)",
                    flush=True,
                )
        rng.shuffle(splits[split])
    return splits


def build_shapes_bucket_dataset(
    shapes: Sequence[Tuple[int, int]],
    per_shape_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_pairs: Optional[set[Tuple[Tuple[int, int], int, int]]] = None,
    record_pairs: Optional[Dict[str, set[Tuple[Tuple[int, int], int, int]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
) -> Dict[str, List[MultiplicationExample]]:
    """Generate raw multiplication examples for an explicit list of (n, m) shapes.

    Counterpart to `build_length_bucket_dataset` that operates on exact shapes
    instead of digit-sum buckets. Used by curriculum mode for initial training
    (all (i, j) with i, j <= initial_n) and shape-filtered evaluation.
    """
    splits: Dict[str, List[MultiplicationExample]] = {k: [] for k in ("train", "validation", "test")}
    occupied = set(exclude_pairs) if exclude_pairs else set()
    split_order = ("train", "validation", "test")
    for shape in shapes:
        n, m = shape
        duplicates_allowed = False
        for split in split_order:
            requested = per_shape_counts.get(split, 0)
            if requested <= 0:
                continue
            attempts = 0
            generated = 0
            while generated < requested:
                attempts += 1
                example = generate_multiplication_pair(n, m, rng=rng)
                key = example_key(example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique sampling for shape={shape} "
                                f"(progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        splits[split].append(example)
                        generated += 1
                        attempts = 0
                    continue
                occupied.add(key)
                splits[split].append(example)
                if record_pairs is not None and split in record_pairs:
                    record_pairs[split].add(key)
                generated += 1
                attempts = 0
            if progress_name:
                print(
                    f"[INFO] Generated {generated}/{requested} {progress_name} examples "
                    f"for shape={shape} split='{split}'",
                    flush=True,
                )
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def build_shapes_composed_dataset(
    shapes: Sequence[Tuple[int, int]],
    per_shape_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_pairs: Optional[set[Tuple[Tuple[int, int], int, int]]] = None,
    record_pairs: Optional[Dict[str, set[Tuple[Tuple[int, int], int, int]]]] = None,
    record_components: Optional[Dict[str, Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
) -> Dict[str, List[MultiplicationExample]]:
    """Generate composed examples for an explicit list of (n, m) shapes via auto-axis decomposition.

    Shape (1, 1) is skipped. Components are emitted as [high_part, low_part] so
    `target_w_base_predictions` / `target_w_component_map` produce the correct
    `{pred0 * 10} + {pred1} = {final}` chain regardless of which axis was used.
    """
    splits: Dict[str, List[MultiplicationExample]] = {k: [] for k in ("train", "validation", "test")}
    occupied = set(exclude_pairs) if exclude_pairs else set()
    split_order = ("train", "validation", "test")
    for shape in shapes:
        n, m = shape
        if n == 1 and m == 1:
            if progress_name:
                print(
                    f"[WARN] Skipping shape=(1,1) in composed dataset (progress={progress_name}): cannot decompose.",
                    flush=True,
                )
            continue
        duplicates_allowed = False
        for split in split_order:
            requested = per_shape_counts.get(split, 0)
            if requested <= 0:
                continue
            attempts = 0
            generated = 0
            while generated < requested:
                attempts += 1
                composed, components = generate_composable_multiplication_pairs_auto(n, m, rng)
                key = example_key(composed)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique composed sampling for shape={shape} "
                                f"(progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        splits[split].append(composed)
                        if record_components is not None:
                            record_components.setdefault(split, {})[key] = [example_key(c) for c in components]
                        generated += 1
                        attempts = 0
                    continue
                occupied.add(key)
                splits[split].append(composed)
                if record_pairs is not None and split in record_pairs:
                    record_pairs[split].add(key)
                if record_components is not None:
                    record_components.setdefault(split, {})[key] = [example_key(c) for c in components]
                generated += 1
                attempts = 0
            if progress_name:
                print(
                    f"[INFO] Generated {generated}/{requested} {progress_name} composed examples "
                    f"for shape={shape} split='{split}'",
                    flush=True,
                )
    for split in splits:
        rng.shuffle(splits[split])
    return splits


def build_initial_ground_truth_cot_artifacts(
    examples: Sequence[MultiplicationExample],
) -> Tuple[
    Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]],
    Dict[Tuple[Tuple[int, int], int, int], str],
]:
    """Build component_map and ground-truth base_predictions for initial
    examples with both digits >= 2, using the schoolbook decomposition.

    Mirrors `generate_schoolbook_multiplication_pairs`: for n >= m splits b
    into m digits (components of shape (n, 1)); for n < m splits a into n
    digits (components of shape (1, m)). Components are emitted high-to-low so
    component i contributes `pred_i * 10^(k-1-i)`. Fed to
    `TokenizedMultiplicationDataset`, this makes `target_w_base_predictions`
    emit ground-truth CoT chains for eligible initial samples.
    """
    component_map: Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]] = {}
    base_predictions: Dict[Tuple[Tuple[int, int], int, int], str] = {}
    for example in examples:
        n, m = example.digits
        if n < 2 or m < 2:
            continue
        a, b = example.a, example.b
        if n >= m:
            component_digits = (n, 1)
            slices = [(b // 10 ** i) % 10 for i in range(m - 1, -1, -1)]
            component_keys = [(component_digits, a, bi) for bi in slices]
            for bi in slices:
                base_predictions[(component_digits, a, bi)] = str(a * bi)
        else:
            component_digits = (1, m)
            slices = [(a // 10 ** i) % 10 for i in range(n - 1, -1, -1)]
            component_keys = [(component_digits, ai, b) for ai in slices]
            for ai in slices:
                base_predictions[(component_digits, ai, b)] = str(ai * b)
        component_map[example_key(example)] = component_keys
    return component_map, base_predictions


# diamond operator
def _build_generation_encodings(
    tokenizer: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Mirrors `self.self_improvement_core.build_generation_encodings`.

    Inlined here so the multiplication pipeline can run inference through the
    arithmetic-character tokenizer without importing the full self-improvement
    core module.
    """
    if not prompts:
        raise ValueError("Expected at least one prompt for generation.")
    prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    if tokenizer.bos_token_id is not None:
        prompt_token_ids = [[int(tokenizer.bos_token_id), *ids] for ids in prompt_token_ids]
    max_length = max(len(ids) for ids in prompt_token_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs pad_token_id or eos_token_id for generation padding.")
    padding_side = getattr(tokenizer, "padding_side", "right")
    batch_input_ids: List[List[int]] = []
    batch_attention: List[List[int]] = []
    for ids in prompt_token_ids:
        pad_count = max_length - len(ids)
        if padding_side == "left":
            batch_input_ids.append([pad_token_id] * pad_count + ids)
            batch_attention.append([0] * pad_count + [1] * len(ids))
        else:
            batch_input_ids.append(ids + [pad_token_id] * pad_count)
            batch_attention.append([1] * len(ids) + [0] * pad_count)
    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long, device=device),
    }


def _addition_model_pairwise_predict(
    addition_model: Any,
    addition_tokenizer: Any,
    pairs: Sequence[Tuple[int, int]],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> List[Optional[int]]:
    """Run the addition language model on a batch of (a, b) integer pairs.

    Returns a list of integer predictions (or None when parsing fails) aligned
    with the input pairs. Identical pairs are deduplicated before generation so
    repeated additions across a level cost a single forward pass; greedy
    decoding makes this cache-safe.
    """
    if not pairs:
        return []
    # Imported lazily so the multiplication pipeline doesn't pull recipe deps
    # unless an addition model is actually being used.
    from self.addition_recipe import tokenizer_padding_side
    from core.addition_pipeline import extract_numeric_answer as extract_addition_numeric_answer

    canonical_pairs: Dict[Tuple[int, int], int] = {}
    unique_pairs: List[Tuple[int, int]] = []
    pair_to_unique: List[int] = []
    for a, b in pairs:
        key = (int(a), int(b))
        idx = canonical_pairs.get(key)
        if idx is None:
            idx = len(unique_pairs)
            canonical_pairs[key] = idx
            unique_pairs.append(key)
        pair_to_unique.append(idx)

    device = next(addition_model.parameters()).device
    unique_results: List[Optional[int]] = [None] * len(unique_pairs)
    with tokenizer_padding_side(addition_tokenizer, "left"), torch.inference_mode():
        for start in range(0, len(unique_pairs), batch_size):
            chunk = unique_pairs[start : start + batch_size]
            prompts = [f"Q: {a} + {b} = ?\nA:" for a, b in chunk]
            enc = _build_generation_encodings(addition_tokenizer, prompts, device)
            out = addition_model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            prompt_width = enc["input_ids"].shape[1]
            for offset, ids in enumerate(out):
                generated = ids[prompt_width:]
                decoded = addition_tokenizer.decode(generated, skip_special_tokens=True)
                pred_str = extract_addition_numeric_answer(decoded)
                if pred_str is None:
                    continue
                try:
                    unique_results[start + offset] = int(pred_str)
                except ValueError:
                    continue
    return [unique_results[idx] for idx in pair_to_unique]


def build_composed_pseudo_map(
    base_map: Dict[Tuple[Tuple[int, int], int, int], str],
    composed_examples: Sequence[MultiplicationExample],
    component_map: Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]],
    weak_predictions: Dict[Tuple[Tuple[int, int], int, int], str],
    *,
    addition_model: Optional[Any] = None,
    addition_tokenizer: Optional[Any] = None,
    addition_batch_size: int = 64,
    addition_max_new_tokens: int = 48,
    rng: Optional[random.Random] = None,
) -> Dict[Tuple[Tuple[int, int], int, int], str]:
    """Return pseudo labels for composed examples by stitching weak predictions on their components.

    Components are in high-to-low order; component i contributes
    `pred_i * 10^(k-1-i)` where k = len(components), matching the
    schoolbook CoT format used by `target_w_base_predictions`. When
    `addition_model` and `addition_tokenizer` are provided, the per-example
    summation `pred_0 * 10^(k-1) + ... + pred_{k-1}` is evaluated by tree-style
    pairwise addition through the supplied addition LM: each level pairs
    adjacent partials (carrying a lone trailing partial when the count is odd)
    and runs one batched generation across all still-active candidates.
    Sequential depth is `ceil(log2(k))` instead of `k - 1`. Identical pairs
    inside a level are deduplicated before generation. If no model is supplied,
    the sum is computed with native integer arithmetic.
    """
    pseudo_map = dict(base_map)

    candidates: List[Tuple[Tuple[Tuple[int, int], int, int], List[int]]] = []
    for example in composed_examples:
        key = example_key(example)
        component_keys = component_map.get(key)
        if not component_keys:
            continue
        preds: List[str] = []
        missing = False
        for comp_key in component_keys:
            pred = weak_predictions.get(comp_key)
            if pred is None:
                missing = True
                break
            preds.append(pred)
        if missing or not preds:
            continue
        k = len(preds)
        try:
            partials = [int(preds[i]) * 10 ** (k - 1 - i) for i in range(k)]
        except ValueError:
            continue
        candidates.append((key, partials))

    if not candidates:
        return pseudo_map

    if addition_model is None or addition_tokenizer is None:
        for key, partials in candidates:
            pseudo_map[key] = str(sum(partials))
        return pseudo_map

    # Tree-style pairwise reduction through the addition LM. Each iteration
    # collapses adjacent pairs at every still-active example, all summed in a
    # single batched call; the depth of the chain shrinks logarithmically.
    levels: List[Optional[List[int]]] = [list(partials) for _, partials in candidates]

    while any(parts is not None and len(parts) > 1 for parts in levels):
        pairs: List[Tuple[int, int]] = []
        pair_targets: List[Tuple[int, int]] = []
        next_levels: List[Optional[List[int]]] = []
        for example_idx, parts in enumerate(levels):
            if parts is None or len(parts) <= 1:
                next_levels.append(parts)
                continue
            new_parts: List[Optional[int]] = []
            j = 0
            while j + 1 < len(parts):
                pair_targets.append((example_idx, len(new_parts)))
                pairs.append((parts[j], parts[j + 1]))
                new_parts.append(None)
                j += 2
            if j < len(parts):
                new_parts.append(parts[j])
            next_levels.append(new_parts)

        if not pairs:
            break

        sums = _addition_model_pairwise_predict(
            addition_model,
            addition_tokenizer,
            pairs,
            batch_size=addition_batch_size,
            max_new_tokens=addition_max_new_tokens,
        )
        for (example_idx, write_idx), predicted in zip(pair_targets, sums):
            slot = next_levels[example_idx]
            if slot is None:
                continue
            if predicted is None:
                next_levels[example_idx] = None
            else:
                slot[write_idx] = predicted
        levels = next_levels

    for (key, _), parts in zip(candidates, levels):
        if not parts or any(part is None for part in parts):
            continue
        pseudo_map[key] = str(parts[0])
    return pseudo_map


def _ddp_world() -> Tuple[int, int]:
    """Return (rank, world_size). Falls back to (0, 1) when DDP is not active."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def generate_prediction_map(
    model: AutoModelForCausalLM,
    tokenizer,
    examples: Sequence[MultiplicationExample],
    batch_size: int,
    max_new_tokens: int,
) -> Dict[tuple[Tuple[int, int], int, int], str]:
    """DDP-aware: each rank decodes a stride of the deduplicated example set,
    then results are merged across ranks with `all_gather_object` so every
    process returns the full prediction map. World-size 1 keeps the original
    single-process path."""
    rank, world = _ddp_world()

    unique: Dict[tuple[Tuple[int, int], int, int], MultiplicationExample] = {}
    for example in examples:
        key = example_key(example)
        if key not in unique:
            unique[key] = example

    keys = list(unique.keys())
    shard_keys = keys[rank::world]
    shard_values = [unique[k] for k in shard_keys]

    local_predictions: Dict[tuple[Tuple[int, int], int, int], str] = {}
    if shard_values:
        device = next(model.parameters()).device
        model_was_training = model.training
        model.eval()
        with torch.no_grad():
            for start in range(0, len(shard_values), batch_size):
                batch = shard_values[start : start + batch_size]
                prompts = [ex.prompt() for ex in batch]
                encodings = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                encodings = {k: v.to(device) for k, v in encodings.items()}
                output_ids = model.generate(
                    **encodings,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    use_cache=True,
                )
                input_lengths = encodings["attention_mask"].sum(dim=1)
                for idx, example in enumerate(batch):
                    generated_slice = output_ids[idx, input_lengths[idx] :].tolist()
                    text = tokenizer.decode(generated_slice, skip_special_tokens=True)
                    pred = extract_numeric_answer(text)
                    if pred is not None:
                        local_predictions[example_key(example)] = pred.strip()
        if model_was_training:
            model.train()

    if world == 1:
        return local_predictions

    gathered: List[Optional[Dict[tuple[Tuple[int, int], int, int], str]]] = [None] * world
    torch.distributed.all_gather_object(gathered, local_predictions)
    merged: Dict[tuple[Tuple[int, int], int, int], str] = {}
    for piece in gathered:
        if piece:
            merged.update(piece)
    return merged


def clone_with_override(example: MultiplicationExample, override: Optional[str]) -> MultiplicationExample:
    if override is None:
        return example
    return MultiplicationExample(
        a=example.a,
        b=example.b,
        result=example.result,
        digits=example.digits,
        target_override=override,
    )


class TokenizedMultiplicationDataset(Dataset):
    """Lazily tokenized dataset for causal LM fine-tuning."""

    def __init__(
        self,
        examples: Sequence[MultiplicationExample],
        component_map: Dict[tuple[Tuple[int, int], int, int], List[tuple[Tuple[int, int], int, int]]],
        tokenizer,
        add_eos: bool = True,
        base_predictions: Optional[Dict[Tuple[Tuple[int, int], int, int], str]] = None,
    ):
        self.tokenizer = tokenizer
        self.examples = list(examples)
        self.component_map = component_map # only for composed examples
        self.add_eos = add_eos
        self.base_predictions = base_predictions

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        prompt = example.prompt()
        if self.base_predictions is not None:
            target_text = f" {example.target_w_base_predictions(self.component_map, self.base_predictions)}"
        else:
            target_text = f" {example.target()}"

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)

        input_ids: List[int] = []
        labels: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            labels.append(-100)

        input_ids.extend(prompt_ids)
        labels.extend([-100] * len(prompt_ids))

        input_ids.extend(target_ids)
        labels.extend(target_ids)

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def evaluate_accuracy_with_breakdown(
    model: AutoModelForCausalLM,
    tokenizer,
    examples: Sequence[MultiplicationExample],
    batch_size: int,
    max_new_tokens: int,
    *,
    return_details: bool = False,
    debug_interval: Optional[int] = None,
    debug_label: Optional[str] = None,
    accept_any_numeric_match: bool = False,
) -> tuple[float, Dict[int, float]]:
    """Compute accuracy plus per-digit breakdown via greedy decoding.

    When `return_details` is True, a third value is returned containing per-example
    prediction metadata useful for inspection or serialization. When
    `debug_interval` is set to a positive integer, periodic progress prints are
    emitted every `debug_interval` examples.

    DDP-aware: under an initialized process group, each rank decodes a stride
    of `examples` (rank `r` takes positions `r, r+world, ...`) and the per-digit
    counts plus per-example details are aggregated via `all_gather_object`. With
    `world_size == 1` the original single-process behavior is preserved.
    """
    if not examples:
        if return_details:
            return math.nan, {}, []
        return math.nan, {}

    rank, world = _ddp_world()
    indexed_shard = list(enumerate(examples))[rank::world]

    device = next(model.parameters()).device

    model_was_training = model.training
    model.eval()
    correct = 0
    digit_totals: Dict[Tuple[int, int], int] = defaultdict(int)
    digit_correct: Dict[Tuple[int, int], int] = defaultdict(int)
    details: List[Dict[str, Any]] = []

    debug_interval = debug_interval if debug_interval and debug_interval > 0 else None
    label = debug_label or "eval"

    with torch.no_grad():
        for start in range(0, len(indexed_shard), batch_size):
            batch_pairs = indexed_shard[start : start + batch_size]
            if not batch_pairs:
                continue
            prompts = [ex.prompt() for _, ex in batch_pairs]
            encodings = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encodings = {k: v.to(device) for k, v in encodings.items()}
            output_ids = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                use_cache=True,
            )
            input_lengths = encodings["attention_mask"].sum(dim=1)
            for idx, (global_index, example) in enumerate(batch_pairs):
                example_number = global_index + 1
                digit_totals[example.digits] += 1
                generated_slice = output_ids[idx, input_lengths[idx] :].tolist()
                text = tokenizer.decode(generated_slice, skip_special_tokens=True)
                pred_str = extract_numeric_answer(text)
                target_str = example.target()
                if accept_any_numeric_match:
                    numeric_tokens = [match.strip() for match in NUMERIC_PATTERN.findall(text)]
                    is_correct = pred_str == target_str or any(token == target_str for token in numeric_tokens)
                else:
                    numeric_tokens = None
                    is_correct = pred_str == target_str
                if is_correct:
                    correct += 1
                    digit_correct[example.digits] += 1
                if return_details:
                    if numeric_tokens is None:
                        numeric_tokens = [match.strip() for match in NUMERIC_PATTERN.findall(text)]
                    details.append(
                        {
                            "index": global_index,
                            "example_number": example_number,
                            "digits": example.digits,
                            "a": example.a,
                            "b": example.b,
                            "prompt": example.prompt(),
                            "target": example.target(),
                            "prediction": pred_str,
                            "generated_text": text.strip(),
                            "is_correct": is_correct,
                            "all_predictions": numeric_tokens,
                        }
                    )
                if debug_interval and example_number % debug_interval == 0:
                    preview = (text or "").strip().replace("\n", " ")
                    if len(preview) > 80:
                        preview = preview[:77] + "..."
                    rank_tag = f"][rank={rank}" if world > 1 else ""
                    print(
                        f"[DEBUG][{label}{rank_tag}] idx={example_number} digits={example.digits} "
                        f"target={example.target()} pred={pred_str} correct={is_correct} text='{preview}'",
                        flush=True,
                    )

    if model_was_training:
        model.train()

    if world > 1:
        payload: Dict[str, Any] = {
            "correct": int(correct),
            "digit_totals": dict(digit_totals),
            "digit_correct": dict(digit_correct),
            "details": details if return_details else None,
        }
        gathered: List[Optional[Dict[str, Any]]] = [None] * world
        torch.distributed.all_gather_object(gathered, payload)

        correct = sum(int(piece["correct"]) for piece in gathered if piece)
        digit_totals = defaultdict(int)
        digit_correct = defaultdict(int)
        for piece in gathered:
            if not piece:
                continue
            for k, v in piece["digit_totals"].items():
                digit_totals[k] += int(v)
            for k, v in piece["digit_correct"].items():
                digit_correct[k] += int(v)
        if return_details:
            details = []
            for piece in gathered:
                if piece and piece["details"]:
                    details.extend(piece["details"])
            details.sort(key=lambda d: d["index"])

    total = sum(digit_totals.values())
    overall_accuracy = correct / total if total > 0 else math.nan
    per_digit_accuracy = {
        digits: digit_correct[digits] / count if count > 0 else math.nan
        for digits, count in digit_totals.items()
    }
    if return_details:
        return overall_accuracy, per_digit_accuracy, details
    return overall_accuracy, per_digit_accuracy


def cot_target_length_upper_bound(digits_n: int, digits_m: int) -> int:
    """Worst-case character length of the schoolbook CoT chain for an E_{n,m} example.

    Mirrors `MultiplicationExample.target_w_base_predictions`: decomposes along
    the shorter dimension into k = min(n, m) components; component i contributes
    `pred_i * 10^(k-1-i)` where pred_i is at most max(n, m) + 1 digits (since
    (10^L - 1) * 9 has L+1 digits). Returns 0 when CoT is not emitted (a side
    < 2 digits), so the plain-target length governs in that case.
    """
    if digits_n < 2 or digits_m < 2:
        return 0
    k = min(digits_n, digits_m)
    long_side = max(digits_n, digits_m)
    term_chars = sum((long_side + 1) + (k - 1 - i) for i in range(k))
    sep_chars = 3 * (k - 1) + 3  # " + " between terms, plus " = " before final
    final_chars = digits_n + digits_m  # a*b has at most n + m digits
    return term_chars + sep_chars + final_chars


def resolve_max_new_tokens(
    examples: Sequence[MultiplicationExample],
    base_value: int,
    buffer: int = 2,
    cot_buffer: int = 12,
) -> int:
    """Return a decoding budget large enough for the longest target plus a small buffer.

    For CoT-eligible shapes (both digits >= 2) the model is trained to emit the
    schoolbook chain `pred_0 * 10^(k-1) + ... = final`, which is much longer
    than `str(result)`. We size against an upper bound on that chain length so
    eval-time generation isn't truncated mid-chain (which would let
    `extract_numeric_answer` parse a partial product as the final answer).

    `cot_buffer` is intentionally larger than `buffer` to absorb edge cases the
    analytical bound doesn't capture: the trailing EOS, BPE fragmentation of
    `" + "` / `" = "` separators into 2-3 tokens each, and components emitted
    with one extra digit (the model is trained on pseudo predictions, so it can
    inherit and reproduce slight digit overshoot at eval time). Plain-target
    callsites (e.g. component prediction over 1-digit-side shapes) keep the
    smaller `buffer` since they emit a single number with no separators.
    """
    if not examples:
        return base_value
    candidate = base_value
    for example in examples:
        plain_len = len(example.target())
        candidate = max(candidate, plain_len + buffer)
        cot_len = cot_target_length_upper_bound(example.digits[0], example.digits[1])
        if cot_len > 0:
            candidate = max(candidate, cot_len + cot_buffer)
    return candidate






if __name__ == "__main__":
    from beautify_print import bprint
    # Test 1: build_length_bucket_dataset returns the requested number of examples
    # per split, each example is a valid MultiplicationExample, and digit sums
    # fall within [min_digits, max_digits].
    rng = random.Random(42)
    record_pairs: Dict[str, set[tuple[int, int, int]]] = {
        "train": set(), "validation": set(), "test": set(),
    }
    bucket_splits = build_length_bucket_dataset(
        min_digits=2,
        max_digits=4,
        per_digit_counts={"train": 5, "validation": 2, "test": 2},
        rng=rng,
        record_pairs=record_pairs,
        progress_name="test_bucket",
    )
    print(bucket_splits)
    expected_per_split = {"train": 5 * 3, "validation": 2 * 3, "test": 2 * 3}
    for split, expected in expected_per_split.items():
        got = len(bucket_splits.get(split, []))
        assert got == expected, f"[TEST 1] split={split}: expected {expected}, got {got}"
        for ex in bucket_splits[split]:
            assert isinstance(ex, MultiplicationExample), (
                f"[TEST 1] split={split}: expected MultiplicationExample, got {type(ex).__name__}"
            )
            assert ex.a * ex.b == ex.result, f"[TEST 1] bad result for {ex}"
            digit_sum = ex.digits[0] + ex.digits[1]
            assert 2 <= digit_sum <= 4, f"[TEST 1] digit sum {digit_sum} out of range"
    print("[TEST 1] build_length_bucket_dataset passed.")

    # Test 2: build_composed_datasets yields examples whose a-digits are the
    # concatenation of the two components (E_{n-1,m}.a*10 + E_{1,m}.a) with a
    # shared b factor, and the number of generated examples matches the request.
    rng = random.Random(123)
    record_components: Dict[str, Dict[Tuple[Tuple[int, int], int, int], List[Tuple[Tuple[int, int], int, int]]]] = {}
    composed_splits = build_composed_datasets(
        min_digits=4,
        max_digits=4,
        per_digit_counts={"train": 3, "validation": 1, "test": 1},
        rng=rng,
        record_components=record_components,
        progress_name="test_composed",
    )
    bprint(composed_splits['test'])
    bprint(record_components)

    bprint(composed_splits['test'][0].prompt())
    res = composed_splits['test'][0].target_w_component_map(record_components['test'])
    bprint(res)
    print(extract_numeric_answer(res))
    # expected_composed = {"train": 3 * 1, "validation": 1 * 1, "test": 1 * 1}
    # for split, expected in expected_composed.items():
    #     entries = composed_splits.get(split, [])
    #     assert len(entries) == expected, (
    #         f"[TEST 2] split={split}: expected {expected}, got {len(entries)}"
    #     )
    #     for entry in entries:
    #         composed = entry
    #         components = record_components.get(split, {}).get(example_key(composed), [])
    #         assert isinstance(composed, MultiplicationExample), (
    #             f"[TEST 2] composed should be MultiplicationExample, got {type(composed).__name__}"
    #         )
    #         assert composed.a * composed.b == composed.result, f"[TEST 2] bad result for {composed}"
    #         assert len(components) == 2, f"[TEST 2] expected 2 components, got {len(components)}"
    #         print(f"Composed example: {composed}, components: {components}")
    #         e_nm1, e_1m = components
    #         assert composed.a == e_nm1.a * 10 + e_1m.a, (
    #             f"[TEST 2] composition failed: {composed.a} != {e_nm1.a}*10 + {e_1m.a}"
    #         )
    #         assert composed.b == e_nm1.b == e_1m.b, (
    #             f"[TEST 2] shared b factor mismatch: {composed.b}, {e_nm1.b}, {e_1m.b}"
    #         )
    # print("[TEST 2] build_composed_datasets passed.")

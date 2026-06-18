#!/usr/bin/env python3
"""Multiplication task adapter, data generation, and pseudolabel helpers."""

from __future__ import annotations

# --- from multiplication_data.py ---
from dataclasses import dataclass
from typing import Optional, Tuple

from self.tasks.bit import format_multiplication_target


MultiplicationKey = Tuple[int, int, int]


@dataclass(frozen=True)
class MultiplicationExample:
    a: int
    b: int
    digits: int
    result: int
    operand_width: int
    format_version: str = "legacy"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"{self.a:0{self.operand_width}d}×{self.b:0{self.operand_width}d}="
        return f"Q: {self.a:0{self.operand_width}d} * {self.b:0{self.operand_width}d} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_multiplication_target(self.result, self.digits, self.format_version)

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def multiplication_key(example: MultiplicationExample) -> MultiplicationKey:
    return example.digits, example.a, example.b


def encode_multiplication_key(key: MultiplicationKey) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def decode_multiplication_key(value: str) -> MultiplicationKey:
    digits, a, b = value.split("|", 2)
    return int(digits), int(a), int(b)


def clone_multiplication_with_override(
    example: MultiplicationExample,
    override: Optional[str],
) -> MultiplicationExample:
    if override is None:
        return example
    return MultiplicationExample(
        a=example.a,
        b=example.b,
        digits=example.digits,
        result=example.result,
        operand_width=example.operand_width,
        format_version=example.format_version,
        target_override=override,
    )


# --- from multiplication_sampling.py ---
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _example_cls() -> type["MultiplicationExample"]:

    return MultiplicationExample


def _multiplication_key(example: "MultiplicationExample") -> "MultiplicationKey":

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


# --- from multiplication_pseudolabels.py ---
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit import normalize_task_format_version
from self.tasks.bit import format_multiplication_target
from self.tasks.bit import build_direct_pseudo_examples

GeneratePredictionMap = Callable[..., Dict[Tuple[int, int, int], str]]


def _empty_pseudo_result(mode: str, candidate_total: int) -> Tuple[List[MultiplicationExample], int, JsonDict]:
    return [], 0, {
        "mode": mode,
        "candidate_total": candidate_total,
        "retained_total": 0,
        "missing_total": 0,
    }


def _component_from_partial(partial: Dict[str, Any], *, args: Any) -> MultiplicationExample:
    return MultiplicationExample(
        a=int(partial["a"]),
        b=int(partial["b"]),
        digits=args.block_size,
        result=int(partial["a"]) * int(partial["b"]),
        operand_width=args.block_size,
        format_version=normalize_task_format_version(args),
    )


def derive_multiplication_round_targets(
    *,
    model: Any,
    tokenizer: Any,
    composed_examples: Sequence[MultiplicationExample],
    component_map: Dict[Tuple[int, int, int], Dict[str, Any]],
    target_max_size: int,
    base_examples: Sequence[MultiplicationExample],
    batch_size: int,
    decode_max_new_tokens: int,
    args: Any,
    rng: random.Random,
    prediction_parser: Callable[[str, Optional[MultiplicationExample]], Optional[str]],
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[MultiplicationExample], int, JsonDict]:
    del base_examples
    candidate_examples = [example for example in composed_examples if example.digits <= target_max_size]
    if args.pseudo_label_mode == "direct":
        return build_direct_pseudo_examples(
            candidate_examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
            key_getter=multiplication_key,
            prediction_parser=prediction_parser,
            clone_builder=clone_multiplication_with_override,
            mode="direct",
        )
    if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
        return _empty_pseudo_result(args.pseudo_label_mode, len(candidate_examples))

    component_examples: Dict[Tuple[int, int, int], MultiplicationExample] = {}
    for example in candidate_examples:
        payload = component_map.get(multiplication_key(example))
        if payload is None:
            continue
        for partial in payload.get("partials", []):
            component = _component_from_partial(partial, args=args)
            component_examples[multiplication_key(component)] = component

    component_predictions = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=list(component_examples.values()),
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=multiplication_key,
        prediction_parser=prediction_parser,
    )

    pseudo_examples: List[MultiplicationExample] = []
    missing_total = 0
    corrupted_component_total = 0
    corrupted_example_total = 0

    for example in candidate_examples:
        payload = component_map.get(multiplication_key(example))
        if payload is None:
            missing_total += 1
            continue
        partial_predictions: List[Tuple[int, int]] = []
        example_corrupted = False
        missing = False
        for partial in payload.get("partials", []):
            component = _component_from_partial(partial, args=args)
            prediction = component_predictions.get(multiplication_key(component))
            if prediction is None:
                missing = True
                break
            numeric_prediction = int(prediction)
            if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
                numeric_prediction += 1
                corrupted_component_total += 1
                example_corrupted = True
            partial_predictions.append((numeric_prediction, int(partial["shift"])))
        if missing:
            missing_total += 1
            continue
        if example_corrupted:
            corrupted_example_total += 1
        composed_value = sum(value * (10**shift) for value, shift in partial_predictions)
        pseudo_examples.append(
            clone_multiplication_with_override(
                example,
                format_multiplication_target(composed_value, example.digits, example.format_version),
            )
        )

    diagnostics: JsonDict = {
        "mode": args.pseudo_label_mode,
        "target_max_digits": int(target_max_size),
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
        "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
        "corrupted_component_total": corrupted_component_total,
        "corrupted_example_total": corrupted_example_total,
    }
    return pseudo_examples, missing_total, diagnostics


# --- from multiplication_splits.py ---
import random
from typing import Any, Dict, List, Optional, Tuple

from self.tasks.bit import normalize_task_format_version

SplitName = str
ComponentMap = Dict[MultiplicationKey, Dict[str, Any]]


def _empty_split_records() -> Dict[SplitName, set[MultiplicationKey]]:
    return {"train": set(), "validation": set(), "test": set()}


def _empty_component_records() -> Dict[SplitName, ComponentMap]:
    return {"train": {}, "validation": {}, "test": {}}


def _effective_composed_min_size(args: Any, min_size: int) -> int:
    return max(
        args.block_size * 2,
        ((min_size + args.block_size - 1) // args.block_size) * args.block_size,
    )


def _used_base_keys(
    base_records: Dict[SplitName, set[MultiplicationKey]],
    additional_exclude: Optional[set[MultiplicationKey]],
) -> set[MultiplicationKey]:
    exclude = set().union(*base_records.values())
    if additional_exclude:
        exclude.update(additional_exclude)
    return exclude


def prepare_multiplication_initial_splits(
    rng: random.Random,
    args: Any,
) -> Tuple[Dict[SplitName, List[MultiplicationExample]], Dict[SplitName, set[MultiplicationKey]]]:
    records = _empty_split_records()
    if args.initial_min_size == args.block_size and args.initial_max_size == args.block_size:
        splits = build_multiplication_seed_dataset(
            block_size=args.block_size,
            per_split_counts={
                "train": args.initial_train_per_size,
                "validation": args.initial_eval_per_size,
                "test": args.initial_eval_per_size,
            },
            rng=rng,
            record_keys=records,
            progress_name="initial",
            format_version=normalize_task_format_version(args),
        )
    else:
        splits = build_multiplication_long_dataset(
            min_digits=args.initial_min_size,
            max_digits=args.initial_max_size,
            per_digit_counts={
                "train": args.initial_train_per_size,
                "validation": args.initial_eval_per_size,
                "test": args.initial_eval_per_size,
            },
            rng=rng,
            block_size=args.block_size,
            record_keys=records,
            progress_name="initial",
            format_version=normalize_task_format_version(args),
        )
    return splits, records


def prepare_multiplication_composed_train(
    rng: random.Random,
    args: Any,
    base_records: Dict[SplitName, set[MultiplicationKey]],
    min_size: int,
    max_size: int,
    additional_exclude: Optional[set[MultiplicationKey]] = None,
) -> Tuple[List[MultiplicationExample], ComponentMap, set[MultiplicationKey]]:
    if max_size < min_size or args.expand_train_per_size <= 0:
        return [], {}, set()
    effective_min_size = _effective_composed_min_size(args, min_size)
    if max_size < effective_min_size:
        return [], {}, set()
    composed_records = _empty_split_records()
    component_records = _empty_component_records()
    composed = build_multiplication_long_dataset(
        min_digits=effective_min_size,
        max_digits=max_size,
        per_digit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
        rng=rng,
        block_size=args.block_size,
        exclude_keys=_used_base_keys(base_records, additional_exclude),
        record_keys=composed_records,
        progress_name="composed",
        record_components=component_records,
        format_version=normalize_task_format_version(args),
    )
    return composed.get("train", []), component_records.get("train", {}), composed_records.get("train", set())


def prepare_multiplication_composed_eval(
    rng: random.Random,
    args: Any,
    base_records: Dict[SplitName, set[MultiplicationKey]],
    min_size: int,
    max_size: int,
    additional_exclude: Optional[set[MultiplicationKey]] = None,
) -> Tuple[List[MultiplicationExample], ComponentMap, set[MultiplicationKey]]:
    if max_size < min_size or args.composed_eval_per_size <= 0:
        return [], {}, set()
    effective_min_size = _effective_composed_min_size(args, min_size)
    if max_size < effective_min_size:
        return [], {}, set()
    composed_records = _empty_split_records()
    component_records = _empty_component_records()
    composed = build_multiplication_long_dataset(
        min_digits=effective_min_size,
        max_digits=max_size,
        per_digit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
        rng=rng,
        block_size=args.block_size,
        exclude_keys=_used_base_keys(base_records, additional_exclude),
        record_keys=composed_records,
        progress_name="composed-eval",
        record_components=component_records,
        format_version=normalize_task_format_version(args),
    )
    return composed.get("test", []), component_records.get("test", {}), composed_records.get("test", set())


def prepare_multiplication_eval_examples(
    rng: random.Random,
    args: Any,
    min_size: int,
    max_size: int,
    exclude: set[MultiplicationKey],
) -> List[MultiplicationExample]:
    generated = build_multiplication_long_dataset(
        min_digits=min_size,
        max_digits=max_size,
        per_digit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
        rng=rng,
        block_size=args.block_size,
        exclude_keys=exclude,
        progress_name="evaluation",
        format_version=normalize_task_format_version(args),
    )
    return list(generated.get("test", []))


def split_multiplication_composed_eval_slices(
    examples: List[MultiplicationExample],
    component_map: ComponentMap,
) -> Dict[str, List[MultiplicationExample]]:
    slices = {
        "low_overlap_low_carry": [],
        "low_overlap_high_carry": [],
        "high_overlap_low_carry": [],
        "high_overlap_high_carry": [],
        "unknown": [],
    }
    for example in examples:
        payload = component_map.get(multiplication_key(example))
        if payload is None:
            slices["unknown"].append(example)
            continue
        slice_name = get_multiplication_slice_name(payload, int(payload.get("block_size", 2)))
        slices[slice_name].append(example)
    return slices


__all__ = [
    "prepare_multiplication_composed_eval",
    "prepare_multiplication_composed_train",
    "prepare_multiplication_eval_examples",
    "prepare_multiplication_initial_splits",
    "split_multiplication_composed_eval_slices",
]


# --- from multiplication.py ---
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map as _default_generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask
from self.tasks.bit import normalize_task_format_version
from self.tasks.bit import (
    MULTIPLICATION_FORMATS,
    parse_multiplication_prediction,
)


SplitName = str


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.tasks")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def generate_prediction_map(**kwargs: Any) -> Dict[Any, str]:
    return _compat_symbol("generate_prediction_map", _default_generate_prediction_map)(**kwargs)


class MultiplicationTask(SelfImprovementTask):
    name = "multiplication"
    size_label = "digits"
    size_alias_singular = "digit"
    size_alias_plural = "digits"

    def validate_args(self, args: Any) -> None:
        if args.block_size <= 0:
            raise ValueError("block_size must be positive.")
        if args.initial_min_size < args.block_size:
            raise ValueError("initial_min_size must be >= block_size for multiplication.")
        if args.initial_max_size < args.initial_min_size:
            raise ValueError("initial_max_size must be >= initial_min_size for multiplication.")
        if args.expand_num_size % args.block_size != 0:
            raise ValueError("expand_num_size must be a multiple of block_size for blocked multiplication.")
        if args.corruption_rate < 0.0 or args.corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        format_version = normalize_task_format_version(args)
        if format_version not in MULTIPLICATION_FORMATS:
            raise ValueError(f"Unsupported multiplication format_version={format_version!r}.")

    def serialize_example(self, example: MultiplicationExample) -> JsonDict:
        return {
            "a": example.a,
            "b": example.b,
            "digits": example.digits,
            "result": example.result,
            "operand_width": example.operand_width,
            "format_version": example.format_version,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> MultiplicationExample:
        return MultiplicationExample(
            a=int(payload["a"]),
            b=int(payload["b"]),
            digits=int(payload["digits"]),
            result=int(payload["result"]),
            operand_width=int(payload["operand_width"]),
            format_version=str(payload.get("format_version", "legacy")),
            target_override=payload.get("target_override"),
        )

    def save_component_map(self, path: Path, component_map: Dict[Tuple[int, int, int], Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {encode_multiplication_key(key): value for key, value in component_map.items()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_multiplication_key(key): dict(value) for key, value in raw.items()}

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[MultiplicationExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
        return prepare_multiplication_initial_splits(rng, args)

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MultiplicationExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[MultiplicationExample], Dict[Tuple[int, int, int], Dict[str, Any]], set[Tuple[int, int, int]]]:
        del base_splits
        return prepare_multiplication_composed_train(
            rng,
            args,
            base_records,
            min_size,
            max_size,
            additional_exclude,
        )

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[MultiplicationExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[MultiplicationExample], Dict[Tuple[int, int, int], Dict[str, Any]], set[Tuple[int, int, int]]]:
        del base_splits
        return prepare_multiplication_composed_eval(
            rng,
            args,
            base_records,
            min_size,
            max_size,
            additional_exclude,
        )

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, int, int]],
    ) -> List[MultiplicationExample]:
        return prepare_multiplication_eval_examples(rng, args, min_size, max_size, exclude)

    def split_composed_eval_slices(
        self,
        examples: Sequence[MultiplicationExample],
        component_map: Dict[Tuple[int, int, int], Dict[str, Any]],
    ) -> Dict[str, List[MultiplicationExample]]:
        return split_multiplication_composed_eval_slices(list(examples), component_map)

    def keys_for_examples(self, examples: Sequence[MultiplicationExample]) -> set[Tuple[int, int, int]]:
        return {multiplication_key(example) for example in examples}

    def rebuild_records(
        self,
        splits: Dict[SplitName, List[MultiplicationExample]],
    ) -> Dict[SplitName, set[Tuple[int, int, int]]]:
        return {split: {multiplication_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: MultiplicationExample) -> Tuple[int, int, int]:
        return multiplication_key(example)

    def clone_with_override(self, example: MultiplicationExample, override: Optional[str]) -> MultiplicationExample:
        return clone_multiplication_with_override(example, override)

    def size_of(self, example: MultiplicationExample) -> int:
        return example.digits

    def prediction_parser(self, text: str, example: Optional[MultiplicationExample] = None) -> Optional[str]:
        return parse_multiplication_prediction(text, example)

    def token_initializers(self, args: Any) -> Dict[str, str]:
        if normalize_task_format_version(args) == "symbolic_v1":
            return {"×": "*"}
        return {}

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[MultiplicationExample],
        component_map: Dict[Tuple[int, int, int], Dict[str, Any]],
        target_max_size: int,
        base_examples: Sequence[MultiplicationExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[MultiplicationExample], int, JsonDict]:
        return derive_multiplication_round_targets(
            model=model,
            tokenizer=tokenizer,
            composed_examples=composed_examples,
            component_map=component_map,
            target_max_size=target_max_size,
            base_examples=base_examples,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
            args=args,
            rng=rng,
            prediction_parser=self.prediction_parser,
            generate_prediction_map_fn=generate_prediction_map,
        )

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "block_size": args.block_size,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "block_size": args.block_size,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "composed_max_digits": final_max_size,
            "format_version": normalize_task_format_version(args),
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_block_size = int(task_config.get("block_size", metadata.get("block_size", args.block_size)))
        if stored_block_size != args.block_size:
            raise ValueError("Stored multiplication dataset uses a different block_size.")
        stored_format = str(task_config.get("format_version", metadata.get("format_version", "legacy")))
        if stored_format != normalize_task_format_version(args):
            raise ValueError("Stored multiplication dataset uses a different format_version.")

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        return {
            "max_digits": summary.max_size,
            "per_digit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_digits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
        }

#!/usr/bin/env python3
"""Multiplication split and composed-evaluation preparation helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from self.tasks.bit_common import normalize_task_format_version
from self.tasks.multiplication_data import (
    MultiplicationExample,
    MultiplicationKey,
    build_multiplication_long_dataset,
    build_multiplication_seed_dataset,
    get_multiplication_slice_name,
    multiplication_key,
)

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

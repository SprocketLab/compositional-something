#!/usr/bin/env python3
"""Addition examples, dataset preparation, and slice helpers."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from core.addition_pipeline import (
    ADDITION_SAMPLING_MODES,
    ADDITION_SAMPLING_NATURAL,
    ADDITION_WIDTH_EXACT_DIGITS,
    ADDITION_WIDTH_FIXED_MIXED_PROMPT,
    ADDITION_WIDTH_MODES,
    COMPOSITION_PATH_MODES,
    COMPOSITION_PATH_RANDOM,
    AdditionExample,
    build_composed_datasets,
    build_composed_pseudo_map,
    build_length_bucket_dataset,
    clone_with_override,
    decode_key,
    encode_key,
    example_key,
    has_component_boundary_carry,
)

SplitName = str


def corrupt_numeric_target(value: str) -> str:
    return str(int(value) + 1)


def prepare_addition_initial_splits(
    rng: random.Random,
    min_digits: int,
    max_digits: int,
    train_per_digit: int,
    eval_per_digit: int,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    addition_sampling_mode: str = ADDITION_SAMPLING_NATURAL,
) -> Tuple[Dict[SplitName, List[AdditionExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
    splits = {name: [] for name in ("train", "validation", "test")}
    records: Dict[SplitName, set[Tuple[int, int, int]]] = {name: set() for name in splits}
    generated = build_length_bucket_dataset(
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={
            "train": train_per_digit,
            "validation": eval_per_digit,
            "test": eval_per_digit,
        },
        allow_carry=True,
        rng=rng,
        record_pairs=records,
        progress_name="initial",
        addition_width_mode=addition_width_mode,
        addition_sampling_mode=addition_sampling_mode,
    )
    for split in splits:
        splits[split] = generated.get(split, [])
    return splits, records


def prepare_addition_composed_train(
    rng: random.Random,
    base_splits: Dict[SplitName, List[AdditionExample]],
    base_records: Dict[SplitName, set[Tuple[int, int, int]]],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    allow_carry: bool,
    boundary_carry_policy: str = "any",
    additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    composition_path_mode: str = COMPOSITION_PATH_RANDOM,
) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
    if max_digits < min_digits or per_digit_count <= 0:
        return [], {}, set()
    composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
    component_records: Dict[SplitName, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    base_used = set().union(*base_records.values())
    if additional_exclude:
        base_used.update(additional_exclude)
    composed_splits = build_composed_datasets(
        base_splits=base_splits,
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": per_digit_count, "validation": 0, "test": 0},
        rng=rng,
        exclude_pairs=base_used,
        record_pairs=composed_records,
        progress_name="composed",
        record_components=component_records,
        allow_carry=allow_carry,
        allow_nocarry=True,
        boundary_carry_policy=boundary_carry_policy,
        addition_width_mode=addition_width_mode,
        composition_path_mode=composition_path_mode,
    )
    return composed_splits.get("train", []), component_records.get("train", {}), composed_records.get("train", set())


def prepare_addition_composed_eval(
    rng: random.Random,
    base_splits: Dict[SplitName, List[AdditionExample]],
    base_records: Dict[SplitName, set[Tuple[int, int, int]]],
    min_digits: int,
    max_digits: int,
    per_digit_count: int,
    additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    composition_path_mode: str = COMPOSITION_PATH_RANDOM,
) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
    if max_digits < min_digits or per_digit_count <= 0:
        return [], {}, set()
    composed_records: Dict[SplitName, set[Tuple[int, int, int]]] = {"train": set(), "validation": set(), "test": set()}
    component_records: Dict[SplitName, Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]] = {
        "train": {},
        "validation": {},
        "test": {},
    }
    base_used = set().union(*base_records.values())
    if additional_exclude:
        base_used.update(additional_exclude)
    stitched_base_splits = {
        "train": list(base_splits.get("train", [])),
        "validation": list(base_splits.get("train", [])),
        "test": list(base_splits.get("train", [])),
    }
    composed_splits = build_composed_datasets(
        base_splits=stitched_base_splits,
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": 0, "validation": 0, "test": per_digit_count},
        rng=rng,
        exclude_pairs=base_used,
        record_pairs=composed_records,
        progress_name="composed-eval",
        record_components=component_records,
        allow_carry=True,
        allow_nocarry=True,
        addition_width_mode=addition_width_mode,
        composition_path_mode=composition_path_mode,
    )
    return composed_splits.get("test", []), component_records.get("test", {}), composed_records.get("test", set())


def prepare_addition_eval_examples(
    rng: random.Random,
    min_digits: int,
    max_digits: int,
    per_digit: int,
    exclude: set[Tuple[int, int, int]],
    addition_width_mode: str = ADDITION_WIDTH_EXACT_DIGITS,
    addition_sampling_mode: str = ADDITION_SAMPLING_NATURAL,
) -> List[AdditionExample]:
    generated = build_length_bucket_dataset(
        min_digits=min_digits,
        max_digits=max_digits,
        per_digit_counts={"train": 0, "validation": 0, "test": per_digit},
        allow_carry=True,
        rng=rng,
        exclude_pairs=exclude,
        record_pairs={split: set() for split in ("train", "validation", "test")},
        progress_name="evaluation",
        addition_width_mode=addition_width_mode,
        addition_sampling_mode=addition_sampling_mode,
    )
    return list(generated.get("test", []))


def get_boundary_carry_status(
    example: AdditionExample,
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
) -> Optional[bool]:
    component_keys = component_map.get(example_key(example))
    if not component_keys:
        return None
    component_digits = [key[0] for key in component_keys]
    if len(component_digits) <= 1:
        return None
    if sum(component_digits) != example.digits:
        return None
    return has_component_boundary_carry(example, component_digits)


def split_addition_examples_by_boundary_status(
    examples: Sequence[AdditionExample],
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
) -> Dict[str, List[AdditionExample]]:
    slices: Dict[str, List[AdditionExample]] = {
        "boundary_carry": [],
        "no_boundary_carry": [],
        "unknown": [],
    }
    for example in examples:
        status = get_boundary_carry_status(example, component_map)
        if status is True:
            slices["boundary_carry"].append(example)
        elif status is False:
            slices["no_boundary_carry"].append(example)
        else:
            slices["unknown"].append(example)
    return slices

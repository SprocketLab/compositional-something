#!/usr/bin/env python3
"""Run-length split and composed-evaluation preparation helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.tasks.bit_common import (
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_symbol_alphabet_size,
    normalize_task_format_version,
)
from self.tasks.bit_composition import bit_composed_target_sizes_from_examples
from self.tasks.bit_parsing import RUN_LENGTH_ALPHABET_SYMBOLS
from self.tasks.bit_pseudolabels import guard_slice_partition, run_length_guard_accepts_true_components
from self.tasks.run_length_data import (
    RunLengthExample,
    build_run_length_composed_dataset,
    build_run_length_length_bucket_dataset,
    run_length_key,
)

SplitName = str
RunLengthKey = Tuple[int, str]


def _empty_split_records() -> Dict[SplitName, set[RunLengthKey]]:
    return {"train": set(), "validation": set(), "test": set()}


def _empty_component_records() -> Dict[SplitName, Dict[RunLengthKey, List[RunLengthKey]]]:
    return {"train": {}, "validation": {}, "test": {}}


def prepare_run_length_initial_splits(
    rng: random.Random,
    args: Any,
) -> Tuple[Dict[SplitName, List[RunLengthExample]], Dict[SplitName, set[RunLengthKey]]]:
    splits = {name: [] for name in ("train", "validation", "test")}
    records = _empty_split_records()
    split_order = ("validation", "test", "train") if getattr(args, "reserve_heldout_first", False) else (
        "train",
        "validation",
        "test",
    )
    generated = build_run_length_length_bucket_dataset(
        min_bits=args.initial_min_size,
        max_bits=args.initial_max_size,
        per_bit_counts={
            "train": args.initial_train_per_size,
            "validation": args.initial_eval_per_size,
            "test": args.initial_eval_per_size,
        },
        rng=rng,
        exclude_keys=getattr(args, "_initial_exclude_keys", None),
        record_keys=records,
        progress_name="initial",
        format_version=normalize_task_format_version(args),
        target_mode=normalize_bit_target_mode(args),
        alphabet=RUN_LENGTH_ALPHABET_SYMBOLS[:normalize_symbol_alphabet_size(args)],
        split_order=split_order,
    )
    for split in splits:
        splits[split] = generated.get(split, [])
    return splits, records


def _used_base_keys(
    base_records: Dict[SplitName, set[RunLengthKey]],
    additional_exclude: Optional[set[RunLengthKey]],
) -> set[RunLengthKey]:
    base_used = set().union(*base_records.values())
    if additional_exclude:
        base_used.update(additional_exclude)
    return base_used


def prepare_run_length_composed_train(
    rng: random.Random,
    args: Any,
    base_splits: Dict[SplitName, List[RunLengthExample]],
    base_records: Dict[SplitName, set[RunLengthKey]],
    min_size: int,
    max_size: int,
    additional_exclude: Optional[set[RunLengthKey]] = None,
) -> Tuple[List[RunLengthExample], Dict[RunLengthKey, List[RunLengthKey]], set[RunLengthKey]]:
    if max_size < min_size or args.expand_train_per_size <= 0:
        return [], {}, set()
    composed_records = _empty_split_records()
    component_records = _empty_component_records()
    base_used = _used_base_keys(base_records, additional_exclude)
    compose_arity = normalize_compose_arity(args)
    bit_composition_path_mode = normalize_bit_composition_path_mode(args)
    target_sizes = bit_composed_target_sizes_from_examples(
        base_splits.get("train", []),
        size_getter=lambda example: example.bits,
        min_size=min_size,
        max_size=max_size,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    if target_sizes is not None:
        train_examples: List[RunLengthExample] = []
        for bits in target_sizes:
            composed_splits = build_run_length_composed_dataset(
                base_splits=base_splits,
                min_bits=bits,
                max_bits=bits,
                per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
                rng=rng,
                exclude_keys=base_used,
                record_keys=composed_records,
                progress_name="composed",
                record_components=component_records,
                compose_arity=compose_arity,
                bit_composition_path_mode=bit_composition_path_mode,
            )
            train_examples.extend(composed_splits.get("train", []))
        return train_examples, component_records.get("train", {}), composed_records.get("train", set())
    composed_splits = build_run_length_composed_dataset(
        base_splits=base_splits,
        min_bits=min_size,
        max_bits=max_size,
        per_bit_counts={"train": args.expand_train_per_size, "validation": 0, "test": 0},
        rng=rng,
        exclude_keys=base_used,
        record_keys=composed_records,
        progress_name="composed",
        record_components=component_records,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    return composed_splits.get("train", []), component_records.get("train", {}), composed_records.get("train", set())


def prepare_run_length_composed_eval(
    rng: random.Random,
    args: Any,
    base_splits: Dict[SplitName, List[RunLengthExample]],
    base_records: Dict[SplitName, set[RunLengthKey]],
    min_size: int,
    max_size: int,
    additional_exclude: Optional[set[RunLengthKey]] = None,
) -> Tuple[List[RunLengthExample], Dict[RunLengthKey, List[RunLengthKey]], set[RunLengthKey]]:
    if max_size < min_size or args.composed_eval_per_size <= 0:
        return [], {}, set()
    composed_records = _empty_split_records()
    component_records = _empty_component_records()
    base_used = _used_base_keys(base_records, additional_exclude)
    stitched_base_splits = {
        "train": list(base_splits.get("train", [])),
        "validation": list(base_splits.get("train", [])),
        "test": list(base_splits.get("train", [])),
    }
    compose_arity = normalize_compose_arity(args)
    bit_composition_path_mode = normalize_bit_composition_path_mode(args)
    target_sizes = bit_composed_target_sizes_from_examples(
        stitched_base_splits.get("train", []),
        size_getter=lambda example: example.bits,
        min_size=min_size,
        max_size=max_size,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    if target_sizes is not None:
        test_examples: List[RunLengthExample] = []
        for bits in target_sizes:
            composed_splits = build_run_length_composed_dataset(
                base_splits=stitched_base_splits,
                min_bits=bits,
                max_bits=bits,
                per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
                rng=rng,
                exclude_keys=base_used,
                record_keys=composed_records,
                progress_name="composed-eval",
                record_components=component_records,
                compose_arity=compose_arity,
                bit_composition_path_mode=bit_composition_path_mode,
            )
            test_examples.extend(composed_splits.get("test", []))
        return test_examples, component_records.get("test", {}), composed_records.get("test", set())
    composed_splits = build_run_length_composed_dataset(
        base_splits=stitched_base_splits,
        min_bits=min_size,
        max_bits=max_size,
        per_bit_counts={"train": 0, "validation": 0, "test": args.composed_eval_per_size},
        rng=rng,
        exclude_keys=base_used,
        record_keys=composed_records,
        progress_name="composed-eval",
        record_components=component_records,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    return composed_splits.get("test", []), component_records.get("test", {}), composed_records.get("test", set())


def prepare_run_length_eval_examples(
    rng: random.Random,
    args: Any,
    min_size: int,
    max_size: int,
    exclude: set[RunLengthKey],
) -> List[RunLengthExample]:
    generated = build_run_length_length_bucket_dataset(
        min_bits=min_size,
        max_bits=max_size,
        per_bit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
        rng=rng,
        exclude_keys=exclude,
        record_keys=_empty_split_records(),
        progress_name="evaluation",
        format_version=normalize_task_format_version(args),
        target_mode=normalize_bit_target_mode(args),
        alphabet=RUN_LENGTH_ALPHABET_SYMBOLS[:normalize_symbol_alphabet_size(args)],
    )
    return list(generated.get("test", []))


def split_run_length_composed_eval_slices(
    examples: Sequence[RunLengthExample],
    component_map: Dict[RunLengthKey, List[RunLengthKey]],
) -> Dict[str, List[RunLengthExample]]:
    if examples and examples[0].target_mode in {"plain_output", "symbol_run_pair"}:
        return guard_slice_partition(
            examples,
            component_map,
            key_getter=run_length_key,
            guard_fn=run_length_guard_accepts_true_components,
        )
    return {"all": list(examples)}


__all__ = [
    "prepare_run_length_composed_eval",
    "prepare_run_length_composed_train",
    "prepare_run_length_eval_examples",
    "prepare_run_length_initial_splits",
    "split_run_length_composed_eval_slices",
]

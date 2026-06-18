#!/usr/bin/env python3
"""Addition pseudolabel derivation helpers."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.addition_data import (
    AdditionExample,
    clone_with_override,
    corrupt_numeric_target,
    example_key,
    get_boundary_carry_status,
)

GeneratePredictionMap = Callable[..., Dict[Tuple[int, int, int], str]]
BuildComposedPseudoMap = Callable[..., Dict[Tuple[int, int, int], str]]
PredictionParser = Callable[[str], Optional[str]]


def _build_direct_pseudo_examples(
    candidate_examples: Sequence[AdditionExample],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    decode_max_new_tokens: int,
    prediction_parser: PredictionParser,
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[AdditionExample], int, JsonDict]:
    prediction_map = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=candidate_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=example_key,
        prediction_parser=prediction_parser,
    )
    pseudo_examples: List[AdditionExample] = []
    missing_total = 0
    for example in candidate_examples:
        override = prediction_map.get(example_key(example))
        if override is None:
            missing_total += 1
            continue
        pseudo_examples.append(clone_with_override(example, override))
    diagnostics: JsonDict = {
        "mode": "direct",
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_total,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
    }
    return pseudo_examples, missing_total, diagnostics


def derive_addition_round_targets(
    *,
    model: Any,
    tokenizer: Any,
    composed_examples: Sequence[AdditionExample],
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    target_max_size: int,
    base_examples: Sequence[AdditionExample],
    batch_size: int,
    decode_max_new_tokens: int,
    args: Any,
    rng: random.Random,
    prediction_parser: PredictionParser,
    generate_prediction_map_fn: GeneratePredictionMap,
    build_composed_pseudo_map_fn: BuildComposedPseudoMap,
) -> Tuple[List[AdditionExample], int, JsonDict]:
    candidate_examples = [example for example in composed_examples if example.digits <= target_max_size]
    if args.pseudo_label_mode == "direct":
        return _build_direct_pseudo_examples(
            candidate_examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
            prediction_parser=prediction_parser,
            generate_prediction_map_fn=generate_prediction_map_fn,
        )
    if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
        return [], 0, {
            "mode": args.pseudo_label_mode,
            "candidate_total": len(candidate_examples),
            "retained_total": 0,
            "missing_total": 0,
        }

    filter_component_carries = args.composed_strategy == "with_carry_filtered"
    carry_error_fraction = args.composition_error_percent / 100.0
    candidate_keys = {example_key(example) for example in candidate_examples}
    base_predictions = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=base_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=example_key,
        prediction_parser=prediction_parser,
    )
    base_map = {
        key: base_predictions[key]
        for key in (example_key(example) for example in base_examples)
        if key in base_predictions
    }
    component_subset = {key: component_map[key] for key in component_map if key in candidate_keys}
    pseudo_map = build_composed_pseudo_map_fn(
        base_map,
        candidate_examples,
        component_subset,
        base_predictions,
        filter_component_carries=filter_component_carries,
        carry_error_fraction=carry_error_fraction if filter_component_carries else 0.0,
        rng=rng,
    )

    candidate_boundary = 0
    candidate_no_boundary = 0
    candidate_unknown = 0
    kept_boundary = 0
    kept_no_boundary = 0
    kept_unknown = 0
    missing_boundary = 0
    missing_no_boundary = 0
    missing_unknown = 0
    corrupted_total = 0

    pseudo_examples: List[AdditionExample] = []
    missing_labels = 0
    for example in candidate_examples:
        status = get_boundary_carry_status(example, component_subset)
        if status is True:
            candidate_boundary += 1
        elif status is False:
            candidate_no_boundary += 1
        else:
            candidate_unknown += 1

        override = pseudo_map.get(example_key(example))
        if override is None:
            missing_labels += 1
            if status is True:
                missing_boundary += 1
            elif status is False:
                missing_no_boundary += 1
            else:
                missing_unknown += 1
            continue

        if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
            override = corrupt_numeric_target(override)
            corrupted_total += 1

        pseudo_examples.append(clone_with_override(example, override))
        if status is True:
            kept_boundary += 1
        elif status is False:
            kept_no_boundary += 1
        else:
            kept_unknown += 1

    diagnostics: JsonDict = {
        "mode": args.pseudo_label_mode,
        "target_max_digits": int(target_max_size),
        "candidate_total": len(candidate_examples),
        "candidate_boundary_carry": candidate_boundary,
        "candidate_no_boundary_carry": candidate_no_boundary,
        "candidate_unknown_boundary": candidate_unknown,
        "retained_total": len(pseudo_examples),
        "retained_boundary_carry": kept_boundary,
        "retained_no_boundary_carry": kept_no_boundary,
        "retained_unknown_boundary": kept_unknown,
        "missing_total": missing_labels,
        "missing_boundary_carry": missing_boundary,
        "missing_no_boundary_carry": missing_no_boundary,
        "missing_unknown_boundary": missing_unknown,
        "retained_boundary_fraction": kept_boundary / candidate_boundary if candidate_boundary > 0 else math.nan,
        "retained_no_boundary_fraction": kept_no_boundary / candidate_no_boundary if candidate_no_boundary > 0 else math.nan,
        "retained_unknown_fraction": kept_unknown / candidate_unknown if candidate_unknown > 0 else math.nan,
        "filter_component_carries": bool(filter_component_carries),
        "carry_error_fraction": carry_error_fraction if filter_component_carries else 0.0,
        "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
        "corrupted_total": corrupted_total,
    }
    return pseudo_examples, missing_labels, diagnostics

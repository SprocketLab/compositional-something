#!/usr/bin/env python3
"""Multiplication pseudolabel derivation helpers."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit_common import normalize_task_format_version
from self.tasks.bit_parsing import format_multiplication_target
from self.tasks.bit_pseudolabels import build_direct_pseudo_examples
from self.tasks.multiplication_data import (
    MultiplicationExample,
    clone_multiplication_with_override,
    multiplication_key,
)

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

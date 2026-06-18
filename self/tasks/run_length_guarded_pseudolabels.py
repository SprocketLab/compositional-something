#!/usr/bin/env python3
"""Run-length guarded plain-output and symbol-pair pseudolabel helpers."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit_common import (
    build_guarded_bit_pseudo_examples,
    normalize_bit_composition_path_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
    parse_run_length_prediction,
    parse_run_length_symbol_pair_prediction,
    run_length_guard_accepts_true_components,
)
from self.tasks.run_length_data import (
    RunLengthExample,
    build_run_length_composed_dataset,
    clone_run_length_with_override,
    run_length_key,
)

GeneratePredictionMap = Callable[..., Dict[Tuple[int, str], str]]


def derive_guarded_pair_pseudo(
    candidate_examples: Sequence[RunLengthExample],
    component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    target_max_size: int,
    base_examples: Sequence[RunLengthExample],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    decode_max_new_tokens: int,
    args: Any,
    rng: random.Random,
    target_mode: str,
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[RunLengthExample], int, JsonDict]:
    guarded_rule = normalize_guarded_compose_rule(args)
    base_predictions = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=base_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=run_length_key,
        prediction_parser=parse_run_length_prediction,
    )

    def evaluate_candidate(
        example: RunLengthExample,
        component_keys: Optional[Sequence[Tuple[int, str]]],
    ) -> Tuple[str, Optional[str]]:
        del example
        if not component_keys or len(component_keys) != 2:
            return "missing", None
        if (
            guarded_rule == "run_length_no_boundary_continue"
            and run_length_guard_accepts_true_components(component_keys) is not True
        ):
            return "rejected", None
        plain_values: List[int] = []
        pair_values: List[Tuple[str, int]] = []
        for component_key in component_keys:
            prediction = base_predictions.get(component_key)
            if prediction is None:
                return "missing", None
            if target_mode == "plain_output":
                try:
                    value = int(prediction)
                except ValueError:
                    return "missing", None
                if value < 0 or value > component_key[0]:
                    return "missing", None
                plain_values.append(value)
            else:
                parsed_pair = parse_run_length_symbol_pair_prediction(
                    prediction,
                    RunLengthExample(
                        bitstring=component_key[1],
                        bits=component_key[0],
                        max_run=0,
                        prefix_run=0,
                        suffix_run=0,
                        target_mode="symbol_run_pair",
                    ),
                )
                if parsed_pair is None:
                    return "missing", None
                symbol, value_text = parsed_pair.split("|", 1)
                value = int(value_text)
                if value < 0 or value > component_key[0] or symbol not in component_key[1]:
                    return "missing", None
                pair_values.append((symbol, value))
        if target_mode == "plain_output":
            return "accepted", str(max(plain_values))
        best_symbol, best_value = pair_values[0]
        for symbol, value in pair_values[1:]:
            if value > best_value:
                best_symbol = symbol
                best_value = value
        return "accepted", f"{best_symbol}|{best_value}"

    def refill_builder(
        bits: int,
        need: int,
        occupied_keys: set[Tuple[int, str]],
    ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]]]:
        record_components = {"train": {}, "validation": {}, "test": {}}
        refill_splits = build_run_length_composed_dataset(
            base_splits={"train": list(base_examples), "validation": [], "test": []},
            min_bits=bits,
            max_bits=bits,
            per_bit_counts={"train": need, "validation": 0, "test": 0},
            rng=rng,
            exclude_keys=occupied_keys,
            record_keys={"train": set(), "validation": set(), "test": set()},
            progress_name="guarded-refill",
            record_components=record_components,
            compose_arity=normalize_compose_arity(args),
            bit_composition_path_mode=normalize_bit_composition_path_mode(args),
        )
        return refill_splits.get("train", []), record_components.get("train", {})

    return build_guarded_bit_pseudo_examples(
        candidate_examples,
        component_map,
        target_max_size=target_max_size,
        requested_per_size=args.expand_train_per_size,
        size_getter=lambda example: example.bits,
        key_getter=run_length_key,
        clone_builder=clone_run_length_with_override,
        evaluate_candidate=evaluate_candidate,
        refill_builder=refill_builder,
        mode="compose_unfiltered_pair" if guarded_rule == "run_length_unfiltered_pair" else "compose_guarded",
    )


__all__ = ["derive_guarded_pair_pseudo"]

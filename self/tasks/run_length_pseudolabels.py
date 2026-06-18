#!/usr/bin/env python3
"""Run-length pseudolabel derivation helpers."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit_common import (
    INTEGER_PATTERN,
    RUN_LENGTH_TARGET_RUN_STATE,
    build_direct_pseudo_examples,
    build_guarded_bit_pseudo_examples,
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
    parse_run_length_symbol_pair_prediction,
    run_length_guard_accepts_true_components,
)
from self.tasks.run_length_data import (
    RunLengthExample,
    build_run_length_composed_dataset,
    clone_run_length_with_override,
    run_length_key,
)
from self.tasks.run_length_logic import format_run_length_run_state, merge_run_state

GeneratePredictionMap = Callable[..., Dict[Tuple[int, str], str]]


def _empty_pseudo_result(mode: str, candidate_total: int) -> Tuple[List[RunLengthExample], int, JsonDict]:
    return [], 0, {
        "mode": mode,
        "candidate_total": candidate_total,
        "retained_total": 0,
        "missing_total": 0,
    }


def _derive_direct_pseudo(
    candidate_examples: Sequence[RunLengthExample],
    *,
    model: Any,
    tokenizer: Any,
    batch_size: int,
    decode_max_new_tokens: int,
) -> Tuple[List[RunLengthExample], int, JsonDict]:
    return build_direct_pseudo_examples(
        candidate_examples,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        decode_max_new_tokens=decode_max_new_tokens,
        key_getter=run_length_key,
        prediction_parser=parse_run_length_prediction,
        clone_builder=clone_run_length_with_override,
        mode="direct",
    )


def _derive_guarded_pair_pseudo(
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


def _derive_run_state_pseudo(
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
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[RunLengthExample], int, JsonDict]:
    if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
        return _empty_pseudo_result(args.pseudo_label_mode, len(candidate_examples))
    base_predictions = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=base_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=run_length_key,
        prediction_parser=parse_run_length_prediction,
    )
    pseudo_examples: List[RunLengthExample] = []
    missing_labels = 0
    corrupted_examples = 0
    for example in candidate_examples:
        component_keys = component_map.get(run_length_key(example))
        if not component_keys:
            missing_labels += 1
            continue
        components: List[Tuple[int, int, str, int, str, int]] = []
        missing = False
        for component_key in component_keys:
            prediction = base_predictions.get(component_key)
            if prediction is None:
                missing = True
                break
            parsed = parse_run_length_run_state_prediction(
                prediction,
                RunLengthExample(
                    bitstring=component_key[1],
                    bits=component_key[0],
                    max_run=0,
                    prefix_run=0,
                    suffix_run=0,
                    target_mode=RUN_LENGTH_TARGET_RUN_STATE,
                ),
            )
            if parsed is None:
                missing = True
                break
            max_text, prefix_symbol, prefix_text, suffix_symbol, suffix_text = parsed.split("|")
            bits = component_key[0]
            components.append(
                (
                    bits,
                    int(max_text),
                    prefix_symbol,
                    int(prefix_text),
                    suffix_symbol,
                    int(suffix_text),
                )
            )
        if missing or not components:
            missing_labels += 1
            continue
        if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
            idx = rng.randrange(len(components))
            bits, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = components[idx]
            if max_run < bits:
                max_run += 1
            elif max_run > 0:
                max_run -= 1
            elif bits > 0:
                max_run = 1
            components[idx] = (bits, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run)
            corrupted_examples += 1
        merged = components[0]
        for nxt in components[1:]:
            merged = merge_run_state(merged, nxt)
        _, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = merged
        override = format_run_length_run_state((max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run))
        pseudo_examples.append(clone_run_length_with_override(example, override))
    diagnostics: JsonDict = {
        "mode": args.pseudo_label_mode,
        "target_max_bits": int(target_max_size),
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_labels,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
        "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
        "corrupted_examples": corrupted_examples,
    }
    return pseudo_examples, missing_labels, diagnostics


def _merge_default_stats(
    left: Tuple[int, int, int, int],
    right: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    left_bits, left_max, left_prefix, left_suffix = left
    right_bits, right_max, right_prefix, right_suffix = right
    bits = left_bits + right_bits
    prefix = left_bits + right_prefix if left_prefix == left_bits else left_prefix
    suffix = right_bits + left_suffix if right_suffix == right_bits else right_suffix
    max_run = max(left_max, right_max, left_suffix + right_prefix)
    return bits, max_run, prefix, suffix


def _derive_default_tuple_pseudo(
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
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[RunLengthExample], int, JsonDict]:
    if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
        return _empty_pseudo_result(args.pseudo_label_mode, len(candidate_examples))

    base_predictions = generate_prediction_map_fn(
        model=model,
        tokenizer=tokenizer,
        examples=base_examples,
        batch_size=batch_size,
        max_new_tokens=decode_max_new_tokens,
        key_getter=run_length_key,
        prediction_parser=parse_run_length_prediction,
    )
    pseudo_examples: List[RunLengthExample] = []
    missing_labels = 0
    corrupted_examples = 0

    for example in candidate_examples:
        component_keys = component_map.get(run_length_key(example))
        if not component_keys:
            missing_labels += 1
            continue
        components: List[Tuple[int, int, int, int]] = []
        missing = False
        for component_key in component_keys:
            prediction = base_predictions.get(component_key)
            if prediction is None:
                missing = True
                break
            parsed = INTEGER_PATTERN.findall(prediction)
            if len(parsed) < 3:
                missing = True
                break
            max_run = int(parsed[0])
            prefix = int(parsed[1])
            suffix = int(parsed[2])
            bits = component_key[0]
            if max_run < 0 or prefix < 0 or suffix < 0 or max_run > bits or prefix > bits or suffix > bits:
                missing = True
                break
            components.append((bits, max_run, prefix, suffix))
        if missing or not components:
            missing_labels += 1
            continue
        if args.pseudo_label_mode == "compose_corrupt" and rng.random() < args.corruption_rate:
            idx = rng.randrange(len(components))
            bits, max_run, prefix, suffix = components[idx]
            if max_run < bits:
                max_run += 1
            elif max_run > 0:
                max_run -= 1
            elif bits > 0:
                max_run = 1
            components[idx] = (bits, max_run, prefix, suffix)
            corrupted_examples += 1
        merged = components[0]
        for nxt in components[1:]:
            merged = _merge_default_stats(merged, nxt)
        _, max_run, prefix, suffix = merged
        override = f"{max_run}|{prefix}|{suffix}"
        pseudo_examples.append(clone_run_length_with_override(example, override))
    diagnostics: JsonDict = {
        "mode": args.pseudo_label_mode,
        "target_max_bits": int(target_max_size),
        "candidate_total": len(candidate_examples),
        "retained_total": len(pseudo_examples),
        "missing_total": missing_labels,
        "retained_fraction": len(pseudo_examples) / len(candidate_examples) if candidate_examples else math.nan,
        "corruption_rate": args.corruption_rate if args.pseudo_label_mode == "compose_corrupt" else 0.0,
        "corrupted_examples": corrupted_examples,
    }
    return pseudo_examples, missing_labels, diagnostics


def derive_run_length_round_targets(
    *,
    model: Any,
    tokenizer: Any,
    composed_examples: Sequence[RunLengthExample],
    component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    target_max_size: int,
    base_examples: Sequence[RunLengthExample],
    batch_size: int,
    decode_max_new_tokens: int,
    args: Any,
    rng: random.Random,
    generate_prediction_map_fn: GeneratePredictionMap,
) -> Tuple[List[RunLengthExample], int, JsonDict]:
    candidate_examples = [example for example in composed_examples if example.bits <= target_max_size]
    target_mode = normalize_bit_target_mode(args)
    if args.pseudo_label_mode == "direct":
        return _derive_direct_pseudo(
            candidate_examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
        )
    if target_mode in {"plain_output", "symbol_run_pair"}:
        return _derive_guarded_pair_pseudo(
            candidate_examples,
            component_map,
            target_max_size,
            base_examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
            args=args,
            rng=rng,
            target_mode=target_mode,
            generate_prediction_map_fn=generate_prediction_map_fn,
        )
    if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
        return _derive_run_state_pseudo(
            candidate_examples,
            component_map,
            target_max_size,
            base_examples,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            decode_max_new_tokens=decode_max_new_tokens,
            args=args,
            rng=rng,
            generate_prediction_map_fn=generate_prediction_map_fn,
        )
    return _derive_default_tuple_pseudo(
        candidate_examples,
        component_map,
        target_max_size,
        base_examples,
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        decode_max_new_tokens=decode_max_new_tokens,
        args=args,
        rng=rng,
        generate_prediction_map_fn=generate_prediction_map_fn,
    )

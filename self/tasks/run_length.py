#!/usr/bin/env python3
"""Run-length task adapter, data generation, and pseudolabel helpers."""

from __future__ import annotations

# --- from run_length_logic.py ---
from typing import Optional, Tuple

from self.tasks.bit import RUN_LENGTH_TARGET_RUN_STATE


def compute_run_stats(bitstring: str) -> Tuple[int, int, int]:
    if not bitstring:
        return 0, 0, 0
    max_run = 1
    current = 1
    for previous, current_symbol in zip(bitstring, bitstring[1:]):
        if current_symbol == previous:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    prefix_symbol = bitstring[0]
    prefix = 0
    for ch in bitstring:
        if ch == prefix_symbol:
            prefix += 1
        else:
            break
    suffix_symbol = bitstring[-1]
    suffix = 0
    for ch in reversed(bitstring):
        if ch == suffix_symbol:
            suffix += 1
        else:
            break
    return max_run, prefix, suffix


def compute_run_state(bitstring: str) -> Tuple[int, str, int, str, int]:
    max_run, prefix, suffix = compute_run_stats(bitstring)
    if not bitstring:
        return max_run, "", prefix, "", suffix
    return max_run, bitstring[0], prefix, bitstring[-1], suffix


def format_run_length_run_state(state: Tuple[int, str, int, str, int]) -> str:
    max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = state
    return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"


def merge_run_state(
    left: Tuple[int, int, str, int, str, int],
    right: Tuple[int, int, str, int, str, int],
) -> Tuple[int, int, str, int, str, int]:
    left_bits, left_max, left_prefix_symbol, left_prefix_run, left_suffix_symbol, left_suffix_run = left
    right_bits, right_max, right_prefix_symbol, right_prefix_run, right_suffix_symbol, right_suffix_run = right
    bits = left_bits + right_bits
    boundary = left_suffix_run + right_prefix_run if left_suffix_symbol == right_prefix_symbol else 0
    max_run = max(left_max, right_max, boundary)
    prefix_symbol = left_prefix_symbol
    prefix_run = left_prefix_run
    if left_prefix_run == left_bits and left_prefix_symbol == right_prefix_symbol:
        prefix_run = left_bits + right_prefix_run
    suffix_symbol = right_suffix_symbol
    suffix_run = right_suffix_run
    if right_suffix_run == right_bits and left_suffix_symbol == right_suffix_symbol:
        suffix_run = right_bits + left_suffix_run
    return bits, max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run


def leftmost_max_run_pair(bitstring: str) -> Tuple[str, int]:
    if not bitstring:
        return "", 0
    best_symbol = bitstring[0]
    best_length = 1
    current_symbol = bitstring[0]
    current_length = 1
    for ch in bitstring[1:]:
        if ch == current_symbol:
            current_length += 1
        else:
            current_symbol = ch
            current_length = 1
        if current_length > best_length:
            best_symbol = current_symbol
            best_length = current_length
    return best_symbol, best_length


def format_run_length_target(
    max_run: int,
    prefix: int,
    suffix: int,
    format_version: str,
    target_mode: str = "default",
    *,
    bitstring: Optional[str] = None,
) -> str:
    del format_version
    if target_mode == "plain_output":
        return str(max_run)
    if target_mode == "symbol_run_pair":
        if bitstring is None:
            raise ValueError("symbol_run_pair run-length targets require bitstring context.")
        symbol, run_length = leftmost_max_run_pair(bitstring)
        return f"{symbol}|{run_length}"
    if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
        if bitstring is None:
            raise ValueError("run_state run-length targets require bitstring context.")
        state = compute_run_state(bitstring)
        return format_run_length_run_state(state)
    return f"{max_run}|{prefix}|{suffix}"


# --- from run_length_data.py ---
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from self.tasks.bit import sample_unique_bitstrings
from self.tasks.bit import BIT_COMPOSITION_PATH_RANDOM, choose_component_sizes


@dataclass(frozen=True)
class RunLengthExample:
    bitstring: str
    bits: int
    max_run: int
    prefix_run: int
    suffix_run: int
    format_version: str = "legacy"
    target_mode: str = "default"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"runlen({self.bitstring})="
        return f"Q: runlen({self.bitstring}) = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_run_length_target(
            self.max_run,
            self.prefix_run,
            self.suffix_run,
            self.format_version,
            self.target_mode,
            bitstring=self.bitstring,
        )

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def run_length_key(example: RunLengthExample) -> Tuple[int, str]:
    return example.bits, example.bitstring


def encode_run_length_key(key: Tuple[int, str]) -> str:
    return f"{key[0]}|{key[1]}"


def decode_run_length_key(value: str) -> Tuple[int, str]:
    bits, bitstring = value.split("|", 1)
    return int(bits), bitstring


def generate_run_length_example(
    num_bits: int,
    rng: random.Random,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
) -> RunLengthExample:
    bitstring = "".join(rng.choice(alphabet) for _ in range(num_bits))
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=num_bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=format_version,
        target_mode=target_mode,
    )


def merge_run_length(left: RunLengthExample, right: RunLengthExample) -> RunLengthExample:
    bitstring = left.bitstring + right.bitstring
    bits = left.bits + right.bits
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=bits,
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=left.format_version,
        target_mode=left.target_mode,
    )


def compose_run_length_examples(*examples: RunLengthExample) -> RunLengthExample:
    if len(examples) < 2:
        raise ValueError("Need at least two run-length examples to compose a longer instance.")
    merged = examples[0]
    for nxt in examples[1:]:
        merged = merge_run_length(merged, nxt)
    return merged


def compose_run_length_to_length(
    buckets: Dict[int, List[RunLengthExample]],
    target_bits: int,
    rng: random.Random,
    *,
    compose_arity: str = "at_least2",
    bit_composition_path_mode: str = BIT_COMPOSITION_PATH_RANDOM,
) -> Tuple[RunLengthExample, List[RunLengthExample]]:
    sizes = list(buckets.keys())
    chosen_sizes = choose_component_sizes(
        target_bits,
        sizes,
        rng,
        compose_arity=compose_arity,
        bit_composition_path_mode=bit_composition_path_mode,
    )
    if not chosen_sizes:
        raise ValueError(
            f"Unable to compose a run-length example of {target_bits} bits from base bucket sizes {sorted(sizes)}."
        )
    chosen = [rng.choice(buckets[size]) for size in chosen_sizes]
    return compose_run_length_examples(*chosen), chosen


def build_run_length_length_bucket_dataset(
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    format_version: str = "legacy",
    target_mode: str = "default",
    alphabet: str = "01",
    split_order: Sequence[str] = ("train", "validation", "test"),
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for bits in range(min_bits, max_bits + 1):
        per_bit_per_split = {split: per_bit_counts.get(split, 0) for split in split_order}
        total_requested = sum(per_bit_per_split.values())
        if total_requested == 0:
            continue
        available_unique = max(0, (len(alphabet) ** bits) - used_counts.get(bits, 0))
        if available_unique < total_requested:
            print(
                f"[WARN] Requested {total_requested} examples for bits={bits} exceeds available unique strings ({available_unique}); capping counts.",
                flush=True,
            )
            remaining = available_unique
            for split in split_order:
                requested = per_bit_per_split[split]
                if requested > remaining:
                    per_bit_per_split[split] = remaining
                    remaining = 0
                else:
                    remaining -= requested
            total_requested = sum(per_bit_per_split.values())
            if total_requested == 0:
                continue

        generated: List[Tuple[RunLengthExample, Tuple[int, str], bool]] = []
        for bitstring in sample_unique_bitstrings(
            bits,
            total_requested,
            rng,
            occupied,
            alphabet=alphabet,
            max_attempts=max_attempts,
        ):
            max_run, prefix, suffix = compute_run_stats(bitstring)
            example = RunLengthExample(
                bitstring=bitstring,
                bits=bits,
                max_run=max_run,
                prefix_run=prefix,
                suffix_run=suffix,
                format_version=format_version,
                target_mode=target_mode,
            )
            key = run_length_key(example)
            occupied.add(key)
            used_counts[bits] += 1
            generated.append((example, key, False))

        index = 0
        for split in split_order:
            count = per_bit_per_split.get(split, 0)
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
                    f"[INFO] Generated {len(chunk)}/{count} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )

    for split in splits:
        rng.shuffle(splits[split])
    return splits


def bucket_run_length_by_bits(examples: Sequence[RunLengthExample]) -> Dict[int, List[RunLengthExample]]:
    buckets: Dict[int, List[RunLengthExample]] = defaultdict(list)
    for example in examples:
        buckets[example.bits].append(example)
    return buckets


def build_run_length_composed_dataset(
    base_splits: Dict[str, List[RunLengthExample]],
    min_bits: int,
    max_bits: int,
    per_bit_counts: Dict[str, int],
    rng: random.Random,
    *,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    record_keys: Optional[Dict[str, set[Tuple[int, str]]]] = None,
    progress_name: Optional[str] = None,
    max_attempts: int = 10_000,
    record_components: Optional[Dict[str, Dict[Tuple[int, str], List[Tuple[int, str]]]]] = None,
    compose_arity: str = "at_least2",
    bit_composition_path_mode: str = BIT_COMPOSITION_PATH_RANDOM,
) -> Dict[str, List[RunLengthExample]]:
    splits = {key: [] for key in ("train", "validation", "test")}
    occupied = set(exclude_keys) if exclude_keys else set()
    used_counts: Dict[int, int] = defaultdict(int)
    for key in occupied:
        used_counts[key[0]] += 1

    for split in ("train", "validation", "test"):
        requested_per_bit = per_bit_counts.get(split, 0)
        if requested_per_bit <= 0:
            continue
        buckets = bucket_run_length_by_bits(base_splits.get(split, []))
        component_map = None
        if record_components is not None:
            component_map = record_components.setdefault(split, {})
        for bits in range(min_bits, max_bits + 1):
            available_unique = max(0, (2**bits) - used_counts.get(bits, 0))
            effective_target = min(requested_per_bit, available_unique)
            if effective_target < requested_per_bit:
                print(
                    f"[WARN] Requested {requested_per_bit} composed examples for bits={bits} split='{split}' exceeds available unique strings ({available_unique}); capping.",
                    flush=True,
                )
            if effective_target <= 0:
                continue
            generated: List[Tuple[RunLengthExample, Tuple[int, str], bool, List[Tuple[int, str]]]] = []
            attempts = 0
            duplicates_allowed = False
            while len(generated) < effective_target:
                attempts += 1
                example, components = compose_run_length_to_length(
                    buckets,
                    bits,
                    rng,
                    compose_arity=compose_arity,
                    bit_composition_path_mode=bit_composition_path_mode,
                )
                component_keys = [run_length_key(component) for component in components]
                key = run_length_key(example)
                if key in occupied:
                    if attempts >= max_attempts:
                        if not duplicates_allowed:
                            print(
                                f"[WARN] Exhausted unique composed sampling for bits={bits} split='{split}' (progress={progress_name}); allowing duplicates.",
                                flush=True,
                            )
                            duplicates_allowed = True
                        generated.append((example, key, True, component_keys))
                        attempts = 0
                    continue
                occupied.add(key)
                used_counts[bits] += 1
                generated.append((example, key, False, component_keys))
                attempts = 0
            splits[split].extend(example for example, _, _, _ in generated)
            if record_keys and split in record_keys:
                for _, key, is_duplicate, _ in generated:
                    if not is_duplicate:
                        record_keys[split].add(key)
            if component_map is not None:
                for _, key, _, component_keys in generated:
                    component_map[key] = component_keys
            if progress_name:
                print(
                    f"[INFO] Generated {len(generated)}/{effective_target} {progress_name} examples for split='{split}' bits={bits}",
                    flush=True,
                )
        rng.shuffle(splits[split])
    return splits


def clone_run_length_with_override(example: RunLengthExample, override: Optional[str]) -> RunLengthExample:
    if override is None:
        return example
    return RunLengthExample(
        bitstring=example.bitstring,
        bits=example.bits,
        max_run=example.max_run,
        prefix_run=example.prefix_run,
        suffix_run=example.suffix_run,
        format_version=example.format_version,
        target_mode=example.target_mode,
        target_override=override,
    )


# --- from run_length.py ---
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit import (
    normalize_bit_composition_path_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
)
from self.tasks.bit import (
    parse_run_length_prediction,
    parse_run_length_symbol_pair_prediction,
)
from self.tasks.bit import (
    build_guarded_bit_pseudo_examples,
    run_length_guard_accepts_true_components,
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


# --- from run_length.py ---
import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple

from self.core.task_protocols import JsonDict
from self.tasks.bit import (
    normalize_bit_target_mode,
)
from self.tasks.bit import (
    INTEGER_PATTERN,
    RUN_LENGTH_TARGET_RUN_STATE,
    parse_run_length_prediction,
    parse_run_length_run_state_prediction,
)
from self.tasks.bit import build_direct_pseudo_examples

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
        return derive_guarded_pair_pseudo(
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


# --- from run_length_splits.py ---
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.tasks.bit import (
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_symbol_alphabet_size,
    normalize_task_format_version,
)
from self.tasks.bit import bit_composed_target_sizes_from_examples
from self.tasks.bit import RUN_LENGTH_ALPHABET_SYMBOLS
from self.tasks.bit import guard_slice_partition, run_length_guard_accepts_true_components

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


# --- from run_length.py ---
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map as _default_generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask
from self.tasks.bit import (
    BIT_COMPOSE_ARITIES,
    BIT_GUARDED_COMPOSE_RULES,
    BIT_TARGET_MODES,
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
    normalize_symbol_alphabet_size,
    normalize_task_format_version,
)
from self.tasks.bit import (
    RUN_LENGTH_ALPHABET_SYMBOLS,
    RUN_LENGTH_FORMATS,
    RUN_LENGTH_TARGET_RUN_STATE,
    parse_run_length_prediction,
)
from self.tasks.bit import (
    BIT_COMPOSITION_PATH_FIXED_BINARY,
    BIT_COMPOSITION_PATH_MODES,
    BIT_COMPOSITION_PATH_RANDOM,
)

SplitName = str


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.tasks")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def generate_prediction_map(**kwargs: Any) -> Any:
    return _compat_symbol("generate_prediction_map", _default_generate_prediction_map)(**kwargs)


class RunLengthTask(SelfImprovementTask):
    name = "run_length"
    size_label = "bits"
    size_alias_singular = "bit"
    size_alias_plural = "bits"

    def validate_args(self, args: Any) -> None:
        corruption_rate = float(getattr(args, "corruption_rate", 0.0))
        pseudo_label_mode = str(getattr(args, "pseudo_label_mode", "none"))
        if corruption_rate < 0.0 or corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        symbol_alphabet_size = normalize_symbol_alphabet_size(args)
        if symbol_alphabet_size < 2:
            raise ValueError("run_length symbol_alphabet_size must be at least 2.")
        if symbol_alphabet_size > len(RUN_LENGTH_ALPHABET_SYMBOLS):
            raise ValueError(
                f"run_length symbol_alphabet_size={symbol_alphabet_size} exceeds supported alphabet size "
                f"{len(RUN_LENGTH_ALPHABET_SYMBOLS)}."
            )
        format_version = normalize_task_format_version(args)
        if format_version not in RUN_LENGTH_FORMATS:
            raise ValueError(f"Unsupported run_length format_version={format_version!r}.")
        target_mode = normalize_bit_target_mode(args)
        if target_mode not in BIT_TARGET_MODES:
            raise ValueError(f"Unsupported run_length target_mode={target_mode!r}.")
        compose_arity = normalize_compose_arity(args)
        if compose_arity not in BIT_COMPOSE_ARITIES:
            raise ValueError(f"Unsupported run_length compose_arity={compose_arity!r}.")
        bit_composition_path_mode = normalize_bit_composition_path_mode(args)
        if bit_composition_path_mode not in BIT_COMPOSITION_PATH_MODES:
            raise ValueError(f"Unsupported run_length bit_composition_path_mode={bit_composition_path_mode!r}.")
        guarded_rule = normalize_guarded_compose_rule(args)
        if guarded_rule not in BIT_GUARDED_COMPOSE_RULES:
            raise ValueError(f"Unsupported run_length guarded_compose_rule={guarded_rule!r}.")
        if target_mode in {"plain_output", "symbol_run_pair"} and pseudo_label_mode in {"compose", "compose_corrupt"}:
            if guarded_rule not in {"run_length_no_boundary_continue", "run_length_unfiltered_pair"}:
                raise ValueError(
                    "run_length guarded output compose mode requires --guarded-compose-rule "
                    "run_length_no_boundary_continue or run_length_unfiltered_pair."
                )
            if compose_arity != "exact2" and bit_composition_path_mode != BIT_COMPOSITION_PATH_FIXED_BINARY:
                raise ValueError("run_length pair-output compose mode requires --compose-arity exact2.")
            if pseudo_label_mode == "compose_corrupt":
                raise ValueError("run_length guarded output diagnostics do not support compose_corrupt.")

    def serialize_example(self, example: RunLengthExample) -> JsonDict:
        return {
            "bitstring": example.bitstring,
            "bits": example.bits,
            "max_run": example.max_run,
            "prefix_run": example.prefix_run,
            "suffix_run": example.suffix_run,
            "format_version": example.format_version,
            "target_mode": example.target_mode,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> RunLengthExample:
        return RunLengthExample(
            bitstring=str(payload["bitstring"]),
            bits=int(payload["bits"]),
            max_run=int(payload["max_run"]),
            prefix_run=int(payload["prefix_run"]),
            suffix_run=int(payload["suffix_run"]),
            format_version=str(payload.get("format_version", "legacy")),
            target_mode=str(payload.get("target_mode", "default")),
            target_override=payload.get("target_override"),
        )

    def save_component_map(self, path: Path, component_map: Dict[Tuple[int, str], List[Tuple[int, str]]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            encode_run_length_key(key): [encode_run_length_key(child) for child in children]
            for key, children in component_map.items()
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_run_length_key(key): [decode_run_length_key(child) for child in children] for key, children in raw.items()}

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[RunLengthExample]], Dict[SplitName, set[Tuple[int, str]]]]:
        return prepare_run_length_initial_splits(rng, args)

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[RunLengthExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        return prepare_run_length_composed_train(
            rng,
            args,
            base_splits,
            base_records,
            min_size,
            max_size,
            additional_exclude,
        )

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[RunLengthExample]],
        base_records: Dict[SplitName, set[Tuple[int, str]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, str]]] = None,
    ) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]], set[Tuple[int, str]]]:
        return prepare_run_length_composed_eval(
            rng,
            args,
            base_splits,
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
        exclude: set[Tuple[int, str]],
    ) -> List[RunLengthExample]:
        return prepare_run_length_eval_examples(rng, args, min_size, max_size, exclude)

    def split_composed_eval_slices(
        self,
        examples: Sequence[RunLengthExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    ) -> Dict[str, List[RunLengthExample]]:
        return split_run_length_composed_eval_slices(examples, component_map)

    def keys_for_examples(self, examples: Sequence[RunLengthExample]) -> set[Tuple[int, str]]:
        return {run_length_key(example) for example in examples}

    def rebuild_records(self, splits: Dict[SplitName, List[RunLengthExample]]) -> Dict[SplitName, set[Tuple[int, str]]]:
        return {split: {run_length_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: RunLengthExample) -> Tuple[int, str]:
        return run_length_key(example)

    def clone_with_override(self, example: RunLengthExample, override: Optional[str]) -> RunLengthExample:
        return clone_run_length_with_override(example, override)

    def size_of(self, example: RunLengthExample) -> int:
        return example.bits

    def prediction_parser(self, text: str, example: Optional[RunLengthExample] = None) -> Optional[str]:
        return parse_run_length_prediction(text, example)

    def token_initializers(self, args: Any) -> Dict[str, str]:
        return {}

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[RunLengthExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
        target_max_size: int,
        base_examples: Sequence[RunLengthExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[RunLengthExample], int, JsonDict]:
        return derive_run_length_round_targets(
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
            generate_prediction_map_fn=generate_prediction_map,
        )

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "bit_composition_path_mode": normalize_bit_composition_path_mode(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
            "symbol_alphabet_size": normalize_symbol_alphabet_size(args),
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "initial_min_bits": args.initial_min_size,
            "initial_max_bits": args.initial_max_size,
            "expand_num_bits": args.expand_num_size,
            "expand_train_per_bit": args.expand_train_per_size,
            "eval_per_bit": args.eval_per_size,
            "composed_eval_per_bit": args.composed_eval_per_size,
            "composed_max_bits": final_max_size,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
            "format_version": normalize_task_format_version(args),
            "target_mode": normalize_bit_target_mode(args),
            "compose_arity": normalize_compose_arity(args),
            "bit_composition_path_mode": normalize_bit_composition_path_mode(args),
            "guarded_compose_rule": normalize_guarded_compose_rule(args),
            "symbol_alphabet_size": normalize_symbol_alphabet_size(args),
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_format = str(task_config.get("format_version", metadata.get("format_version", "legacy")))
        if stored_format != normalize_task_format_version(args):
            raise ValueError("Stored run-length dataset uses a different format_version.")
        stored_target_mode = str(task_config.get("target_mode", metadata.get("target_mode", "default")))
        if stored_target_mode != normalize_bit_target_mode(args):
            raise ValueError("Stored run-length dataset uses a different target_mode.")
        stored_compose_arity = str(task_config.get("compose_arity", metadata.get("compose_arity", "at_least2")))
        if stored_compose_arity != normalize_compose_arity(args):
            raise ValueError("Stored run-length dataset uses a different compose_arity.")
        stored_path_mode = str(
            task_config.get(
                "bit_composition_path_mode",
                metadata.get("bit_composition_path_mode", BIT_COMPOSITION_PATH_RANDOM),
            )
        )
        if stored_path_mode != normalize_bit_composition_path_mode(args):
            raise ValueError("Stored run-length dataset uses a different bit_composition_path_mode.")
        stored_guarded_rule = str(
            task_config.get("guarded_compose_rule", metadata.get("guarded_compose_rule", "none"))
        )
        if stored_guarded_rule != normalize_guarded_compose_rule(args):
            raise ValueError("Stored run-length dataset uses a different guarded_compose_rule.")
        stored_symbol_alphabet_size = int(
            task_config.get("symbol_alphabet_size", metadata.get("symbol_alphabet_size", 2))
        )
        if stored_symbol_alphabet_size != normalize_symbol_alphabet_size(args):
            raise ValueError("Stored run-length dataset uses a different symbol_alphabet_size.")

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        return {
            "max_bits": summary.max_size,
            "per_bit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_bits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
        }

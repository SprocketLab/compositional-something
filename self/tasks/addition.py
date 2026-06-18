#!/usr/bin/env python3
"""Addition task adapter, data generation, and pseudolabel helpers."""

from __future__ import annotations

# --- from addition_data.py ---
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


# --- from addition_pseudolabels.py ---
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from self.core.task_protocols import JsonDict

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


# --- from addition.py ---
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import extract_numeric_answer, generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask


SplitName = str
_DEFAULT_GENERATE_PREDICTION_MAP = generate_prediction_map
_DEFAULT_BUILD_COMPOSED_PSEUDO_MAP = build_composed_pseudo_map


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.tasks")
    if facade is None:
        return fallback
    return getattr(facade, name, fallback)


def _generate_prediction_map(**kwargs: Any) -> Any:
    return _compat_symbol("generate_prediction_map", _DEFAULT_GENERATE_PREDICTION_MAP)(**kwargs)


def _build_composed_pseudo_map(*args: Any, **kwargs: Any) -> Any:
    return _compat_symbol("build_composed_pseudo_map", _DEFAULT_BUILD_COMPOSED_PSEUDO_MAP)(*args, **kwargs)


class AdditionTask(SelfImprovementTask):
    name = "addition"
    size_label = "digits"
    size_alias_singular = "digit"
    size_alias_plural = "digits"

    def validate_args(self, args: Any) -> None:
        if args.composition_error_percent < 0.0 or args.composition_error_percent > 100.0:
            raise ValueError("composition_error_percent must be between 0 and 100.")
        if args.corruption_rate < 0.0 or args.corruption_rate > 1.0:
            raise ValueError("corruption_rate must be between 0 and 1.")
        if getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS) not in ADDITION_WIDTH_MODES:
            raise ValueError(f"Unsupported addition_width_mode={args.addition_width_mode!r}.")
        if getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL) not in ADDITION_SAMPLING_MODES:
            raise ValueError(f"Unsupported addition_sampling_mode={args.addition_sampling_mode!r}.")
        if (
            getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL) != ADDITION_SAMPLING_NATURAL
            and getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS) != ADDITION_WIDTH_FIXED_MIXED_PROMPT
        ):
            raise ValueError("Balanced addition sampling requires --addition-width-mode fixed_width_mixed_prompt.")
        if getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM) not in COMPOSITION_PATH_MODES:
            raise ValueError(
                f"Unsupported addition_composition_path_mode={args.addition_composition_path_mode!r}."
            )

    def serialize_example(self, example: AdditionExample) -> JsonDict:
        return {
            "a": example.a,
            "b": example.b,
            "result": example.result,
            "digits": example.digits,
            "operand_width": example.block_width,
            "has_carry": example.has_carry,
            "target_override": example.target_override,
        }

    def deserialize_example(self, payload: JsonDict) -> AdditionExample:
        return AdditionExample(
            a=int(payload["a"]),
            b=int(payload["b"]),
            result=int(payload["result"]),
            digits=int(payload["digits"]),
            has_carry=bool(payload["has_carry"]),
            target_override=payload.get("target_override"),
            operand_width=int(payload.get("operand_width", payload["digits"])),
        )

    def save_component_map(
        self,
        path: Path,
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {encode_key(key): [encode_key(child) for child in children] for key, children in component_map.items()}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def load_component_map(self, path: Path) -> Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {decode_key(key): [decode_key(child) for child in children] for key, children in raw.items()}

    def _allow_carry_for_composed(self, args: Any) -> bool:
        return args.composed_strategy in ("with_carry", "with_carry_filtered")

    def _boundary_carry_policy_for_composed(self, args: Any) -> str:
        if args.composed_strategy == "with_carry_filtered":
            return "no_boundary_carry"
        return "any"

    def prepare_initial_splits(
        self,
        rng: random.Random,
        args: Any,
    ) -> Tuple[Dict[SplitName, List[AdditionExample]], Dict[SplitName, set[Tuple[int, int, int]]]]:
        return prepare_addition_initial_splits(
            rng,
            args.initial_min_size,
            args.initial_max_size,
            args.initial_train_per_size,
            args.initial_eval_per_size,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            addition_sampling_mode=getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL),
        )

    def prepare_composed_train(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[AdditionExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
        return prepare_addition_composed_train(
            rng,
            base_splits,
            base_records,
            min_size,
            max_size,
            args.expand_train_per_size,
            allow_carry=self._allow_carry_for_composed(args),
            boundary_carry_policy=self._boundary_carry_policy_for_composed(args),
            additional_exclude=additional_exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            composition_path_mode=getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM),
        )

    def prepare_composed_eval(
        self,
        rng: random.Random,
        args: Any,
        base_splits: Dict[SplitName, List[AdditionExample]],
        base_records: Dict[SplitName, set[Tuple[int, int, int]]],
        min_size: int,
        max_size: int,
        additional_exclude: Optional[set[Tuple[int, int, int]]] = None,
    ) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]], set[Tuple[int, int, int]]]:
        return prepare_addition_composed_eval(
            rng,
            base_splits,
            base_records,
            min_size,
            max_size,
            args.composed_eval_per_size,
            additional_exclude=additional_exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            composition_path_mode=getattr(args, "addition_composition_path_mode", COMPOSITION_PATH_RANDOM),
        )

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, int, int]],
    ) -> List[AdditionExample]:
        return prepare_addition_eval_examples(
            rng,
            min_size,
            max_size,
            args.eval_per_size,
            exclude,
            addition_width_mode=getattr(args, "addition_width_mode", ADDITION_WIDTH_EXACT_DIGITS),
            addition_sampling_mode=getattr(args, "addition_sampling_mode", ADDITION_SAMPLING_NATURAL),
        )

    def split_composed_eval_slices(
        self,
        examples: Sequence[AdditionExample],
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
    ) -> Dict[str, List[AdditionExample]]:
        return split_addition_examples_by_boundary_status(examples, component_map)

    def keys_for_examples(self, examples: Sequence[AdditionExample]) -> set[Tuple[int, int, int]]:
        return {example_key(example) for example in examples}

    def rebuild_records(
        self,
        splits: Dict[SplitName, List[AdditionExample]],
    ) -> Dict[SplitName, set[Tuple[int, int, int]]]:
        return {split: {example_key(example) for example in splits.get(split, [])} for split in ("train", "validation", "test")}

    def key_for_example(self, example: AdditionExample) -> Tuple[int, int, int]:
        return example_key(example)

    def clone_with_override(self, example: AdditionExample, override: Optional[str]) -> AdditionExample:
        return clone_with_override(example, override)

    def size_of(self, example: AdditionExample) -> int:
        return example.digits

    def prediction_parser(self, text: str) -> Optional[str]:
        return extract_numeric_answer(text)

    def derive_round_targets(
        self,
        model: Any,
        tokenizer: Any,
        composed_examples: Sequence[AdditionExample],
        component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]],
        target_max_size: int,
        base_examples: Sequence[AdditionExample],
        *,
        batch_size: int,
        decode_max_new_tokens: int,
        args: Any,
        rng: random.Random,
    ) -> Tuple[List[AdditionExample], int, JsonDict]:
        return derive_addition_round_targets(
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
            generate_prediction_map_fn=_generate_prediction_map,
            build_composed_pseudo_map_fn=_build_composed_pseudo_map,
        )

    def build_task_metadata(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "composed_strategy": args.composed_strategy,
            "filter_component_carries": args.composed_strategy == "with_carry_filtered",
            "composed_boundary_carry_policy": self._boundary_carry_policy_for_composed(args),
            "composition_error_percent": args.composition_error_percent,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
        }

    def metadata_aliases(self, args: Any, final_max_size: int) -> JsonDict:
        return {
            "initial_min_digits": args.initial_min_size,
            "initial_max_digits": args.initial_max_size,
            "expand_num_digits": args.expand_num_size,
            "expand_train_per_digit": args.expand_train_per_size,
            "eval_per_digit": args.eval_per_size,
            "composed_eval_per_digit": args.composed_eval_per_size,
            "composed_max_digits": final_max_size,
            "composed_strategy": args.composed_strategy,
            "composed_without_carry": args.composed_strategy == "without_carry",
            "filter_component_carries": args.composed_strategy == "with_carry_filtered",
            "composed_boundary_carry_policy": self._boundary_carry_policy_for_composed(args),
            "composition_error_percent": args.composition_error_percent,
            "pseudo_label_mode": args.pseudo_label_mode,
            "corruption_rate": args.corruption_rate,
        }

    def validate_loaded_metadata(
        self,
        args: Any,
        metadata: JsonDict,
        final_max_size: int,
        dynamic_composed: bool,
    ) -> None:
        task_config = metadata.get("task_config", {}) if isinstance(metadata.get("task_config"), dict) else {}
        stored_strategy = task_config.get("composed_strategy", metadata.get("composed_strategy"))
        if stored_strategy is None:
            stored_strategy = "without_carry" if metadata.get("composed_without_carry", False) else "with_carry"
        stored_allow_carry = stored_strategy in ("with_carry", "with_carry_filtered")
        if stored_allow_carry != self._allow_carry_for_composed(args):
            raise ValueError(
                "Stored composed dataset carry configuration does not match current --composed-strategy. "
                "Please regenerate datasets or choose a compatible strategy."
            )
        stored_filter_flag = bool(task_config.get("filter_component_carries", metadata.get("filter_component_carries", False)))
        stored_boundary_policy = task_config.get(
            "composed_boundary_carry_policy",
            metadata.get("composed_boundary_carry_policy"),
        )
        expected_boundary_policy = self._boundary_carry_policy_for_composed(args)
        if stored_boundary_policy is None:
            stored_boundary_policy = "any"
        if stored_boundary_policy != expected_boundary_policy:
            if (
                args.composed_strategy == "with_carry_filtered"
                and stored_boundary_policy == "any"
                and stored_filter_flag
            ):
                print(
                    "[INFO] Stored metadata predates explicit boundary-carry buckets; "
                    "reusing the broad composed pool and filtering pseudo labels on-the-fly.",
                    flush=True,
                )
            else:
                raise ValueError(
                    "Stored composed dataset boundary-carry bucket does not match current --composed-strategy. "
                    "Please regenerate datasets or choose a compatible strategy."
                )
        if args.composed_strategy == "with_carry_filtered" and not stored_filter_flag:
            print(
                "[INFO] Stored metadata indicates composed dataset was generated without filtering carries; "
                "pseudo labels will be filtered on-the-fly.",
                flush=True,
            )
        stored_error_percent = float(task_config.get("composition_error_percent", metadata.get("composition_error_percent", 0.0)))
        if abs(stored_error_percent - args.composition_error_percent) > 1e-6:
            raise ValueError(
                "Stored dataset was created with a different composition_error_percent; please regenerate datasets or "
                "specify a matching value."
            )

    def summary_payload_aliases(self, summary: Any) -> JsonDict:
        boundary = summary.composed_eval_slices.get("boundary_carry")
        no_boundary = summary.composed_eval_slices.get("no_boundary_carry")
        unknown = summary.composed_eval_slices.get("unknown")
        return {
            "max_digits": summary.max_size,
            "per_digit_accuracy": {str(size): score for size, score in summary.per_size_accuracy.items()},
            "max_digits_at_90_accuracy": max(
                [size for size, score in summary.per_size_accuracy.items() if score >= 0.90],
                default=None,
            ),
            "stitched_boundary_carry_accuracy": boundary.accuracy if boundary else None,
            "stitched_no_boundary_carry_accuracy": no_boundary.accuracy if no_boundary else None,
            "stitched_unknown_accuracy": unknown.accuracy if unknown else None,
            "stitched_boundary_carry_count": boundary.count if boundary else 0,
            "stitched_no_boundary_carry_count": no_boundary.count if no_boundary else 0,
            "stitched_unknown_count": unknown.count if unknown else 0,
        }

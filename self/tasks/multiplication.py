#!/usr/bin/env python3
"""Multiplication task adapter."""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map as _default_generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask
from self.tasks.bit_common import normalize_task_format_version
from self.tasks.bit_parsing import (
    MULTIPLICATION_FORMATS,
    format_multiplication_target,
    parse_multiplication_prediction,
)
from self.tasks.bit_pseudolabels import build_direct_pseudo_examples
from self.tasks.multiplication_data import (
    MultiplicationExample,
    analyze_partial_products,
    build_multiplication_component_payload,
    build_multiplication_long_dataset,
    build_multiplication_seed_dataset,
    clone_multiplication_with_override,
    decode_multiplication_key,
    encode_multiplication_key,
    generate_long_multiplication_example,
    generate_multiplication_seed_example,
    get_multiplication_slice_name,
    iter_multiplication_sizes,
    multiplication_key,
    random_int_with_exact_digits,
    split_value_into_blocks,
)
from self.tasks.multiplication_splits import (
    prepare_multiplication_composed_eval,
    prepare_multiplication_composed_train,
    prepare_multiplication_eval_examples,
    prepare_multiplication_initial_splits,
    split_multiplication_composed_eval_slices,
)


SplitName = str


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.self_improvement_tasks")
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
        candidate_examples = [example for example in composed_examples if example.digits <= target_max_size]
        if args.pseudo_label_mode == "direct":
            return build_direct_pseudo_examples(
                candidate_examples,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                decode_max_new_tokens=decode_max_new_tokens,
                key_getter=self.key_for_example,
                prediction_parser=self.prediction_parser,
                clone_builder=self.clone_with_override,
                mode="direct",
            )
        if args.pseudo_label_mode not in {"compose", "compose_corrupt"}:
            return [], 0, {
                "mode": args.pseudo_label_mode,
                "candidate_total": len(candidate_examples),
                "retained_total": 0,
                "missing_total": 0,
            }

        component_examples: Dict[Tuple[int, int, int], MultiplicationExample] = {}
        for example in candidate_examples:
            payload = component_map.get(multiplication_key(example))
            if payload is None:
                continue
            for partial in payload.get("partials", []):
                component = MultiplicationExample(
                    a=int(partial["a"]),
                    b=int(partial["b"]),
                    digits=args.block_size,
                    result=int(partial["a"]) * int(partial["b"]),
                    operand_width=args.block_size,
                    format_version=normalize_task_format_version(args),
                )
                component_examples[multiplication_key(component)] = component

        component_predictions = generate_prediction_map(
            model=model,
            tokenizer=tokenizer,
            examples=list(component_examples.values()),
            batch_size=batch_size,
            max_new_tokens=decode_max_new_tokens,
            key_getter=multiplication_key,
            prediction_parser=self.prediction_parser,
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
                component = MultiplicationExample(
                    a=int(partial["a"]),
                    b=int(partial["b"]),
                    digits=args.block_size,
                    result=int(partial["a"]) * int(partial["b"]),
                    operand_width=args.block_size,
                    format_version=normalize_task_format_version(args),
                )
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

#!/usr/bin/env python3
"""Run-length task adapter and dataset helpers."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from self.core.evaluation import generate_prediction_map as _default_generate_prediction_map
from self.core.task_protocols import JsonDict, SelfImprovementTask
from self.tasks.bit_common import (
    BIT_COMPOSE_ARITIES,
    BIT_GUARDED_COMPOSE_RULES,
    BIT_TARGET_MODES,
    RUN_LENGTH_ALPHABET_SYMBOLS,
    RUN_LENGTH_FORMATS,
    RUN_LENGTH_TARGET_RUN_STATE,
    guard_slice_partition,
    normalize_bit_composition_path_mode,
    normalize_bit_target_mode,
    normalize_compose_arity,
    normalize_guarded_compose_rule,
    normalize_symbol_alphabet_size,
    normalize_task_format_version,
    parse_run_length_prediction,
    run_length_guard_accepts_true_components,
)
from self.tasks.bit_composition import (
    BIT_COMPOSITION_PATH_FIXED_BINARY,
    BIT_COMPOSITION_PATH_MODES,
    BIT_COMPOSITION_PATH_RANDOM,
    bit_composed_target_sizes_from_examples,
)
from self.tasks.run_length_data import (
    RunLengthExample,
    bucket_run_length_by_bits,
    build_run_length_composed_dataset,
    build_run_length_length_bucket_dataset,
    clone_run_length_with_override,
    compose_run_length_examples,
    compose_run_length_to_length,
    decode_run_length_key,
    encode_run_length_key,
    generate_run_length_example,
    merge_run_length,
    run_length_key,
)
from self.tasks.run_length_logic import compute_run_stats, format_run_length_run_state, format_run_length_target
from self.tasks.run_length_pseudolabels import derive_run_length_round_targets

SplitName = str


def _compat_symbol(name: str, fallback: Any) -> Any:
    facade = sys.modules.get("self.self_improvement_tasks")
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
        splits = {name: [] for name in ("train", "validation", "test")}
        records: Dict[SplitName, set[Tuple[int, str]]] = {name: set() for name in splits}
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
        if max_size < min_size or args.expand_train_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
            "train": {},
            "validation": {},
            "test": {},
        }
        base_used = set().union(*base_records.values())
        if additional_exclude:
            base_used.update(additional_exclude)
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
        if max_size < min_size or args.composed_eval_per_size <= 0:
            return [], {}, set()
        composed_records: Dict[SplitName, set[Tuple[int, str]]] = {"train": set(), "validation": set(), "test": set()}
        component_records: Dict[SplitName, Dict[Tuple[int, str], List[Tuple[int, str]]]] = {
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

    def prepare_eval_examples(
        self,
        rng: random.Random,
        args: Any,
        min_size: int,
        max_size: int,
        exclude: set[Tuple[int, str]],
    ) -> List[RunLengthExample]:
        generated = build_run_length_length_bucket_dataset(
            min_bits=min_size,
            max_bits=max_size,
            per_bit_counts={"train": 0, "validation": 0, "test": args.eval_per_size},
            rng=rng,
            exclude_keys=exclude,
            record_keys={split: set() for split in ("train", "validation", "test")},
            progress_name="evaluation",
            format_version=normalize_task_format_version(args),
            target_mode=normalize_bit_target_mode(args),
            alphabet=RUN_LENGTH_ALPHABET_SYMBOLS[:normalize_symbol_alphabet_size(args)],
        )
        return list(generated.get("test", []))

    def split_composed_eval_slices(
        self,
        examples: Sequence[RunLengthExample],
        component_map: Dict[Tuple[int, str], List[Tuple[int, str]]],
    ) -> Dict[str, List[RunLengthExample]]:
        if examples and examples[0].target_mode in {"plain_output", "symbol_run_pair"}:
            return guard_slice_partition(
                examples,
                component_map,
                key_getter=run_length_key,
                guard_fn=run_length_guard_accepts_true_components,
            )
        return {"all": list(examples)}

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

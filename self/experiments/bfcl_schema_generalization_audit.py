#!/usr/bin/env python3
"""Audit BFCL robustness to schema order and unseen identifier names.

The audit changes only information already present in each evaluation prompt:

* ``original`` preserves the existing prompt;
* five ``schema_permutation_*`` variants independently reorder the available
  function schemas without changing requests, targets, or accepted calls;
* ``identifier_renamed`` replaces function names and top-level argument keys
  consistently across schemas, targets, and accepted calls.

Each variant is evaluated for the atomic seed and the completed 1k Round-3 G1
adapter.  No training data or model parameters are changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from self.coding.atomic_data import (
    AtomicExample,
    canonical_json,
    read_examples,
    stable_hash,
    write_examples,
    write_json,
)
from self.coding.bfcl_composition import read_jsonl, write_jsonl
from self.coding.evaluation import (
    _bfcl_call_matches,
    evaluate_predictions,
    parse_strict_json_array,
)
from self.coding.training import generate_predictions, load_adapter_for_evaluation
from self.experiments.bfcl_compositional_pilot import DEFAULT_MODEL, DEFAULT_PYTHON


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_RUN = (
    ROOT_DIR / "artifacts/runs/bfcl_cumulative_size_sweep_20260721_132230"
)
DEFAULT_SEED_ADAPTER = (
    ROOT_DIR
    / "artifacts/runs/coding_atomic_sweep_20260718_014707"
    / "cells/bfcl/n240-s30-lr2em04-seed7/adapter"
)
DEFAULT_G1_ADAPTER = (
    DEFAULT_SOURCE_RUN / "cells/n1000-compose_g1/round_03/adapter"
)
PERMUTATION_COUNT = 5
PERMUTATION_SEED = 20260724
RENAME_SEED = 20260724
VARIANTS = (
    "original",
    *(f"schema_permutation_{index}" for index in range(PERMUTATION_COUNT)),
    "identifier_renamed",
)
MODEL_NAMES = ("seed", "g1_round3")
DATASET_NAMES = (
    "atomic_test",
    "controlled_heldout_2",
    "controlled_heldout_4",
    "controlled_heldout_8",
    "controlled_seen_2",
    "controlled_seen_4",
    "controlled_seen_8",
    "natural_parallel",
    "natural_parallel_multiple",
)
USER_SCHEMA_MARKER = "\n\nAvailable functions:\n"


@dataclass(frozen=True)
class AuditCell:
    index: int
    model_name: str
    variant: str

    @property
    def cell_id(self) -> str:
        return f"{self.model_name}--{self.variant}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "cell_id": self.cell_id,
            "model_name": self.model_name,
            "variant": self.variant,
        }


def audit_cells() -> List[AuditCell]:
    return [
        AuditCell(index=index, model_name=model_name, variant=variant)
        for index, (model_name, variant) in enumerate(
            (model_name, variant)
            for model_name in MODEL_NAMES
            for variant in VARIANTS
        )
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identifier_alias(kind: str, source_id: str, name: str) -> str:
    digest = hashlib.sha256(
        f"{RENAME_SEED}\x1f{kind}\x1f{source_id}\x1f{name}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{'fn' if kind == 'function' else 'arg'}_{digest}"


def _replace_identifier_tokens(text: str, replacements: Mapping[str, str]) -> str:
    output = text
    for old in sorted(replacements, key=lambda value: (-len(value), value)):
        output = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])",
            replacements[old],
            output,
        )
    return output


def _rename_text_values(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, list):
        return [_rename_text_values(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _rename_text_values(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return _replace_identifier_tokens(value, replacements)
    return value


def _rewrite_messages(
    example: AtomicExample,
    functions: Sequence[Mapping[str, Any]],
    *,
    request_replacements: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, str], ...]:
    messages: List[Dict[str, str]] = []
    replaced_user = False
    for raw_message in example.messages:
        message = dict(raw_message)
        if message.get("role") == "user" and USER_SCHEMA_MARKER in message.get("content", ""):
            request, _old_schemas = message["content"].rsplit(USER_SCHEMA_MARKER, 1)
            if request_replacements:
                request = _replace_identifier_tokens(request, request_replacements)
            message["content"] = (
                request + USER_SCHEMA_MARKER + canonical_json(list(functions))
            )
            replaced_user = True
        messages.append(message)
    if not replaced_user:
        raise ValueError(
            f"BFCL example {example.source_id!r} lacks {USER_SCHEMA_MARKER!r}"
        )
    return tuple(messages)


def permute_schema_order(
    example: AtomicExample,
    *,
    permutation_index: int,
) -> AtomicExample:
    if not 0 <= permutation_index < PERMUTATION_COUNT:
        raise ValueError(f"Invalid permutation index: {permutation_index}")
    original = copy.deepcopy(example.evaluator.get("functions", []))
    functions = sorted(
        original,
        key=lambda function: stable_hash(
            PERMUTATION_SEED,
            permutation_index,
            example.source_id,
            function["name"],
        ),
    )
    return AtomicExample(
        **{
            **example.__dict__,
            "messages": _rewrite_messages(example, functions),
            "evaluator": {
                **copy.deepcopy(example.evaluator),
                "functions": functions,
            },
            "metadata": {
                **copy.deepcopy(example.metadata),
                "audit_variant": f"schema_permutation_{permutation_index}",
                "schema_order_original": [
                    str(function["name"]) for function in original
                ],
                "schema_order_audit": [
                    str(function["name"]) for function in functions
                ],
            },
        }
    )


def _rename_call(
    call: Mapping[str, Any],
    *,
    function_names: Mapping[str, str],
    argument_names: Mapping[str, str],
) -> Dict[str, Any]:
    arguments = call.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Call arguments must be an object: {call!r}")
    return {
        "name": function_names.get(str(call.get("name")), str(call.get("name"))),
        "arguments": {
            argument_names.get(str(key), str(key)): copy.deepcopy(value)
            for key, value in arguments.items()
        },
    }


def rename_schema_identifiers(example: AtomicExample) -> AtomicExample:
    original_functions = copy.deepcopy(example.evaluator.get("functions", []))
    function_names = {
        str(function["name"]): _identifier_alias(
            "function", example.source_id, str(function["name"])
        )
        for function in original_functions
    }
    top_level_arguments = sorted(
        {
            str(argument)
            for function in original_functions
            for argument in (
                function.get("parameters", {}).get("properties", {}) or {}
            )
        }
    )
    argument_names = {
        argument: _identifier_alias("argument", example.source_id, argument)
        for argument in top_level_arguments
    }
    text_replacements = {**function_names, **argument_names}
    functions: List[Dict[str, Any]] = []
    for original in original_functions:
        function = copy.deepcopy(original)
        function["name"] = function_names[str(original["name"])]
        parameters = copy.deepcopy(function.get("parameters", {}))
        properties = parameters.get("properties", {}) or {}
        parameters["properties"] = {
            argument_names.get(str(key), str(key)): _rename_text_values(
                value, text_replacements
            )
            for key, value in properties.items()
        }
        parameters["required"] = [
            argument_names.get(str(key), str(key))
            for key in parameters.get("required", [])
        ]
        function["parameters"] = parameters
        for key, value in list(function.items()):
            if key not in {"name", "parameters"}:
                function[key] = _rename_text_values(value, text_replacements)
        functions.append(function)

    target_calls = json.loads(example.target)
    renamed_target = [
        _rename_call(
            call,
            function_names=function_names,
            argument_names=argument_names,
        )
        for call in target_calls
    ]
    accepted_calls = [
        _rename_call(
            call,
            function_names=function_names,
            argument_names=argument_names,
        )
        for call in example.evaluator.get("accepted_calls", [])
    ]
    return AtomicExample(
        **{
            **example.__dict__,
            "messages": _rewrite_messages(
                example,
                functions,
                request_replacements=function_names,
            ),
            "target": canonical_json(renamed_target),
            "evaluator": {
                **copy.deepcopy(example.evaluator),
                "functions": functions,
                "accepted_calls": accepted_calls,
            },
            "metadata": {
                **copy.deepcopy(example.metadata),
                "audit_variant": "identifier_renamed",
                "function_identifier_map": function_names,
                "argument_identifier_map": argument_names,
            },
        }
    )


def transform_examples(
    examples: Sequence[AtomicExample],
    variant: str,
) -> List[AtomicExample]:
    if variant == "original":
        return [
            AtomicExample(
                **{
                    **example.__dict__,
                    "metadata": {
                        **copy.deepcopy(example.metadata),
                        "audit_variant": "original",
                    },
                }
            )
            for example in examples
        ]
    if variant.startswith("schema_permutation_"):
        index = int(variant.rsplit("_", 1)[1])
        return [
            permute_schema_order(example, permutation_index=index)
            for example in examples
        ]
    if variant == "identifier_renamed":
        return [rename_schema_identifiers(example) for example in examples]
    raise ValueError(f"Unknown audit variant: {variant!r}")


def prepare(args: argparse.Namespace) -> None:
    complete = args.run_root / "PREPARED"
    if complete.exists() and args.resume:
        print(f"[INFO] Audit already prepared: {args.run_root}", flush=True)
        return
    source_sets = args.source_run_root / "data/evaluation/sets"
    missing = [
        name for name in DATASET_NAMES if not (source_sets / f"{name}.jsonl").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing source evaluation sets: {missing}")
    if not args.seed_adapter.exists():
        raise FileNotFoundError(f"Missing seed adapter: {args.seed_adapter}")
    if not args.g1_adapter.exists():
        raise FileNotFoundError(f"Missing G1 adapter: {args.g1_adapter}")

    construction_rows: List[Dict[str, Any]] = []
    for name in DATASET_NAMES:
        source = read_examples(source_sets / f"{name}.jsonl")
        for variant in VARIANTS:
            transformed = transform_examples(source, variant)
            path = args.run_root / "data/variants" / variant / f"{name}.jsonl"
            write_examples(path, transformed)
            changed_order = sum(
                example.metadata.get("schema_order_original")
                != example.metadata.get("schema_order_audit")
                for example in transformed
                if "schema_order_original" in example.metadata
            )
            construction_rows.append(
                {
                    "dataset": name,
                    "variant": variant,
                    "count": len(transformed),
                    "schema_order_changed_count": changed_order,
                    "sha256": _sha256(path),
                }
            )

    manifest = {
        "experiment": "bfcl_schema_generalization_audit",
        "status": "prepared",
        "source_run_root": str(args.source_run_root),
        "model_name": args.model_name,
        "models": {
            "seed": str(args.seed_adapter),
            "g1_round3": str(args.g1_adapter),
        },
        "variants": list(VARIANTS),
        "datasets": list(DATASET_NAMES),
        "cells": [cell.to_dict() for cell in audit_cells()],
        "construction": construction_rows,
        "prepared_at_unix": time.time(),
        "jobs": {},
    }
    write_json(args.run_root / "manifest.json", manifest)
    complete.touch()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _maximum_matching_count(
    predicted: Sequence[Any],
    references: Sequence[Mapping[str, Any]],
    predicate: Callable[[Any, Mapping[str, Any]], bool],
) -> int:
    matched_prediction_by_reference = [-1] * len(references)

    def augment(predicted_index: int, seen: List[bool]) -> bool:
        for reference_index, reference in enumerate(references):
            if seen[reference_index] or not predicate(
                predicted[predicted_index], reference
            ):
                continue
            seen[reference_index] = True
            previous = matched_prediction_by_reference[reference_index]
            if previous == -1 or augment(previous, seen):
                matched_prediction_by_reference[reference_index] = predicted_index
                return True
        return False

    return sum(
        augment(index, [False] * len(references))
        for index in range(len(predicted))
    )


def _same_function(predicted: Any, reference: Mapping[str, Any]) -> bool:
    return (
        isinstance(predicted, dict)
        and isinstance(predicted.get("name"), str)
        and predicted.get("name") == reference.get("name")
    )


def _argument_keys_match(
    predicted: Any,
    reference: Mapping[str, Any],
) -> bool:
    if not _same_function(predicted, reference):
        return False
    predicted_arguments = predicted.get("arguments")
    accepted_arguments = reference.get("arguments")
    if not isinstance(predicted_arguments, dict) or not isinstance(
        accepted_arguments, dict
    ):
        return False
    if any(key not in accepted_arguments for key in predicted_arguments):
        return False
    required = {
        key
        for key, raw_options in accepted_arguments.items()
        if "" not in (raw_options if isinstance(raw_options, list) else [raw_options])
    }
    return required.issubset(predicted_arguments)


def _value_match(predicted: Any, reference: Mapping[str, Any]) -> bool:
    return isinstance(predicted, dict) and _bfcl_call_matches(predicted, reference)


def structured_bfcl_metrics(
    examples: Sequence[AtomicExample],
    predictions: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(examples) != len(predictions):
        raise ValueError("examples and predictions must have equal lengths")
    totals = Counter()
    rows: List[Dict[str, Any]] = []
    for example, prediction in zip(examples, predictions):
        parsed, _error = parse_strict_json_array(prediction)
        predicted = parsed or []
        references = list(example.evaluator.get("accepted_calls", []))
        predicted_names = Counter(
            call.get("name")
            for call in predicted
            if isinstance(call, dict) and isinstance(call.get("name"), str)
        )
        reference_names = Counter(str(call.get("name")) for call in references)
        name_matches = _maximum_matching_count(
            predicted, references, _same_function
        )
        key_matches = _maximum_matching_count(
            predicted, references, _argument_keys_match
        )
        value_matches = _maximum_matching_count(
            predicted, references, _value_match
        )
        call_count_exact = len(predicted) == len(references)
        selection_exact = call_count_exact and predicted_names == reference_names
        key_exact = call_count_exact and key_matches == len(references)
        value_exact = call_count_exact and value_matches == len(references)
        totals.update(
            {
                "examples": 1,
                "reference_calls": len(references),
                "predicted_calls": len(predicted),
                "name_matches": name_matches,
                "key_matches": key_matches,
                "value_matches": value_matches,
                "call_count_exact": int(call_count_exact),
                "selection_exact": int(selection_exact),
                "argument_keys_exact": int(key_exact),
                "argument_values_exact": int(value_exact),
            }
        )
        rows.append(
            {
                "source_id": example.source_id,
                "reference_call_count": len(references),
                "predicted_call_count": len(predicted),
                "name_match_count": name_matches,
                "argument_key_match_count": key_matches,
                "argument_value_match_count": value_matches,
                "call_count_exact": call_count_exact,
                "selection_exact": selection_exact,
                "argument_keys_exact": key_exact,
                "argument_values_exact": value_exact,
            }
        )
    example_denominator = max(totals["examples"], 1)
    reference_denominator = max(totals["reference_calls"], 1)
    predicted_denominator = max(totals["predicted_calls"], 1)
    summary = {
        "count": totals["examples"],
        "reference_call_count": totals["reference_calls"],
        "predicted_call_count": totals["predicted_calls"],
        "call_count_accuracy": totals["call_count_exact"] / example_denominator,
        "function_selection_accuracy": totals["selection_exact"]
        / example_denominator,
        "function_name_recall": totals["name_matches"] / reference_denominator,
        "function_name_precision": totals["name_matches"] / predicted_denominator,
        "argument_key_call_accuracy": totals["key_matches"]
        / reference_denominator,
        "argument_key_exact_accuracy": totals["argument_keys_exact"]
        / example_denominator,
        "argument_value_call_accuracy": totals["value_matches"]
        / reference_denominator,
        "argument_value_exact_accuracy": totals["argument_values_exact"]
        / example_denominator,
    }
    return summary, rows


def _resolve_cell(args: argparse.Namespace) -> AuditCell:
    raw_index = args.cell_index
    if raw_index is None:
        raw_index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    cells = audit_cells()
    if not 0 <= raw_index < len(cells):
        raise ValueError(f"Cell index {raw_index} is outside [0, {len(cells)})")
    return cells[raw_index]


def _adapter_for_cell(args: argparse.Namespace, cell: AuditCell) -> Path:
    return args.seed_adapter if cell.model_name == "seed" else args.g1_adapter


def generation_budget(component_count: int, variant: str) -> int:
    """Leave extra room for deliberately long randomized identifiers."""

    baseline = max(128, 64 * int(component_count) + 32)
    return 2 * baseline if variant == "identifier_renamed" else baseline


def evaluate_cell(args: argparse.Namespace) -> None:
    cell = _resolve_cell(args)
    output_dir = args.run_root / "cells" / cell.cell_id
    complete = output_dir / "COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Audit cell already complete: {cell.cell_id}", flush=True)
        return
    adapter = _adapter_for_cell(args, cell)
    model = None
    try:
        model, tokenizer = load_adapter_for_evaluation(args.model_name, adapter)
        summaries: Dict[str, Any] = {}
        for name in DATASET_NAMES:
            examples = read_examples(
                args.run_root
                / "data/variants"
                / cell.variant
                / f"{name}.jsonl"
            )
            largest = max(
                (example.component_count for example in examples), default=1
            )
            max_new_tokens = generation_budget(largest, cell.variant)
            predictions = generate_predictions(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                batch_size=args.eval_batch_size,
                max_new_tokens=max_new_tokens,
            )
            standard, standard_rows = evaluate_predictions(examples, predictions)
            structured, structured_rows = structured_bfcl_metrics(
                examples, predictions
            )
            output_rows = []
            for example, standard_row, structured_row in zip(
                examples, standard_rows, structured_rows
            ):
                output_rows.append(
                    {
                        **standard_row,
                        "source_component_ids": list(
                            example.source_component_ids
                        ),
                        "evaluation_track": example.evaluation_track,
                        "audit_variant": cell.variant,
                        "structured": structured_row,
                    }
                )
            write_jsonl(
                output_dir / "predictions" / f"{name}.jsonl",
                output_rows,
            )
            summaries[name] = {
                **standard,
                **structured,
                "max_new_tokens": max_new_tokens,
            }
        metrics = {
            "cell": cell.to_dict(),
            "adapter": str(adapter),
            "datasets": summaries,
        }
        write_json(output_dir / "metrics.json", metrics)
        complete.touch()
        print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


SUMMARY_METRICS = (
    "exact_accuracy",
    "format_accuracy",
    "behavior_valid_accuracy",
    "call_count_accuracy",
    "function_selection_accuracy",
    "function_name_recall",
    "function_name_precision",
    "argument_key_call_accuracy",
    "argument_key_exact_accuracy",
    "argument_value_call_accuracy",
    "argument_value_exact_accuracy",
)


def _normalized_prediction(text: str) -> Optional[str]:
    parsed, _error = parse_strict_json_array(text)
    if parsed is None:
        return None
    return canonical_json(
        sorted(parsed, key=lambda call: canonical_json(call))
    )


def collect(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for cell in audit_cells():
        path = args.run_root / "cells" / cell.cell_id / "metrics.json"
        if not path.exists():
            failures.append({"cell_id": cell.cell_id, "missing": "metrics.json"})
            continue
        metrics = _read_json(path)
        for dataset, summary in metrics["datasets"].items():
            rows.append(
                {
                    "cell_index": cell.index,
                    "model_name": cell.model_name,
                    "variant": cell.variant,
                    "dataset": dataset,
                    **{
                        metric: summary[metric]
                        for metric in SUMMARY_METRICS
                    },
                }
            )

    if rows:
        with (args.run_root / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    row_by_key = {
        (row["model_name"], row["variant"], row["dataset"]): row
        for row in rows
    }
    robustness_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        for dataset in DATASET_NAMES:
            baseline = row_by_key.get((model_name, "original", dataset))
            renamed = row_by_key.get(
                (model_name, "identifier_renamed", dataset)
            )
            permutations = [
                row_by_key.get(
                    (model_name, f"schema_permutation_{index}", dataset)
                )
                for index in range(PERMUTATION_COUNT)
            ]
            if baseline is None or any(row is None for row in permutations):
                continue
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in permutations if row]
                robustness_rows.append(
                    {
                        "model_name": model_name,
                        "dataset": dataset,
                        "metric": metric,
                        "original": baseline[metric],
                        "permutation_mean": statistics.fmean(values),
                        "permutation_std": statistics.pstdev(values),
                        "permutation_min": min(values),
                        "permutation_max": max(values),
                        "permutation_mean_delta": (
                            statistics.fmean(values) - float(baseline[metric])
                        ),
                        "renamed": (
                            renamed[metric] if renamed is not None else None
                        ),
                        "renamed_delta": (
                            float(renamed[metric]) - float(baseline[metric])
                            if renamed is not None
                            else None
                        ),
                    }
                )
    if robustness_rows:
        with (args.run_root / "robustness_summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(robustness_rows[0])
            )
            writer.writeheader()
            writer.writerows(robustness_rows)

    invariance_rows: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        for dataset in DATASET_NAMES:
            variant_maps: Dict[str, Dict[str, Optional[str]]] = {}
            for variant in (
                "original",
                *(
                    f"schema_permutation_{index}"
                    for index in range(PERMUTATION_COUNT)
                ),
            ):
                path = (
                    args.run_root
                    / "cells"
                    / f"{model_name}--{variant}"
                    / "predictions"
                    / f"{dataset}.jsonl"
                )
                if not path.exists():
                    continue
                variant_maps[variant] = {
                    str(row["source_id"]): _normalized_prediction(
                        str(row["prediction"])
                    )
                    for row in read_jsonl(path)
                }
            if len(variant_maps) != PERMUTATION_COUNT + 1:
                continue
            source_ids = sorted(
                set.intersection(
                    *(set(values) for values in variant_maps.values())
                )
            )
            original = variant_maps["original"]
            agreement_counts = [
                sum(
                    variant_maps[
                        f"schema_permutation_{index}"
                    ][source_id]
                    == original[source_id]
                    for source_id in source_ids
                )
                for index in range(PERMUTATION_COUNT)
            ]
            all_consistent = sum(
                len(
                    {
                        variant_maps[variant][source_id]
                        for variant in variant_maps
                    }
                )
                == 1
                for source_id in source_ids
            )
            denominator = max(len(source_ids), 1)
            invariance_rows.append(
                {
                    "model_name": model_name,
                    "dataset": dataset,
                    "count": len(source_ids),
                    "mean_permutation_agreement_with_original": (
                        statistics.fmean(agreement_counts) / denominator
                    ),
                    "all_permutations_consistent_accuracy": (
                        all_consistent / denominator
                    ),
                }
            )
    if invariance_rows:
        with (args.run_root / "prediction_invariance.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(invariance_rows[0])
            )
            writer.writeheader()
            writer.writerows(invariance_rows)

    summary = {
        "experiment": "bfcl_schema_generalization_audit",
        "expected_cells": len(audit_cells()),
        "completed_cells": len(audit_cells()) - len(failures),
        "partial": bool(failures),
        "failures": failures,
        "metric_rows": rows,
        "robustness_rows": robustness_rows,
        "invariance_rows": invariance_rows,
    }
    write_json(args.run_root / "summary.json", summary)
    manifest = _read_json(args.run_root / "manifest.json")
    manifest["status"] = "partial" if failures else "complete"
    manifest["completed_cells"] = summary["completed_cells"]
    manifest["collected_at_unix"] = time.time()
    write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _run(command: Sequence[str], *, dry_run: bool) -> str:
    print(f"[INFO] Command: {shlex.join(list(command))}", flush=True)
    if dry_run:
        return ""
    result = subprocess.run(
        list(command),
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    return result.stdout.strip()


def _common_args(args: argparse.Namespace) -> List[str]:
    return [
        "--run-root",
        str(args.run_root),
        "--source-run-root",
        str(args.source_run_root),
        "--model-name",
        args.model_name,
        "--seed-adapter",
        str(args.seed_adapter),
        "--g1-adapter",
        str(args.g1_adapter),
        "--resume",
    ]


def submit(args: argparse.Namespace) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    evaluate_command = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        str(args.python_bin),
        "-m",
        "self.experiments.bfcl_schema_generalization_audit",
        "evaluate-cell",
        *_common_args(args),
        "--eval-batch-size",
        str(args.eval_batch_size),
    ]
    evaluate_sbatch = [
        "sbatch",
        "--parsable",
        "--partition=ailab",
        "--gres=gpu:h200:1",
        "--cpus-per-task=8",
        "--mem=64G",
        f"--time={args.time_limit}",
        "--job-name=bfcl-schema-audit",
        f"--array=0-{len(audit_cells()) - 1}%{args.max_concurrent}",
        f"--output={args.log_dir / 'bfcl-schema-audit-%A_%a.out'}",
        f"--error={args.log_dir / 'bfcl-schema-audit-%A_%a.err'}",
        f"--wrap={shlex.join(evaluate_command)}",
    ]
    output = _run(evaluate_sbatch, dry_run=args.dry_run)
    evaluate_job = (
        "dryrun-bfcl-schema-audit"
        if args.dry_run
        else output.split(";")[0].strip()
    )

    collect_command = [
        "env",
        str(args.python_bin),
        "-m",
        "self.experiments.bfcl_schema_generalization_audit",
        "collect",
        *_common_args(args),
    ]
    collect_sbatch = [
        "sbatch",
        "--parsable",
        "--partition=cpu",
        "--cpus-per-task=1",
        "--mem=8G",
        "--time=00:15:00",
        "--job-name=bfcl-schema-collect",
        f"--dependency=afterany:{evaluate_job}",
        f"--output={args.log_dir / 'bfcl-schema-collect-%j.out'}",
        f"--error={args.log_dir / 'bfcl-schema-collect-%j.err'}",
        f"--wrap={shlex.join(collect_command)}",
    ]
    output = _run(collect_sbatch, dry_run=args.dry_run)
    collect_job = (
        "dryrun-bfcl-schema-collect"
        if args.dry_run
        else output.split(";")[0].strip()
    )
    jobs = {
        "evaluate_array": evaluate_job,
        "collector": collect_job,
        "dependency_policy": "afterany",
    }
    if not args.dry_run:
        manifest = _read_json(args.run_root / "manifest.json")
        manifest["jobs"] = jobs
        manifest["status"] = "submitted"
        manifest["submitted_at_unix"] = time.time()
        write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(jobs, indent=2, sort_keys=True), flush=True)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN
    )
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--seed-adapter", type=Path, default=DEFAULT_SEED_ADAPTER
    )
    parser.add_argument("--g1-adapter", type=Path, default=DEFAULT_G1_ADAPTER)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    _add_common(prepare_parser)
    evaluate_parser = subparsers.add_parser("evaluate-cell")
    _add_common(evaluate_parser)
    evaluate_parser.add_argument("--cell-index", type=int)
    evaluate_parser.add_argument("--eval-batch-size", type=int, default=8)
    collect_parser = subparsers.add_parser("collect")
    _add_common(collect_parser)
    submit_parser = subparsers.add_parser("submit")
    _add_common(submit_parser)
    submit_parser.add_argument(
        "--python-bin", type=Path, default=DEFAULT_PYTHON
    )
    submit_parser.add_argument("--log-dir", type=Path)
    submit_parser.add_argument("--eval-batch-size", type=int, default=8)
    submit_parser.add_argument("--max-concurrent", type=int, default=4)
    submit_parser.add_argument("--time-limit", default="01:15:00")
    submit_parser.add_argument("--dry-run", action="store_true")
    return parser


def _normalize(args: argparse.Namespace) -> None:
    args.run_root = args.run_root.resolve()
    args.source_run_root = args.source_run_root.resolve()
    args.seed_adapter = args.seed_adapter.resolve()
    args.g1_adapter = args.g1_adapter.resolve()
    if hasattr(args, "log_dir"):
        args.log_dir = (args.log_dir or args.run_root / "logs").resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _normalize(args)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "evaluate-cell":
        evaluate_cell(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "submit":
        submit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

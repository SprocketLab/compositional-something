#!/usr/bin/env python3
"""Three-round cumulative BFCL 1k/2k/3k data-size sweep."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from self.coding.evaluation import evaluate_predictions
from self.coding.atomic_data import (
    AtomicExample,
    read_examples,
    stable_hash,
    write_examples,
    write_json,
)
from self.coding.bfcl_composition import (
    COMPONENT_CONTEXT_MODES,
    build_controlled_evaluation_candidates,
    build_hierarchical_cross_candidates,
    build_next_repeat_candidates,
    build_round1_repeat_candidates,
    read_jsonl,
    sha256_path,
    write_jsonl,
)
from self.coding.training import (
    adapter_parameter_summary,
    load_adapter_for_evaluation,
    load_adapter_for_training,
    load_qwen_tokenizer,
    train_lora,
)
from self.experiments.bfcl_compositional_pilot import (
    COMPOSE_CONDITIONS,
    DEFAULT_ATOMIC_DATA,
    DIRECT_CONDITIONS,
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_SEED_ADAPTER,
    _condition_decisions,
    _evaluate_all,
    _evaluate_split,
    _evaluation_sets,
    _filter_training_length,
    _generate_rows,
    _persist_decisions,
    _pseudo_examples,
    _raw_map,
    _repeat_to_count,
    _unique_component_specs,
)
from self.coding.bfcl_composition import (
    compose_component_predictions,
    oracle_example,
    public_candidate_to_example,
    public_spec_to_example,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_SEED = 20260721
TRAINING_SEED = 7
SIZES = (1000, 2000, 3000)
CONDITIONS = (
    "direct_g4",
    "compose_g1",
    "compose_g4",
    "compose_g4_repeat20",
    "oracle",
)
SUPPORTED_CONDITIONS = (
    "direct_g1",
    "direct_g4",
    "compose_g1",
    "compose_g4",
    "compose_g4_repeat20",
    "oracle",
)
CALLS_BY_ROUND = {1: (2,), 2: (2, 4), 3: (2, 4, 8)}
CANDIDATE_POOL = 5000
REPEAT_POOL = 1000
MAX_LENGTH = 2048
LEARNING_RATE = 2e-4
EFFECTIVE_BATCH_SIZE = 16
PRIMARY_CONDITION = "compose_g1"
ACCEPTANCE_RESERVE = 0.75
# Selection cells built from the otherwise unused 40-atom validation pool.
VALIDATION_CALL_COUNTS = (2, 4)
VALIDATION_EXAMPLES_PER_CELL = 100
DEFAULT_COMPONENT_CONTEXT = "component_schemas"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _raw_map_rows(rows: Sequence[Mapping[str, Any]], id_key: str) -> Dict[str, str]:
    return {str(row[id_key]): str(row["prediction"]) for row in rows}


def _format_learning_rate(learning_rate: float) -> str:
    """Render a learning rate as a filesystem-safe cell-ID fragment."""

    return "lr" + f"{float(learning_rate):.0e}".replace("-", "m").replace("+", "p")


def _cell_grid(
    sizes: Sequence[int] = SIZES,
    conditions: Sequence[str] = CONDITIONS,
    learning_rates: Sequence[float] = (LEARNING_RATE,),
) -> List[Dict[str, Any]]:
    triples = [
        (size, condition, learning_rate)
        for size in sizes
        for condition in conditions
        for learning_rate in learning_rates
    ]
    return [
        {
            "cell_index": index,
            "size": size,
            "condition": condition,
            "learning_rate": float(learning_rate),
            "cell_id": f"n{size:04d}-{condition}-{_format_learning_rate(learning_rate)}",
        }
        for index, (size, condition, learning_rate) in enumerate(triples)
    ]


def _run_grid(run_root: Path) -> List[Dict[str, Any]]:
    """Read the grid a run was prepared with, falling back to the default.

    Reading it back keeps archived runs collectible after the default condition
    or size list changes.
    """

    grid_path = run_root / "grid.json"
    if grid_path.exists():
        return list(_read_json_any(grid_path))
    return _cell_grid()


def _cell(args: argparse.Namespace) -> Dict[str, Any]:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    index = args.cell_index if args.cell_index is not None else (int(raw) if raw is not None else None)
    if index is None:
        raise ValueError("Provide --cell-index or SLURM_ARRAY_TASK_ID")
    grid = _run_grid(args.run_root)
    if index < 0 or index >= len(grid):
        raise ValueError(f"Cell index {index} is outside [0, {len(grid)})")
    return grid[index]


def _cell_round_dir(run_root: Path, cell: Mapping[str, Any], round_index: int) -> Path:
    return run_root / "cells" / str(cell["cell_id"]) / f"round_{round_index:02d}"


SELECTION_FILE = "selected_checkpoint.json"


def _selected_adapter(round_dir: Path) -> Optional[Path]:
    path = round_dir / SELECTION_FILE
    if not path.exists():
        return None
    adapter = Path(str(_read_json(path)["adapter"]))
    if not adapter.is_dir():
        raise FileNotFoundError(f"Selected checkpoint is missing: {adapter}")
    return adapter


def _starting_adapter(
    args: argparse.Namespace, cell: Mapping[str, Any], round_index: int
) -> Path:
    """Where a round continues from.

    The optimal training length moves between rounds, so a round continues from
    the previous round's *selected* checkpoint when one was recorded, and from
    its final adapter otherwise.
    """

    if round_index == 1:
        return args.seed_adapter
    previous = _cell_round_dir(args.run_root, cell, round_index - 1)
    return _selected_adapter(previous) or previous / "adapter"


def _candidate_paths(run_root: Path, calls: int, family: str) -> Tuple[Path, Path]:
    stem = f"calls_{calls}_{family}.jsonl"
    return (
        run_root / "data/public_candidates" / stem,
        run_root / "data/oracle" / stem,
    )


def _load_candidates(
    run_root: Path, calls: int, family: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    public_path, oracle_path = _candidate_paths(run_root, calls, family)
    return read_jsonl(public_path), read_jsonl(oracle_path)


def _write_candidates(
    run_root: Path,
    calls: int,
    family: str,
    public: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
) -> None:
    public_path, oracle_path = _candidate_paths(run_root, calls, family)
    write_jsonl(public_path, public)
    write_jsonl(oracle_path, oracle)


def _length_filter_and_take(
    tokenizer: Any,
    public: Sequence[Mapping[str, Any]],
    oracle: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kept_public, kept_oracle, audit = _filter_training_length(
        tokenizer, public, oracle, max_length=MAX_LENGTH
    )
    if len(kept_public) < count:
        raise ValueError(
            f"Only {len(kept_public)} candidates fit max_length={MAX_LENGTH}; need {count}"
        )
    chosen_public = kept_public[:count]
    chosen_ids = {str(row["candidate_id"]) for row in chosen_public}
    if len(chosen_ids) != count:
        raise AssertionError(
            f"Candidate IDs are not unique: {count} rows carry {len(chosen_ids)} IDs"
        )
    chosen_oracle = [row for row in kept_oracle if str(row["candidate_id"]) in chosen_ids]
    if len(chosen_oracle) != count:
        raise AssertionError("Public/oracle candidate counts diverged")
    return chosen_public, chosen_oracle, {**audit, "selected_count": count}


def _checksums(root: Path, relative_root: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(relative_root)): sha256_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def prepare(args: argparse.Namespace) -> None:
    complete = args.run_root / "PREPARED"
    if complete.exists() and args.resume:
        print(f"[INFO] Already prepared: {args.run_root}", flush=True)
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    train = read_examples(args.atomic_data_dir / "train.jsonl")
    hidden = read_examples(args.atomic_data_dir / "hidden_composition.jsonl")
    validation = read_examples(args.atomic_data_dir / "validation.jsonl")
    test = read_examples(args.atomic_data_dir / "test.jsonl")
    if (len(train), len(hidden), len(validation), len(test)) != (240, 60, 40, 60):
        raise ValueError("Expected the pinned 240/60/40/60 BFCL split")
    source_pool = [*train, *hidden]
    if len({item.source_id for item in source_pool}) != 300:
        raise ValueError("The combined composition source pool must contain 300 unique atoms")
    tokenizer = load_qwen_tokenizer(args.model_name)

    audit: Dict[str, Any] = {
        "source_pool": {
            "seed_train": len(train),
            "hidden_composition": len(hidden),
            "total": len(source_pool),
        },
        "candidate_pool": args.candidate_pool,
        "repeat_pool": args.repeat_pool,
        "max_length": MAX_LENGTH,
        "regimes": {},
    }
    audit["component_context"] = args.component_context
    for calls in (2, 4, 8):
        public, oracle = build_hierarchical_cross_candidates(
            source_pool,
            component_count=calls,
            count=args.candidate_pool + 1000,
            seed=args.data_seed,
            component_context=args.component_context,
        )
        public, oracle, length_audit = _length_filter_and_take(
            tokenizer, public, oracle, count=args.candidate_pool
        )
        _write_candidates(args.run_root, calls, "cross", public, oracle)
        audit["regimes"][f"calls_{calls}_cross"] = {
            "count": len(public),
            "distinct_sources": len(
                {source for row in public for source in row["source_component_ids"]}
            ),
            "length_filter": length_audit,
        }

    repeat2_public, repeat2_oracle, repeat2_audit = build_round1_repeat_candidates(
        source_pool,
        split="hidden_composition",
        seed=args.data_seed,
        max_variants_per_source=4,
        template_partition="train",
        renders_per_pair=2,
        component_context=args.component_context,
    )
    repeat2_public, repeat2_oracle, repeat2_lengths = _length_filter_and_take(
        tokenizer, repeat2_public, repeat2_oracle, count=args.repeat_pool
    )
    _write_candidates(args.run_root, 2, "repeat", repeat2_public, repeat2_oracle)
    audit["regimes"]["calls_2_repeat"] = {
        **repeat2_audit,
        "count": len(repeat2_public),
        "length_filter": repeat2_lengths,
    }

    previous_public, previous_oracle = repeat2_public, repeat2_oracle
    for round_index, calls in ((2, 4), (3, 8)):
        public, oracle = build_next_repeat_candidates(
            previous_public,
            previous_oracle,
            round_index=round_index,
            component_call_count=calls // 2,
            seed=args.data_seed,
            count=args.repeat_pool + 200,
            component_context=args.component_context,
        )
        public, oracle, length_audit = _length_filter_and_take(
            tokenizer, public, oracle, count=args.repeat_pool
        )
        _write_candidates(args.run_root, calls, "repeat", public, oracle)
        audit["regimes"][f"calls_{calls}_repeat"] = {
            "count": len(public),
            "distinct_sources": len(
                {source for row in public for source in row["source_component_ids"]}
            ),
            "length_filter": length_audit,
        }
        previous_public, previous_oracle = public, oracle

    sets_root = args.run_root / "data/evaluation/sets"
    candidates_root = args.run_root / "data/evaluation/candidates"
    write_examples(sets_root / "atomic_test.jsonl", test)
    for name, (public, oracle) in build_controlled_evaluation_candidates(
        test, seed=args.data_seed, component_context=args.component_context
    ).items():
        write_examples(
            sets_root / f"{name}.jsonl",
            [oracle_example(row, hidden) for row, hidden in zip(public, oracle)],
        )
        # The frozen baseline decomposes evaluation items, so it needs specs.
        write_jsonl(candidates_root / f"{name}.jsonl", public)
    # Hyperparameter selection reads these; the test cells above stay closed
    # until the recipe is frozen.
    validation_root = args.run_root / "data/evaluation/validation_sets"
    write_examples(validation_root / "atomic.jsonl", validation)
    for name, (public, oracle) in build_controlled_evaluation_candidates(
        validation,
        component_counts=VALIDATION_CALL_COUNTS,
        examples_per_cell=VALIDATION_EXAMPLES_PER_CELL,
        seed=args.data_seed,
        component_context=args.component_context,
    ).items():
        write_examples(
            validation_root / f"{name}.jsonl",
            [oracle_example(row, hidden) for row, hidden in zip(public, oracle)],
        )
        write_jsonl(candidates_root / f"validation_{name}.jsonl", public)
    for name in ("natural_parallel", "natural_parallel_multiple"):
        write_examples(
            sets_root / f"{name}.jsonl",
            read_examples(args.atomic_data_dir / "frontier" / f"{name}.jsonl"),
        )
    audit["evaluation_counts"] = {
        path.stem: len(read_examples(path)) for path in sorted(sets_root.glob("*.jsonl"))
    }
    audit["validation_counts"] = {
        path.stem: len(read_examples(path))
        for path in sorted(validation_root.glob("*.jsonl"))
    }
    write_json(args.run_root / "data/audit.json", audit)
    write_json(
        args.run_root / "grid.json",
        _cell_grid(args.sizes, args.conditions, args.learning_rates),
    )
    write_json(args.run_root / "data_checksums.json", _checksums(args.run_root / "data", args.run_root))
    manifest = {
        "experiment": "bfcl_cumulative_size_sweep",
        "created_at_unix": time.time(),
        "status": "prepared",
        "model_name": args.model_name,
        "seed_adapter": str(args.seed_adapter),
        "atomic_data_dir": str(args.atomic_data_dir),
        "sizes_per_regime": list(args.sizes),
        "conditions": list(args.conditions),
        "component_context": args.component_context,
        "round_regimes": {str(key): list(value) for key, value in CALLS_BY_ROUND.items()},
        "source_pool": {"seed_train": 240, "hidden_composition": 60},
        "candidate_pool": args.candidate_pool,
        "repeat_pool": args.repeat_pool,
        "max_length": MAX_LENGTH,
        "learning_rates": [float(rate) for rate in args.learning_rates],
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "training_seed": TRAINING_SEED,
        "curriculum_policy": "fixed_manual_cumulative_refresh",
        "promotion_gate": None,
        "grid": _cell_grid(args.sizes, args.conditions, args.learning_rates),
        "jobs": {},
    }
    write_json(args.run_root / "manifest.json", manifest)
    complete.touch()
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def _quota_limit(quota: int, available: int) -> int:
    return min(available, max(quota, math.ceil(quota / ACCEPTANCE_RESERVE)))


def _family_quota(size: int, condition: str, family: str) -> int:
    if condition != "compose_g4_repeat20":
        return size if family == "cross" else 0
    repeat = round(size * 0.20)
    return repeat if family == "repeat" else size - repeat


def _raw_predictions(
    *,
    model: Any,
    tokenizer: Any,
    condition: str,
    candidates: Sequence[Mapping[str, Any]],
    output_dir: Path,
    eval_batch_size: int,
    calls: int,
) -> Tuple[Optional[Dict[str, str]], Optional[Dict[str, str]]]:
    if condition in DIRECT_CONDITIONS:
        rows = _generate_rows(
            model=model,
            tokenizer=tokenizer,
            examples=[public_candidate_to_example(row) for row in candidates],
            identifiers=[str(row["candidate_id"]) for row in candidates],
            batch_size=eval_batch_size,
            max_new_tokens=max(192, 64 * calls + 32),
            id_key="candidate_id",
        )
        write_jsonl(output_dir / "direct.jsonl", rows)
        return None, {str(row["candidate_id"]): str(row["prediction"]) for row in rows}
    specs = _unique_component_specs(candidates)
    rows = _generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=[public_spec_to_example(spec) for spec in specs],
        identifiers=[str(spec["component_id"]) for spec in specs],
        batch_size=eval_batch_size,
        max_new_tokens=max(128, 64 * (calls // 2) + 32),
        id_key="component_id",
    )
    write_jsonl(output_dir / "components.jsonl", rows)
    return {str(row["component_id"]): str(row["prediction"]) for row in rows}, None


def _records_for_regime(
    *,
    run_root: Path,
    output_dir: Path,
    condition: str,
    calls: int,
    curriculum_round: int,
    family: str,
    quota: int,
    model: Optional[Any],
    tokenizer: Optional[Any],
    eval_batch_size: int,
    prediction_checkpoint: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    public, oracle = _load_candidates(run_root, calls, family)
    limit = quota if condition == "oracle" else _quota_limit(quota, len(public))
    public = public[:limit]
    allowed = {str(row["candidate_id"]) for row in public}
    oracle = [row for row in oracle if str(row["candidate_id"]) in allowed]
    raw_components: Optional[Dict[str, str]] = None
    raw_direct: Optional[Dict[str, str]] = None
    if condition != "oracle":
        if model is None or tokenizer is None:
            raise ValueError("Learned conditions require a loaded model")
        raw_components, raw_direct = _raw_predictions(
            model=model,
            tokenizer=tokenizer,
            condition=condition,
            candidates=public,
            output_dir=output_dir / "raw_predictions",
            eval_batch_size=eval_batch_size,
            calls=calls,
        )
    oracle_by_id = {str(row["candidate_id"]): row for row in oracle}
    decisions = _condition_decisions(
        condition,
        public,
        raw_components=raw_components,
        raw_direct=raw_direct,
        oracle_by_id=oracle_by_id if condition == "oracle" else None,
    )
    summary, records = _persist_decisions(
        output_dir,
        public,
        oracle,
        decisions,
        round_index=curriculum_round,
        condition=condition,
        prediction_checkpoint=prediction_checkpoint,
    )
    if len(records) < quota:
        raise ValueError(
            f"{condition} calls={calls} family={family} accepted {len(records)}; need {quota}"
        )
    return records[:quota], summary


def _learned_conditions(run_root: Path) -> List[str]:
    """Grid conditions that need model generations; ``oracle`` reads gold calls."""

    return [
        str(cell["condition"])
        for cell in _run_grid(run_root)
        if str(cell["condition"]) != "oracle"
    ]


def generate_round1_shared(args: argparse.Namespace) -> None:
    output_dir = args.run_root / "shared/round_01"
    complete = output_dir / "GENERATE_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Shared Round-1 predictions already complete: {output_dir}")
        return
    if not _learned_conditions(args.run_root):
        # An Oracle-only grid never reads these predictions.
        output_dir.mkdir(parents=True, exist_ok=True)
        complete.touch()
        print(json.dumps({"skipped": "no learned conditions in grid"}, sort_keys=True))
        return
    model, tokenizer = load_adapter_for_evaluation(args.model_name, args.seed_adapter)
    # Prefetch only what this run's grid can consume, not what the default
    # size list would have needed.
    largest_size = max(int(cell["size"]) for cell in _run_grid(args.run_root))
    try:
        cross_public, _cross_oracle = _load_candidates(args.run_root, 2, "cross")
        cross_public = cross_public[: _quota_limit(largest_size, len(cross_public))]
        direct_rows = _generate_rows(
            model=model,
            tokenizer=tokenizer,
            examples=[public_candidate_to_example(row) for row in cross_public],
            identifiers=[str(row["candidate_id"]) for row in cross_public],
            batch_size=args.eval_batch_size,
            max_new_tokens=192,
            id_key="candidate_id",
        )
        write_jsonl(output_dir / "cross/direct.jsonl", direct_rows)
        repeat_public = _load_candidates(args.run_root, 2, "repeat")[0]
        for family, candidates in (
            ("cross", cross_public),
            (
                "repeat",
                repeat_public[
                    : _quota_limit(round(largest_size * 0.20), len(repeat_public))
                ],
            ),
        ):
            specs = _unique_component_specs(candidates)
            rows = _generate_rows(
                model=model,
                tokenizer=tokenizer,
                examples=[public_spec_to_example(spec) for spec in specs],
                identifiers=[str(spec["component_id"]) for spec in specs],
                batch_size=args.eval_batch_size,
                max_new_tokens=128,
                id_key="component_id",
            )
            write_jsonl(output_dir / family / "components.jsonl", rows)
        complete.touch()
        print(
            json.dumps(
                {
                    "direct_candidates": len(cross_public),
                    "cross_components": len(
                        read_jsonl(output_dir / "cross/components.jsonl")
                    ),
                    "repeat_components": len(
                        read_jsonl(output_dir / "repeat/components.jsonl")
                    ),
                },
                sort_keys=True,
            )
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def materialize_round1_shared(args: argparse.Namespace) -> None:
    shared = args.run_root / "shared/round_01"
    if not (shared / "GENERATE_COMPLETE").exists():
        raise FileNotFoundError("Shared Round-1 generation is incomplete")
    if _learned_conditions(args.run_root):
        direct_raw = _raw_map(shared / "cross/direct.jsonl", "candidate_id")
        component_raw = {
            family: _raw_map(shared / family / "components.jsonl", "component_id")
            for family in ("cross", "repeat")
        }
    else:
        direct_raw = {}
        component_raw = {family: {} for family in ("cross", "repeat")}
    for cell in _run_grid(args.run_root):
        output_dir = _cell_round_dir(args.run_root, cell, 1)
        complete = output_dir / "GENERATE_MATERIALIZE_COMPLETE"
        if complete.exists() and args.resume:
            continue
        condition = str(cell["condition"])
        size = int(cell["size"])
        records: Dict[Tuple[int, str], Sequence[Mapping[str, Any]]] = {}
        summaries: Dict[Tuple[int, str], Mapping[str, Any]] = {}
        for family in ("cross", "repeat"):
            quota = _family_quota(size, condition, family)
            if quota == 0:
                continue
            public, oracle = _load_candidates(args.run_root, 2, family)
            limit = quota if condition == "oracle" else _quota_limit(quota, len(public))
            public = public[:limit]
            allowed = {str(row["candidate_id"]) for row in public}
            oracle = [row for row in oracle if str(row["candidate_id"]) in allowed]
            oracle_by_id = {str(row["candidate_id"]): row for row in oracle}
            decisions = _condition_decisions(
                condition,
                public,
                raw_components=(
                    component_raw[family]
                    if condition in COMPOSE_CONDITIONS
                    else None
                ),
                raw_direct=direct_raw if condition in DIRECT_CONDITIONS else None,
                oracle_by_id=oracle_by_id if condition == "oracle" else None,
            )
            regime_dir = output_dir / f"regimes/calls_2/{family}"
            summary, accepted = _persist_decisions(
                regime_dir,
                public,
                oracle,
                decisions,
                round_index=1,
                condition=condition,
                prediction_checkpoint=(
                    "hidden_oracle" if condition == "oracle" else str(args.seed_adapter)
                ),
            )
            if len(accepted) < quota:
                raise ValueError(
                    f"{cell['cell_id']} family={family} accepted {len(accepted)}; need {quota}"
                )
            records[(2, family)] = accepted[:quota]
            summaries[(2, family)] = summary
        _materialize(
            args=args,
            cell=cell,
            round_index=1,
            records=records,
            summaries=summaries,
        )
        complete.touch()
    print(json.dumps({"materialized_cells": len(_run_grid(args.run_root)), "round": 1}, sort_keys=True))


def _materialize(
    *,
    args: argparse.Namespace,
    cell: Mapping[str, Any],
    round_index: int,
    records: Mapping[Tuple[int, str], Sequence[Mapping[str, Any]]],
    summaries: Mapping[Tuple[int, str], Mapping[str, Any]],
) -> Dict[str, Any]:
    size = int(cell["size"])
    condition = str(cell["condition"])
    output_dir = _cell_round_dir(args.run_root, cell, round_index)
    atomic_train = read_examples(args.atomic_data_dir / "train.jsonl")
    atomic = _repeat_to_count(
        atomic_train,
        size,
        seed=EXPERIMENT_SEED + round_index,
        origin="atomic_gold_replay",
    )
    materialized: List[AtomicExample] = list(atomic)
    mix: Dict[str, Any] = {"atomic_1_call": len(atomic)}
    selection_rows: List[Dict[str, Any]] = []
    for calls in CALLS_BY_ROUND[round_index]:
        selected: List[Mapping[str, Any]] = []
        for family in ("cross", "repeat"):
            selected.extend(records.get((calls, family), ()))
        if len(selected) != size:
            raise ValueError(
                f"Expected {size} selected calls={calls} records, found {len(selected)}"
            )
        examples = _pseudo_examples(
            selected,
            limit=size,
            round_index=round_index,
            condition=condition,
        )
        examples = [
            AtomicExample(
                **{
                    **example.__dict__,
                    "metadata": {
                        **copy.deepcopy(example.metadata),
                        "training_origin": f"composed_{calls}_call",
                        "regime_calls": calls,
                        "refresh_round": round_index,
                    },
                }
            )
            for example in examples
        ]
        write_examples(output_dir / f"selected/calls_{calls}.jsonl", examples)
        materialized.extend(examples)
        mix[f"composed_{calls}_call"] = len(examples)
        for index, record in enumerate(selected):
            selection_rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "calls": calls,
                    "family": record["candidate"]["composition_family"],
                    "selection_index": index,
                }
            )
    materialized.sort(
        key=lambda item: stable_hash(
            EXPERIMENT_SEED,
            round_index,
            item.metadata.get("training_origin", ""),
            item.source_id,
            item.metadata.get("replay_instance", -1),
        )
    )
    write_examples(output_dir / "training_materialized/train.jsonl", materialized)
    write_jsonl(output_dir / "training_materialized/selection.jsonl", selection_rows)
    mix.update(
        {
            "total": len(materialized),
            "size_per_regime": size,
            "regimes": [1, *CALLS_BY_ROUND[round_index]],
            "trainer_seed": TRAINING_SEED,
        }
    )
    write_json(output_dir / "training_materialized/mix.json", mix)
    write_json(
        output_dir / "materialization.json",
        {
            "cell": dict(cell),
            "round": round_index,
            "mix": mix,
            "guard_summaries": {
                f"calls_{calls}_{family}": dict(summary)
                for (calls, family), summary in summaries.items()
            },
        },
    )
    (output_dir / "MATERIALIZED").touch()
    return mix


def generate_materialize(args: argparse.Namespace) -> None:
    cell = _cell(args)
    round_index = args.round
    output_dir = _cell_round_dir(args.run_root, cell, round_index)
    complete = output_dir / "GENERATE_MATERIALIZE_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Already generated/materialized: {cell['cell_id']} round={round_index}")
        return
    condition = str(cell["condition"])
    size = int(cell["size"])
    starting_adapter = _starting_adapter(args, cell, round_index)
    if not starting_adapter.exists():
        raise FileNotFoundError(f"Missing prerequisite adapter: {starting_adapter}")
    model: Optional[Any] = None
    tokenizer: Optional[Any] = None
    if condition != "oracle":
        model, tokenizer = load_adapter_for_evaluation(args.model_name, starting_adapter)
    records: Dict[Tuple[int, str], Sequence[Mapping[str, Any]]] = {}
    summaries: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    try:
        for calls in CALLS_BY_ROUND[round_index]:
            for family in ("cross", "repeat"):
                quota = _family_quota(size, condition, family)
                if quota == 0:
                    continue
                regime_dir = output_dir / f"regimes/calls_{calls}/{family}"
                selected, summary = _records_for_regime(
                    run_root=args.run_root,
                    output_dir=regime_dir,
                    condition=condition,
                    calls=calls,
                    curriculum_round=round_index,
                    family=family,
                    quota=quota,
                    model=model,
                    tokenizer=tokenizer,
                    eval_batch_size=args.eval_batch_size,
                    prediction_checkpoint="hidden_oracle" if condition == "oracle" else str(starting_adapter),
                )
                records[(calls, family)] = selected
                summaries[(calls, family)] = summary
        mix = _materialize(
            args=args,
            cell=cell,
            round_index=round_index,
            records=records,
            summaries=summaries,
        )
        complete.touch()
        print(json.dumps({"cell": cell, "round": round_index, "mix": mix}, sort_keys=True))
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_evaluate(args: argparse.Namespace) -> None:
    cell = _cell(args)
    output_dir = _cell_round_dir(args.run_root, cell, args.round)
    complete = output_dir / "TRAIN_EVAL_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Already trained/evaluated: {cell['cell_id']} round={args.round}")
        return
    materialized = output_dir / "MATERIALIZED"
    if not materialized.exists():
        raise FileNotFoundError(f"Missing prerequisite materialization: {materialized}")
    examples = read_examples(output_dir / "training_materialized/train.jsonl")
    starting_adapter = _starting_adapter(args, cell, args.round)
    learning_rate = float(cell.get("learning_rate", LEARNING_RATE))
    max_steps = args.max_steps or math.ceil(len(examples) / EFFECTIVE_BATCH_SIZE)
    micro_batch = args.micro_batch_size
    attempts: List[Dict[str, Any]] = []
    while micro_batch >= 1:
        model = None
        try:
            torch.manual_seed(TRAINING_SEED)
            torch.cuda.manual_seed_all(TRAINING_SEED)
            model, tokenizer = load_adapter_for_training(args.model_name, starting_adapter)
            parameters = adapter_parameter_summary(model)
            training = train_lora(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                output_dir=output_dir,
                max_length=MAX_LENGTH,
                max_steps=max_steps,
                learning_rate=learning_rate,
                micro_batch_size=micro_batch,
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
                seed=TRAINING_SEED,
                checkpoint_steps=args.checkpoint_steps,
            )
            evaluation = _evaluate_all(
                model=model,
                tokenizer=tokenizer,
                run_root=args.run_root,
                output_dir=output_dir / "evaluation",
                batch_size=args.eval_batch_size,
            )
            metrics = {
                "cell": dict(cell),
                "round": args.round,
                "starting_adapter": str(starting_adapter),
                "adapter": str(output_dir / "adapter"),
                "training_example_count": len(examples),
                "max_steps": max_steps,
                "one_epoch_steps": math.ceil(len(examples) / EFFECTIVE_BATCH_SIZE),
                "learning_rate": learning_rate,
                "max_length": MAX_LENGTH,
                "training": training,
                "evaluation": evaluation,
                "parameters": parameters,
                "oom_attempts": attempts,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            write_json(output_dir / "metrics.json", metrics)
            complete.touch()
            print(json.dumps({"cell": cell, "round": args.round, "evaluation": evaluation}, sort_keys=True))
            return
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or micro_batch == 1:
                raise
            attempts.append({"micro_batch_size": micro_batch, "error": str(exc)})
            micro_batch //= 2
            print(f"[WARN] CUDA OOM; retrying micro_batch_size={micro_batch}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _checkpoint_adapters(round_dir: Path) -> List[Tuple[int, Path]]:
    """Every saved adapter for a round, keyed by optimizer step."""

    found: List[Tuple[int, Path]] = []
    for path in sorted(round_dir.glob("adapter_step_*")):
        if path.is_dir():
            found.append((int(path.name.rsplit("_", 1)[1]), path))
    final = round_dir / "adapter"
    metrics_path = round_dir / "metrics.json"
    if final.is_dir() and metrics_path.exists():
        final_step = int(_read_json(metrics_path)["max_steps"])
        # A requested checkpoint at max_steps is the same model as the final
        # adapter; evaluating both wastes a pass and duplicates the curve.
        if final_step not in {step for step, _path in found}:
            found.append((final_step, final))
    return sorted(found)


def evaluate_checkpoints(args: argparse.Namespace) -> None:
    """Score every saved checkpoint so a training length can be selected.

    Defaults to the validation cells only: selecting a checkpoint on the test
    cells and then reporting those same cells inflates the reported number.
    """

    cell = _cell(args)
    round_dir = _cell_round_dir(args.run_root, cell, args.round)
    checkpoints = _checkpoint_adapters(round_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No adapters to evaluate under {round_dir}")
    wanted = [
        (name, examples)
        for name, examples in _evaluation_sets(args.run_root)
        if args.sets == "all" or name.startswith("validation_")
    ]
    if not wanted:
        raise FileNotFoundError(
            "No validation cells in this run; re-prepare to build them, or pass --sets all"
        )
    rows: List[Dict[str, Any]] = []
    for step, adapter in checkpoints:
        model = None
        try:
            model, tokenizer = load_adapter_for_evaluation(args.model_name, adapter)
            for name, examples in wanted:
                summary = _evaluate_split(
                    model=model,
                    tokenizer=tokenizer,
                    examples=examples,
                    batch_size=args.eval_batch_size,
                    output_path=round_dir
                    / f"checkpoint_evaluation/step_{step:04d}/predictions"
                    / f"{name}.jsonl",
                )
                rows.append(
                    {
                        "cell_id": cell["cell_id"],
                        "round": args.round,
                        "step": step,
                        "adapter": str(adapter),
                        "dataset": name,
                        "count": summary["count"],
                        "exact_accuracy": summary["exact_accuracy"],
                        "format_accuracy": summary["format_accuracy"],
                    }
                )
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(
        round_dir / "checkpoint_evaluation/summary.json",
        {
            "cell": dict(cell),
            "round": args.round,
            "selection_sets": args.sets,
            "steps": [step for step, _adapter in checkpoints],
            "rows": rows,
        },
    )
    print(json.dumps({"cell": cell["cell_id"], "rows": len(rows)}, sort_keys=True))


def select_checkpoint(args: argparse.Namespace) -> None:
    """Record which checkpoint the next round should continue from.

    Selection reads validation cells only: choosing on the test cells and then
    reporting them inflates the reported number.  This is hyperparameter
    selection, not a curriculum promotion gate -- it never decides whether a
    round runs, only which weights it starts from.
    """

    cell = _cell(args)
    round_dir = _cell_round_dir(args.run_root, cell, args.round)
    available = dict(_checkpoint_adapters(round_dir))
    if not available:
        raise FileNotFoundError(f"No adapters to select from under {round_dir}")
    if args.step is not None:
        if args.step not in available:
            raise ValueError(f"Step {args.step} not among saved checkpoints {sorted(available)}")
        chosen, rule, scores = args.step, "manual", {}
    else:
        summary_path = round_dir / "checkpoint_evaluation/summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Run evaluate-checkpoints first: {summary_path} is missing"
            )
        rows = [
            row
            for row in _read_json(summary_path)["rows"]
            if str(row["dataset"]).startswith("validation_")
            and (args.select_on is None or str(row["dataset"]) == args.select_on)
        ]
        if not rows:
            raise ValueError(
                "No validation rows to select on; selecting on test cells is not permitted"
            )
        scores: Dict[str, float] = {}
        for step in sorted({int(row["step"]) for row in rows}):
            values = [float(row["exact_accuracy"]) for row in rows if int(row["step"]) == step]
            scores[str(step)] = sum(values) / len(values)
        chosen = int(max(scores, key=lambda step: (scores[step], -int(step))))
        rule = f"mean exact accuracy over {args.select_on or 'all validation cells'}"
    payload = {
        "cell": dict(cell),
        "round": args.round,
        "step": chosen,
        "adapter": str(available[chosen]),
        "rule": rule,
        "candidates": {str(step): str(path) for step, path in sorted(available.items())},
        "validation_scores": scores,
    }
    write_json(round_dir / SELECTION_FILE, payload)
    print(json.dumps({k: payload[k] for k in ("cell", "round", "step", "rule")}, sort_keys=True))


def evaluate_baselines(args: argparse.Namespace) -> None:
    """Seed-direct and frozen recursive composition, neither of which trains.

    Frozen composition answers each evaluation item by composing the seed
    model's predictions on that item's own components, so it uses the same
    decomposition and guard as the learned composition arms.
    """

    output_dir = args.run_root / "baselines"
    complete = output_dir / "BASELINES_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Baselines already evaluated: {output_dir}")
        return
    candidates_root = args.run_root / "data/evaluation/candidates"
    model = None
    try:
        model, tokenizer = load_adapter_for_evaluation(args.model_name, args.seed_adapter)
        seed_summaries = _evaluate_all(
            model=model,
            tokenizer=tokenizer,
            run_root=args.run_root,
            output_dir=output_dir / "seed/evaluation",
            batch_size=args.eval_batch_size,
        )
        frozen_summaries: Dict[str, Any] = {}
        for name, examples in _evaluation_sets(args.run_root):
            candidate_path = candidates_root / f"{name}.jsonl"
            if not candidate_path.exists():
                frozen_summaries[name] = {
                    "status": "not_decomposable",
                    "count": len(examples),
                }
                continue
            candidates = read_jsonl(candidate_path)
            specs = _unique_component_specs(candidates)
            largest = max(int(spec["expected_call_count"]) for spec in specs)
            raw = _raw_map_rows(
                _generate_rows(
                    model=model,
                    tokenizer=tokenizer,
                    examples=[public_spec_to_example(spec) for spec in specs],
                    identifiers=[str(spec["component_id"]) for spec in specs],
                    batch_size=args.eval_batch_size,
                    max_new_tokens=max(128, 64 * largest + 32),
                    id_key="component_id",
                ),
                "component_id",
            )
            predictions: List[str] = []
            decisions: List[Dict[str, Any]] = []
            for candidate in candidates:
                decision = compose_component_predictions(
                    candidate, raw, level=args.frozen_guard
                )
                predictions.append(decision["composed_target"] or "[]")
                decisions.append(decision)
            summary, evaluation_rows = evaluate_predictions(examples, predictions)
            frozen_summaries[name] = summary
            write_jsonl(
                output_dir / "frozen/evaluation/predictions" / f"{name}.jsonl",
                evaluation_rows,
            )
            write_jsonl(
                output_dir / "frozen/evaluation/decisions" / f"{name}.jsonl", decisions
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(
        output_dir / "summary.json",
        {
            "seed": {"adapter": str(args.seed_adapter), "evaluation": seed_summaries},
            "frozen": {
                "guard_level": args.frozen_guard,
                "adapter": str(args.seed_adapter),
                "evaluation": frozen_summaries,
            },
        },
    )
    complete.touch()
    print(json.dumps({"seed": seed_summaries, "frozen": frozen_summaries}, sort_keys=True))


def collect(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    completed_cells = 0
    grid = _run_grid(args.run_root)
    for cell in grid:
        cell_complete = True
        for round_index in (1, 2, 3):
            output_dir = _cell_round_dir(args.run_root, cell, round_index)
            metrics_path = output_dir / "metrics.json"
            if not metrics_path.exists():
                cell_complete = False
                failures.append(
                    {"cell_id": cell["cell_id"], "round": round_index, "missing": "metrics.json"}
                )
                continue
            metrics = _read_json(metrics_path)
            for dataset, summary in metrics["evaluation"].items():
                rows.append(
                    {
                        "cell_index": cell["cell_index"],
                        "size": cell["size"],
                        "condition": cell["condition"],
                        "round": round_index,
                        "dataset": dataset,
                        "count": summary["count"],
                        "exact_accuracy": summary["exact_accuracy"],
                        "format_accuracy": summary["format_accuracy"],
                        "behavior_valid_accuracy": summary["behavior_valid_accuracy"],
                    }
                )
        completed_cells += int(cell_complete)
    baselines_path = args.run_root / "baselines/summary.json"
    if baselines_path.exists():
        baselines = _read_json(baselines_path)
        for condition, payload in baselines.items():
            for dataset, entry in payload["evaluation"].items():
                if "exact_accuracy" not in entry:
                    continue  # e.g. natural cells the frozen arm cannot decompose
                rows.append(
                    {
                        "cell_index": -1,
                        "size": 0,
                        "condition": condition,
                        "round": 0,
                        "dataset": dataset,
                        "count": entry["count"],
                        "exact_accuracy": entry["exact_accuracy"],
                        "format_accuracy": entry["format_accuracy"],
                        "behavior_valid_accuracy": entry["behavior_valid_accuracy"],
                    }
                )
    else:
        failures.append({"cell_id": "baselines", "round": 0, "missing": "summary.json"})
    summary = {
        "experiment": "bfcl_cumulative_size_sweep",
        "completed_cells": completed_cells,
        "total_cells": len(grid),
        "partial": completed_cells != len(grid),
        "failures": failures,
        "metric_rows": rows,
    }
    write_json(args.run_root / "summary.json", summary)
    if rows:
        with (args.run_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.run_root / "checksums.json", _checksums(args.run_root / "cells", args.run_root))
    manifest = _read_json(args.run_root / "manifest.json")
    manifest["status"] = "complete" if not summary["partial"] else "partial"
    manifest["completed_cells"] = completed_cells
    manifest["collected_at_unix"] = time.time()
    write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _run(command: Sequence[str], *, dry_run: bool) -> str:
    print(f"[INFO] Command: {shlex.join(list(command))}", flush=True)
    if dry_run:
        return ""
    result = subprocess.run(
        list(command), cwd=ROOT_DIR, check=True, capture_output=True, text=True
    )
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr, flush=True)
    return result.stdout.strip()


def _submit_job(
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    name: str,
    dependency: Optional[str],
    gpu: bool,
    time_limit: str,
    array: bool = False,
) -> str:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition=ailab" if gpu else "--partition=cpu",
        "--cpus-per-task=8" if gpu else "--cpus-per-task=1",
        "--mem=64G" if gpu else "--mem=8G",
        f"--time={time_limit}",
        f"--job-name={name}",
        f"--output={args.log_dir / (name + ('-%A_%a.out' if array else '-%j.out'))}",
        f"--error={args.log_dir / (name + ('-%A_%a.err' if array else '-%j.err'))}",
    ]
    if gpu:
        sbatch.append("--gres=gpu:h200:1")
    if array:
        sbatch.append(f"--array=0-{len(_run_grid(args.run_root)) - 1}%4")
    if dependency:
        sbatch.append(f"--dependency=afterany:{dependency}")
    wrapped = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        *command,
    ]
    sbatch.append(f"--wrap={shlex.join(wrapped)}")
    output = _run(sbatch, dry_run=args.dry_run)
    return f"dryrun-{name}" if args.dry_run else output.split(";")[0].strip()


def _common(args: argparse.Namespace) -> List[str]:
    return [
        "--run-root", str(args.run_root),
        "--atomic-data-dir", str(args.atomic_data_dir),
        "--model-name", args.model_name,
        "--seed-adapter", str(args.seed_adapter),
        "--resume",
    ]


def _training_flags(args: argparse.Namespace) -> List[str]:
    flags: List[str] = []
    if getattr(args, "max_steps", None):
        flags += ["--max-steps", str(args.max_steps)]
    if getattr(args, "checkpoint_steps", ()):
        flags += ["--checkpoint-steps", ",".join(str(step) for step in args.checkpoint_steps)]
    return flags


# The DAG reaches later rounds through staged continuation jobs, so the budget
# has to travel with them: submit -> continue-submit -> ... -> train-evaluate.
# Dropping it anywhere in that chain silently reverts a round to one epoch.
_BUDGET_ACTIONS = ("train-evaluate", "continue-submit", "schedule-continuation")


def _command(args: argparse.Namespace, action: str, *extra: str) -> List[str]:
    return [
        str(args.python_bin),
        "-m", "self.experiments.bfcl_cumulative_size_sweep",
        action,
        *_common(args),
        *extra,
        *(_training_flags(args) if action in _BUDGET_ACTIONS else []),
    ]


STAGED_PHASES = ("r2-gen", "r2-train", "r3-gen", "r3-train", "collect")


def _stage_spec(args: argparse.Namespace, phase: str) -> Dict[str, Any]:
    specs = {
        "r2-gen": {
            "command": _command(args, "generate-materialize", "--round", "2"),
            "name": "bfcl-size-r2-gen",
            "gpu": True,
            "time": "02:00:00",
            "array": True,
        },
        "r2-train": {
            "command": _command(args, "train-evaluate", "--round", "2"),
            "name": "bfcl-size-r2-train",
            "gpu": True,
            "time": "02:00:00",
            "array": True,
        },
        "r3-gen": {
            "command": _command(args, "generate-materialize", "--round", "3"),
            "name": "bfcl-size-r3-gen",
            "gpu": True,
            "time": "02:00:00",
            "array": True,
        },
        "r3-train": {
            "command": _command(args, "train-evaluate", "--round", "3"),
            "name": "bfcl-size-r3-train",
            "gpu": True,
            "time": "02:00:00",
            "array": True,
        },
        "collect": {
            "command": _command(args, "collect"),
            "name": "bfcl-size-collect",
            "gpu": False,
            "time": "00:15:00",
            "array": False,
        },
    }
    if phase not in specs:
        raise ValueError(f"Unknown staged phase {phase!r}")
    return specs[phase]


def _submit_continuation(
    args: argparse.Namespace, *, phase: str, dependency: str
) -> str:
    return _submit_job(
        args=args,
        command=_command(args, "continue-submit", "--phase", phase),
        name=f"bfcl-size-stage-{phase}",
        dependency=dependency,
        gpu=False,
        time_limit="00:15:00",
    )


def continue_submit(args: argparse.Namespace) -> None:
    phase = args.phase
    spec = _stage_spec(args, phase)
    target = _submit_job(
        args=args,
        command=spec["command"],
        name=spec["name"],
        dependency=None,
        gpu=bool(spec["gpu"]),
        time_limit=str(spec["time"]),
        array=bool(spec["array"]),
    )
    phase_index = STAGED_PHASES.index(phase)
    next_phase = STAGED_PHASES[phase_index + 1] if phase_index + 1 < len(STAGED_PHASES) else None
    continuation = (
        _submit_continuation(args, phase=next_phase, dependency=target)
        if next_phase is not None
        else None
    )
    if not args.dry_run:
        manifest = _read_json(args.run_root / "manifest.json")
        manifest.setdefault("jobs", {}).setdefault("staged", {})[phase] = {
            "job_id": target,
            "continuation_job_id": continuation,
            "submitted_at_unix": time.time(),
        }
        manifest["status"] = "submitted"
        write_json(args.run_root / "manifest.json", manifest)
    print(
        json.dumps(
            {"phase": phase, "job_id": target, "next_phase": next_phase, "continuation": continuation},
            indent=2,
            sort_keys=True,
        )
    )


def schedule_continuation(args: argparse.Namespace) -> None:
    job_id = _submit_continuation(
        args, phase=args.phase, dependency=str(args.dependency)
    )
    if not args.dry_run:
        manifest = _read_json(args.run_root / "manifest.json")
        if args.round1_generation and args.round1_materialize and args.round1_training:
            manifest.setdefault("jobs", {})["round_01"] = {
                "shared_generation": str(args.round1_generation),
                "materialize": str(args.round1_materialize),
                "train_evaluate_array": str(args.round1_training),
            }
        manifest.setdefault("jobs", {})["continuation"] = {
            "phase": args.phase,
            "job_id": job_id,
            "dependency": str(args.dependency),
            "policy": "submit_next_array_after_previous_array_finishes",
        }
        manifest["status"] = "submitted"
        manifest["submitted_at_unix"] = time.time()
        write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps({"phase": args.phase, "job_id": job_id}, sort_keys=True))


def submit(args: argparse.Namespace) -> None:
    manifest = _read_json(args.run_root / "manifest.json")
    jobs: Dict[str, Any] = {}
    round1_generation = _submit_job(
        args=args,
        command=_command(args, "generate-round1-shared"),
        name="bfcl-size-r1-gen",
        dependency=None,
        gpu=True,
        time_limit="02:00:00",
    )
    baselines = _submit_job(
        args=args,
        command=_command(args, "evaluate-baselines"),
        name="bfcl-size-baselines",
        dependency=None,
        gpu=True,
        time_limit="02:00:00",
    )
    jobs["baselines"] = baselines
    round1_materialize = _submit_job(
        args=args,
        command=_command(args, "materialize-round1-shared"),
        name="bfcl-size-r1-mat",
        dependency=round1_generation,
        gpu=False,
        time_limit="00:15:00",
    )
    round1_training = _submit_job(
        args=args,
        command=_command(args, "train-evaluate", "--round", "1"),
        name="bfcl-size-r1-train",
        dependency=round1_materialize,
        gpu=True,
        time_limit="02:00:00",
        array=True,
    )
    jobs["round_01"] = {
        "shared_generation": round1_generation,
        "materialize": round1_materialize,
        "train_evaluate_array": round1_training,
    }
    continuation = _submit_continuation(
        args, phase="r2-gen", dependency=round1_training
    )
    jobs["continuation"] = {
        "phase": "r2-gen",
        "job_id": continuation,
        "policy": "submit_next_array_after_previous_array_finishes",
    }
    if not args.dry_run:
        manifest["jobs"] = jobs
        manifest["status"] = "submitted"
        manifest["submitted_at_unix"] = time.time()
        write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(jobs, indent=2, sort_keys=True), flush=True)


def _add_training_budget(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-steps",
        type=int,
        help=(
            "Optimizer updates per round.  Defaults to one epoch, which welds "
            "the update budget to the dataset size and confounds the two."
        ),
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=lambda value: tuple(int(item) for item in value.split(",") if item),
        default=(),
        help="Comma-separated steps at which to also save the adapter.",
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--atomic-data-dir", type=Path, default=DEFAULT_ATOMIC_DATA)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed-adapter", type=Path, default=DEFAULT_SEED_ADAPTER)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    _add_common(prepare_parser)
    prepare_parser.add_argument("--data-seed", type=int, default=EXPERIMENT_SEED)
    prepare_parser.add_argument(
        "--candidate-pool",
        type=int,
        default=CANDIDATE_POOL,
        help="Cross-family candidates built per regime; must exceed the largest quota.",
    )
    prepare_parser.add_argument(
        "--repeat-pool", type=int, default=REPEAT_POOL, help="Repeat-family candidates per regime."
    )
    prepare_parser.add_argument(
        "--learning-rates",
        type=lambda value: tuple(float(item) for item in value.split(",")),
        default=(LEARNING_RATE,),
        help="Comma-separated learning rates; one grid cell per rate.",
    )
    prepare_parser.add_argument(
        "--sizes",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        default=SIZES,
        help="Comma-separated examples-per-regime sizes; one grid column each.",
    )
    prepare_parser.add_argument(
        "--conditions",
        type=lambda value: tuple(value.split(",")),
        default=CONDITIONS,
        help=f"Comma-separated arms from {','.join(SUPPORTED_CONDITIONS)}.",
    )
    prepare_parser.add_argument(
        "--component-context",
        choices=COMPONENT_CONTEXT_MODES,
        default=DEFAULT_COMPONENT_CONTEXT,
        help=(
            "candidate_union shows each subproblem the parent's full shuffled "
            "schema union, so decomposition keeps the schema-selection problem."
        ),
    )
    generate1 = sub.add_parser("generate-round1-shared")
    _add_common(generate1)
    generate1.add_argument("--eval-batch-size", type=int, default=8)
    materialize1 = sub.add_parser("materialize-round1-shared")
    _add_common(materialize1)
    for command in ("generate-materialize", "train-evaluate"):
        child = sub.add_parser(command)
        _add_common(child)
        child.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
        child.add_argument("--cell-index", type=int)
        child.add_argument("--eval-batch-size", type=int, default=8)
        if command == "train-evaluate":
            child.add_argument("--micro-batch-size", type=int, default=2)
            _add_training_budget(child)
    checkpoints_parser = sub.add_parser("evaluate-checkpoints")
    _add_common(checkpoints_parser)
    checkpoints_parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    checkpoints_parser.add_argument("--cell-index", type=int)
    checkpoints_parser.add_argument("--eval-batch-size", type=int, default=8)
    checkpoints_parser.add_argument(
        "--sets",
        choices=("validation", "all"),
        default="validation",
        help=(
            "Which cells to score.  Selecting a checkpoint on the test cells "
            "and then reporting those cells inflates the reported number."
        ),
    )
    select_parser = sub.add_parser("select-checkpoint")
    _add_common(select_parser)
    select_parser.add_argument("--round", type=int, choices=(1, 2, 3), required=True)
    select_parser.add_argument("--cell-index", type=int)
    select_parser.add_argument(
        "--step", type=int, help="Select this step explicitly instead of scoring."
    )
    select_parser.add_argument(
        "--select-on",
        help="A single validation dataset name; default averages all of them.",
    )
    baselines_parser = sub.add_parser("evaluate-baselines")
    _add_common(baselines_parser)
    baselines_parser.add_argument("--eval-batch-size", type=int, default=8)
    baselines_parser.add_argument(
        "--frozen-guard",
        choices=("g1", "g4"),
        default="g1",
        help="Guard for frozen composition; g1 matches the primary condition.",
    )
    collect_parser = sub.add_parser("collect")
    _add_common(collect_parser)
    submit_parser = sub.add_parser("submit")
    _add_common(submit_parser)
    submit_parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    submit_parser.add_argument("--log-dir", type=Path)
    submit_parser.add_argument("--dry-run", action="store_true")
    _add_training_budget(submit_parser)
    continuation_parser = sub.add_parser("continue-submit")
    _add_common(continuation_parser)
    continuation_parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    continuation_parser.add_argument("--log-dir", type=Path)
    continuation_parser.add_argument("--dry-run", action="store_true")
    continuation_parser.add_argument("--phase", choices=STAGED_PHASES, required=True)
    _add_training_budget(continuation_parser)
    recovery_parser = sub.add_parser("schedule-continuation")
    _add_common(recovery_parser)
    recovery_parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    recovery_parser.add_argument("--log-dir", type=Path)
    recovery_parser.add_argument("--dry-run", action="store_true")
    recovery_parser.add_argument("--phase", choices=STAGED_PHASES, required=True)
    recovery_parser.add_argument("--dependency", required=True)
    recovery_parser.add_argument("--round1-generation")
    recovery_parser.add_argument("--round1-materialize")
    recovery_parser.add_argument("--round1-training")
    _add_training_budget(recovery_parser)
    return parser


def _normalize(args: argparse.Namespace) -> None:
    args.run_root = args.run_root.resolve()
    if getattr(args, "conditions", None):
        unknown = [item for item in args.conditions if item not in SUPPORTED_CONDITIONS]
        if unknown:
            raise ValueError(f"Unsupported conditions: {unknown}")
    args.atomic_data_dir = args.atomic_data_dir.resolve()
    args.seed_adapter = args.seed_adapter.resolve()
    if hasattr(args, "log_dir"):
        args.log_dir = (args.log_dir or args.run_root / "logs").resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _normalize(args)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate-round1-shared":
        generate_round1_shared(args)
    elif args.command == "materialize-round1-shared":
        materialize_round1_shared(args)
    elif args.command == "generate-materialize":
        generate_materialize(args)
    elif args.command == "train-evaluate":
        train_evaluate(args)
    elif args.command == "evaluate-checkpoints":
        evaluate_checkpoints(args)
    elif args.command == "select-checkpoint":
        select_checkpoint(args)
    elif args.command == "evaluate-baselines":
        evaluate_baselines(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "submit":
        submit(args)
    elif args.command == "continue-submit":
        continue_submit(args)
    elif args.command == "schedule-continuation":
        schedule_continuation(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

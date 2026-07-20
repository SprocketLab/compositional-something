#!/usr/bin/env python3
"""Persistent BFCL fixed-curriculum compositional self-improvement pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from self.coding.atomic_data import (
    AtomicExample,
    canonical_json,
    read_examples,
    stable_hash,
    write_examples,
    write_json,
)
from self.coding.bfcl_composition import (
    audit_decision,
    build_controlled_evaluation,
    build_round1_cross_candidates,
    build_round1_repeat_candidates,
    build_round2_cross_candidates,
    build_round2_repeat_candidates,
    candidate_counts,
    compose_component_predictions,
    guard_direct_prediction,
    guard_prediction,
    oracle_example,
    public_candidate_to_example,
    public_spec_to_example,
    read_jsonl,
    sha256_path,
    summarize_guard_audit,
    write_jsonl,
)
from self.coding.evaluation import evaluate_predictions
from self.coding.training import (
    adapter_parameter_summary,
    chat_prefix_ids,
    generate_predictions,
    load_adapter_for_evaluation,
    load_adapter_for_training,
    load_qwen_tokenizer,
    train_lora,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
ATOMIC_RUN_ROOT = ROOT_DIR / "artifacts/runs/coding_atomic_sweep_20260718_014707"
DEFAULT_ATOMIC_DATA = ATOMIC_RUN_ROOT / "data/bfcl"
DEFAULT_SEED_ADAPTER = ATOMIC_RUN_ROOT / "cells/bfcl/n240-s30-lr2em04-seed7/adapter"
DEFAULT_MODEL = Path(
    "/scratch/gpfs/BRENDEN/changho/hf_cache/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
DEFAULT_PYTHON = Path("/home/cs1095/.conda/envs/torch-env/bin/python")
EXPERIMENT_SEED = 20260719
TRAINING_SEED = 7
MAX_LENGTH = 1024
EFFECTIVE_BATCH_SIZE = 16
LEARNING_RATE = 2e-4
NEW_EXAMPLE_CAP = 1000
ROUND1_REPLAY_FRACTION = 0.40
ROUND2_MIX = {"new": 0.45, "previous": 0.25, "atomic": 0.30}
MAIN_CONDITIONS = ("direct_g4", "compose_g1", "compose_g4")
TRAINED_CONDITIONS = (*MAIN_CONDITIONS, "compose_g4_repeat20", "oracle")
PRIMARY_CONDITION = "compose_g1"


def _condition_dir(run_root: Path, round_index: int, condition: str) -> Path:
    return run_root / f"round_{round_index:02d}" / "conditions" / condition


def _candidate_paths(run_root: Path, round_index: int, family: str) -> Tuple[Path, Path]:
    stem = f"round_{round_index:02d}_{family}.jsonl"
    return (
        run_root / "data/public_candidates" / stem,
        run_root / "data/oracle" / stem,
    )


def _load_candidate_pair(
    run_root: Path,
    round_index: int,
    family: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    public_path, oracle_path = _candidate_paths(run_root, round_index, family)
    return read_jsonl(public_path), read_jsonl(oracle_path)


def _load_public_candidates(
    run_root: Path,
    round_index: int,
    family: str,
) -> List[Dict[str, Any]]:
    public_path, _oracle_path = _candidate_paths(run_root, round_index, family)
    return read_jsonl(public_path)


def _write_candidate_pair(
    run_root: Path,
    round_index: int,
    family: str,
    public_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
) -> None:
    public_path, oracle_path = _candidate_paths(run_root, round_index, family)
    write_jsonl(public_path, public_rows)
    write_jsonl(oracle_path, oracle_rows)


def _token_length(tokenizer: Any, candidate: Mapping[str, Any], oracle: Mapping[str, Any]) -> int:
    messages = candidate["messages"]
    target = canonical_json(oracle["canonical_calls"])
    return len(chat_prefix_ids(tokenizer, messages)) + len(tokenizer.encode(target, add_special_tokens=False)) + 1


def _filter_training_length(
    tokenizer: Any,
    public_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    oracle_by_id = {str(row["candidate_id"]): row for row in oracle_rows}
    kept_public: List[Dict[str, Any]] = []
    kept_oracle: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for candidate in public_rows:
        candidate_id = str(candidate["candidate_id"])
        oracle = oracle_by_id[candidate_id]
        length = _token_length(tokenizer, candidate, oracle)
        lengths.append(length)
        if length <= max_length:
            kept_public.append(copy.deepcopy(dict(candidate)))
            kept_oracle.append(copy.deepcopy(dict(oracle)))
        else:
            dropped.append({"candidate_id": candidate_id, "token_length": length})
    ordered = sorted(lengths)
    percentile = lambda fraction: ordered[round((len(ordered) - 1) * fraction)] if ordered else 0
    return kept_public, kept_oracle, {
        "input_count": len(public_rows),
        "kept_count": len(kept_public),
        "dropped_count": len(dropped),
        "dropped": dropped,
        "token_lengths": {
            "min": min(ordered, default=0),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": max(ordered, default=0),
        },
    }


def _write_data_checksums(run_root: Path) -> Dict[str, str]:
    checksums = {
        str(path.relative_to(run_root)): sha256_path(path)
        for path in sorted((run_root / "data").rglob("*"))
        if path.is_file()
    }
    write_json(run_root / "data_checksums.json", checksums)
    return checksums


def prepare(args: argparse.Namespace) -> None:
    run_root = args.run_root.resolve()
    data_root = args.atomic_data_dir.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    train = read_examples(data_root / "train.jsonl")
    hidden = read_examples(data_root / "hidden_composition.jsonl")
    validation = read_examples(data_root / "validation.jsonl")
    test = read_examples(data_root / "test.jsonl")
    if (len(train), len(hidden), len(validation), len(test)) != (240, 60, 40, 60):
        raise ValueError("The pilot requires the pinned 240/60/40/60 BFCL split")
    tokenizer = load_qwen_tokenizer(args.model_name)

    r1_cross_public, r1_cross_oracle = build_round1_cross_candidates(hidden, seed=args.data_seed)
    r1_cross_public, r1_cross_oracle, r1_cross_lengths = _filter_training_length(
        tokenizer, r1_cross_public, r1_cross_oracle, max_length=MAX_LENGTH
    )
    if len(r1_cross_public) != 1770:
        raise ValueError(f"Expected all 1770 pair candidates to fit, found {len(r1_cross_public)}")

    r2_cross_public, r2_cross_oracle = build_round2_cross_candidates(
        hidden, seed=args.data_seed, count=2200
    )
    r2_cross_public, r2_cross_oracle, r2_cross_lengths = _filter_training_length(
        tokenizer, r2_cross_public, r2_cross_oracle, max_length=MAX_LENGTH
    )
    if len(r2_cross_public) < 2000:
        raise ValueError(f"Only {len(r2_cross_public)} four-call candidates fit max_length={MAX_LENGTH}")
    r2_cross_public = r2_cross_public[:2000]
    allowed_r2 = {row["candidate_id"] for row in r2_cross_public}
    r2_cross_oracle = [row for row in r2_cross_oracle if row["candidate_id"] in allowed_r2]

    r1_repeat_public, r1_repeat_oracle, repeat_train_audit = build_round1_repeat_candidates(
        hidden,
        split="hidden_composition",
        seed=args.data_seed,
        max_variants_per_source=4,
        template_partition="train",
        renders_per_pair=2,
    )
    r1_repeat_public, r1_repeat_oracle, r1_repeat_lengths = _filter_training_length(
        tokenizer, r1_repeat_public, r1_repeat_oracle, max_length=MAX_LENGTH
    )
    repeat_train_audit["after_length_filter"] = len(r1_repeat_public)
    if repeat_train_audit["qualifying_source_count"] < 25:
        raise ValueError("Fewer than 25 hidden sources qualify for synthetic repetition")

    r2_repeat_public, r2_repeat_oracle = build_round2_repeat_candidates(
        r1_repeat_public,
        r1_repeat_oracle,
        seed=args.data_seed,
        count=600,
    )
    r2_repeat_public, r2_repeat_oracle, r2_repeat_lengths = _filter_training_length(
        tokenizer, r2_repeat_public, r2_repeat_oracle, max_length=MAX_LENGTH
    )
    if len(r2_repeat_public) < 200:
        raise ValueError("Fewer than 200 paired-repeat Round-2 candidates fit the context")

    _write_candidate_pair(run_root, 1, "cross", r1_cross_public, r1_cross_oracle)
    _write_candidate_pair(run_root, 1, "repeat", r1_repeat_public, r1_repeat_oracle)
    _write_candidate_pair(run_root, 2, "cross", r2_cross_public, r2_cross_oracle)
    _write_candidate_pair(run_root, 2, "repeat", r2_repeat_public, r2_repeat_oracle)

    evaluation_root = run_root / "data/evaluation"
    sets_root = evaluation_root / "sets"
    write_examples(sets_root / "atomic_test.jsonl", test)
    for source_name in ("natural_parallel", "natural_parallel_multiple"):
        write_examples(
            sets_root / f"{source_name}.jsonl",
            read_examples(data_root / "frontier" / f"{source_name}.jsonl"),
        )
    controlled = build_controlled_evaluation(test, seed=args.data_seed)
    for name, examples in controlled.items():
        write_examples(sets_root / f"{name}.jsonl", examples)

    repeat_eval_public, repeat_eval_oracle, repeat_eval_audit = build_round1_repeat_candidates(
        test,
        split="test",
        seed=args.data_seed,
        max_variants_per_source=1,
        template_partition="heldout",
        renders_per_pair=2,
    )
    if repeat_eval_audit["qualifying_source_count"] < 25:
        raise ValueError("Fewer than 25 test sources qualify for synthetic repetition")
    write_jsonl(run_root / "data/public_candidates/evaluation_repeat_2.jsonl", repeat_eval_public)
    write_jsonl(run_root / "data/oracle/evaluation_repeat_2.jsonl", repeat_eval_oracle)
    repeat_eval_oracle_by_id = {row["candidate_id"]: row for row in repeat_eval_oracle}
    write_examples(
        sets_root / "synthetic_repeat_2.jsonl",
        [oracle_example(row, repeat_eval_oracle_by_id[row["candidate_id"]]) for row in repeat_eval_public],
    )
    repeat4_public, repeat4_oracle = build_round2_repeat_candidates(
        repeat_eval_public,
        repeat_eval_oracle,
        seed=args.data_seed,
        count=100,
        split="test",
        template_partition="heldout",
    )
    repeat4_oracle_by_id = {row["candidate_id"]: row for row in repeat4_oracle}
    write_jsonl(run_root / "data/public_candidates/evaluation_repeat_4.jsonl", repeat4_public)
    write_jsonl(run_root / "data/oracle/evaluation_repeat_4.jsonl", repeat4_oracle)
    write_examples(
        sets_root / "synthetic_paired_repeat_4.jsonl",
        [oracle_example(row, repeat4_oracle_by_id[row["candidate_id"]]) for row in repeat4_public],
    )

    component_by_id: Dict[str, AtomicExample] = {example.source_id: example for example in test}
    for candidate in repeat_eval_public:
        for spec in candidate["component_specs"]:
            component_by_id.setdefault(str(spec["component_id"]), public_spec_to_example(spec, split="test"))
    write_examples(evaluation_root / "components.jsonl", list(component_by_id.values()))

    audit = {
        "atomic_counts": {"train": 240, "hidden_composition": 60, "validation": 40, "test": 60},
        "round_01_cross": {**candidate_counts(r1_cross_public), "length_filter": r1_cross_lengths},
        "round_02_cross": {**candidate_counts(r2_cross_public), "length_filter": r2_cross_lengths},
        "round_01_repeat": {
            **candidate_counts(r1_repeat_public),
            **repeat_train_audit,
            "length_filter": r1_repeat_lengths,
        },
        "round_02_repeat": {**candidate_counts(r2_repeat_public), "length_filter": r2_repeat_lengths},
        "repeat_evaluation": repeat_eval_audit,
        "evaluation_counts": {
            path.stem: len(read_examples(path)) for path in sorted(sets_root.glob("*.jsonl"))
        },
    }
    write_json(run_root / "data/audit.json", audit)
    manifest = {
        "experiment": "bfcl_compositional_pilot",
        "created_at_unix": time.time(),
        "model_name": args.model_name,
        "seed_adapter": str(args.seed_adapter.resolve()),
        "atomic_data_dir": str(data_root),
        "data_seed": args.data_seed,
        "training_seed": TRAINING_SEED,
        "max_length": MAX_LENGTH,
        "learning_rate": LEARNING_RATE,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "new_example_cap": NEW_EXAMPLE_CAP,
        "conditions": list(TRAINED_CONDITIONS),
        "primary_condition": PRIMARY_CONDITION,
        "curriculum_policy": "fixed_manual",
        "status": "prepared",
        "jobs": {},
    }
    write_json(run_root / "manifest.json", manifest)
    _write_data_checksums(run_root)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def _unique_component_specs(candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        for raw_spec in candidate["component_specs"]:
            spec = copy.deepcopy(dict(raw_spec))
            component_id = str(spec["component_id"])
            if component_id in by_id and by_id[component_id] != spec:
                raise ValueError(f"Component ID {component_id!r} maps to conflicting prompts")
            by_id[component_id] = spec
    return [by_id[key] for key in sorted(by_id)]


def _generate_rows(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[AtomicExample],
    identifiers: Sequence[str],
    batch_size: int,
    max_new_tokens: int,
    id_key: str,
) -> List[Dict[str, Any]]:
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    return [
        {id_key: identifier, "prediction": prediction}
        for identifier, prediction in zip(identifiers, predictions)
    ]


def generate_round1(args: argparse.Namespace) -> None:
    output_dir = args.run_root / "round_01/raw_predictions"
    complete = output_dir / "COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Round-1 raw predictions already complete: {output_dir}", flush=True)
        return
    cross_public = _load_public_candidates(args.run_root, 1, "cross")
    repeat_public = _load_public_candidates(args.run_root, 1, "repeat")
    specs = _unique_component_specs([*cross_public, *repeat_public])
    model, tokenizer = load_adapter_for_evaluation(args.model_name, args.seed_adapter)
    component_examples = [public_spec_to_example(spec) for spec in specs]
    component_rows = _generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=component_examples,
        identifiers=[str(spec["component_id"]) for spec in specs],
        batch_size=args.eval_batch_size,
        max_new_tokens=128,
        id_key="component_id",
    )
    write_jsonl(output_dir / "components.jsonl", component_rows)
    direct_examples = [public_candidate_to_example(row) for row in cross_public]
    direct_rows = _generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=direct_examples,
        identifiers=[str(row["candidate_id"]) for row in cross_public],
        batch_size=args.eval_batch_size,
        max_new_tokens=192,
        id_key="candidate_id",
    )
    write_jsonl(output_dir / "direct_g4.jsonl", direct_rows)
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.touch()
    print(
        json.dumps(
            {"component_predictions": len(component_rows), "direct_predictions": len(direct_rows)},
            sort_keys=True,
        ),
        flush=True,
    )


def generate_round2(args: argparse.Namespace) -> None:
    if args.condition not in {"direct_g4", "compose_g1", "compose_g4", "compose_g4_repeat20"}:
        raise ValueError(f"Round-2 generation is not defined for {args.condition!r}")
    output_dir = _condition_dir(args.run_root, 2, args.condition) / "raw_predictions"
    complete = output_dir / "COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Round-2 raw predictions already complete: {output_dir}", flush=True)
        return
    cross_public = _load_public_candidates(args.run_root, 2, "cross")
    repeat_public = _load_public_candidates(args.run_root, 2, "repeat")
    adapter_dir = _condition_dir(args.run_root, 1, args.condition) / "adapter"
    model, tokenizer = load_adapter_for_evaluation(args.model_name, adapter_dir)
    if args.condition == "direct_g4":
        examples = [public_candidate_to_example(row) for row in cross_public]
        rows = _generate_rows(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            identifiers=[str(row["candidate_id"]) for row in cross_public],
            batch_size=args.eval_batch_size,
            max_new_tokens=320,
            id_key="candidate_id",
        )
        write_jsonl(output_dir / "direct.jsonl", rows)
    else:
        candidates = cross_public
        if args.condition == "compose_g4_repeat20":
            candidates = [*cross_public, *repeat_public]
        specs = _unique_component_specs(candidates)
        examples = [public_spec_to_example(spec) for spec in specs]
        rows = _generate_rows(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            identifiers=[str(spec["component_id"]) for spec in specs],
            batch_size=args.eval_batch_size,
            max_new_tokens=192,
            id_key="component_id",
        )
        write_jsonl(output_dir / "components.jsonl", rows)
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.touch()
    print(json.dumps({"condition": args.condition, "output_dir": str(output_dir)}), flush=True)


def _raw_map(path: Path, id_key: str) -> Dict[str, str]:
    rows = read_jsonl(path)
    return {str(row[id_key]): str(row["prediction"]) for row in rows}


def _condition_decisions(
    condition: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    raw_components: Optional[Mapping[str, str]] = None,
    raw_direct: Optional[Mapping[str, str]] = None,
    oracle_by_id: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    decisions: List[Dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if condition == "direct_g4":
            if raw_direct is None or candidate_id not in raw_direct:
                raise ValueError(f"Missing direct prediction for {candidate_id}")
            decision = guard_direct_prediction(candidate, raw_direct[candidate_id])
        elif condition in {"compose_g1", "compose_g4", "compose_g4_repeat20"}:
            if raw_components is None:
                raise ValueError("Component predictions are required")
            level = "g1" if condition == "compose_g1" else "g4"
            decision = compose_component_predictions(candidate, raw_components, level=level)
        elif condition == "oracle":
            if oracle_by_id is None:
                raise ValueError("Oracle rows are required")
            oracle = oracle_by_id[candidate_id]
            decision = {
                "candidate_id": candidate_id,
                "accepted": True,
                "guard_level": "oracle",
                "reasons": [],
                "component_decisions": [],
                "composed_calls": copy.deepcopy(oracle["canonical_calls"]),
                "composed_target": canonical_json(oracle["canonical_calls"]),
            }
        else:
            raise ValueError(condition)
        decisions.append(decision)
    return decisions


def _persist_decisions(
    output_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    oracles: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    round_index: int,
    condition: str,
    prediction_checkpoint: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    oracle_by_id = {str(row["candidate_id"]): row for row in oracles}
    candidate_by_id = {str(row["candidate_id"]): row for row in candidates}
    audits = [
        audit_decision(candidate_by_id[str(row["candidate_id"])], oracle_by_id[str(row["candidate_id"])], row)
        for row in decisions
    ]
    summary = summarize_guard_audit(decisions, audits)
    write_jsonl(output_dir / "guard_decisions/all.jsonl", decisions)
    write_jsonl(
        output_dir / "parsed_predictions/all.jsonl",
        [
            {
                "candidate_id": row["candidate_id"],
                "accepted": row["accepted"],
                "parsed_calls": row.get("parsed_calls"),
                "composed_calls": row.get("composed_calls"),
                "component_decisions": row.get("component_decisions", []),
            }
            for row in decisions
        ],
    )
    write_jsonl(output_dir / "pseudo_label_audit/all.jsonl", audits)
    accepted_records: List[Dict[str, Any]] = []
    audit_by_id = {str(row["candidate_id"]): row for row in audits}
    for decision in decisions:
        if not decision["accepted"]:
            continue
        candidate_id = str(decision["candidate_id"])
        accepted_records.append(
            {
                "candidate_id": candidate_id,
                "round": round_index,
                "condition": condition,
                "prediction_checkpoint": prediction_checkpoint,
                "candidate": copy.deepcopy(candidate_by_id[candidate_id]),
                "decision": copy.deepcopy(dict(decision)),
                "audit": copy.deepcopy(audit_by_id[candidate_id]),
                "composed_target": decision["composed_target"],
            }
        )
    write_jsonl(output_dir / "composed_unique/all_accepted.jsonl", accepted_records)
    write_json(output_dir / "pseudo_label_audit/summary.json", summary)
    return summary, accepted_records


def _pseudo_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    round_index: int,
    condition: str,
) -> List[AtomicExample]:
    examples: List[AtomicExample] = []
    for record in records[:limit]:
        candidate = record["candidate"]
        example = public_candidate_to_example(candidate, target=str(record["composed_target"]))
        examples.append(
            AtomicExample(
                **{
                    **example.__dict__,
                    "metadata": {
                        **example.metadata,
                        "training_origin": "new_composed",
                        "condition": condition,
                        "round": round_index,
                    },
                }
            )
        )
    return examples


def _repeat_to_count(
    examples: Sequence[AtomicExample],
    count: int,
    *,
    seed: int,
    origin: str,
) -> List[AtomicExample]:
    if count <= 0:
        return []
    if not examples:
        raise ValueError(f"Cannot materialize {origin}: source dataset is empty")
    ordered = sorted(examples, key=lambda item: stable_hash(seed, origin, item.source_id))
    output: List[AtomicExample] = []
    for index in range(count):
        source = ordered[index % len(ordered)]
        output.append(
            AtomicExample(
                **{
                    **source.__dict__,
                    "metadata": {
                        **copy.deepcopy(source.metadata),
                        "training_origin": origin,
                        "replay_instance": index,
                    },
                }
            )
        )
    return output


def _materialize_training(
    *,
    run_root: Path,
    round_index: int,
    condition: str,
    selected_records: Sequence[Mapping[str, Any]],
    atomic_train: Sequence[AtomicExample],
    previous_new: Optional[Sequence[AtomicExample]] = None,
) -> Dict[str, Any]:
    output_dir = _condition_dir(run_root, round_index, condition)
    new_examples = _pseudo_examples(
        selected_records,
        limit=len(selected_records),
        round_index=round_index,
        condition=condition,
    )
    write_examples(output_dir / "composed_unique/selected_new.jsonl", new_examples)
    if round_index == 1:
        atomic_count = round(len(new_examples) * ROUND1_REPLAY_FRACTION / (1.0 - ROUND1_REPLAY_FRACTION))
        previous_count = 0
    else:
        atomic_count = round(len(new_examples) * ROUND2_MIX["atomic"] / ROUND2_MIX["new"])
        previous_count = round(len(new_examples) * ROUND2_MIX["previous"] / ROUND2_MIX["new"])
    atomic_replay = _repeat_to_count(
        atomic_train,
        atomic_count,
        seed=EXPERIMENT_SEED + round_index,
        origin="atomic_replay",
    )
    previous_replay = _repeat_to_count(
        list(previous_new or ()),
        previous_count,
        seed=EXPERIMENT_SEED + round_index,
        origin="previous_frontier_replay",
    ) if previous_count else []
    materialized = [*new_examples, *previous_replay, *atomic_replay]
    # Equivalent conditions must expose Trainer to the same example sequence.
    # Keep condition provenance in metadata, but never use it to order examples.
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
    selection_rows = [
        {
            "candidate_id": record["candidate_id"],
            "selection_index": index,
            "condition": condition,
            "round": round_index,
        }
        for index, record in enumerate(selected_records)
    ]
    write_jsonl(output_dir / "training_materialized/selection.jsonl", selection_rows)
    mix = {
        "new": len(new_examples),
        "previous_frontier_replay": len(previous_replay),
        "atomic_replay": len(atomic_replay),
        "total": len(materialized),
        "trainer_seed": TRAINING_SEED,
    }
    write_json(output_dir / "training_materialized/mix.json", mix)
    return mix


def _select_auxiliary_records(
    *,
    cross_records: Sequence[Mapping[str, Any]],
    repeat_records: Sequence[Mapping[str, Any]],
    total: int,
) -> Tuple[List[Mapping[str, Any]], Dict[str, int]]:
    repeat_target = round(total * 0.20)
    selected_repeat = list(repeat_records[:repeat_target])
    cross_target = total - len(selected_repeat)
    selected_cross = list(cross_records[:cross_target])
    if len(selected_cross) + len(selected_repeat) < total:
        additional_repeat = list(repeat_records[len(selected_repeat) : total - len(selected_cross)])
        selected_repeat.extend(additional_repeat)
    selected = [*selected_cross, *selected_repeat]
    selected.sort(key=lambda row: stable_hash(EXPERIMENT_SEED, "aux-selection", row["candidate_id"]))
    return selected, {"cross": len(selected_cross), "synthetic_repeat": len(selected_repeat)}


def _filter_record_lengths(
    records: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for record in records:
        length = (
            len(chat_prefix_ids(tokenizer, record["candidate"]["messages"]))
            + len(tokenizer.encode(str(record["composed_target"]), add_special_tokens=False))
            + 1
        )
        if length <= MAX_LENGTH:
            kept.append(copy.deepcopy(dict(record)))
        else:
            excluded.append(
                {
                    "candidate_id": record["candidate_id"],
                    "token_length": length,
                    "max_length": MAX_LENGTH,
                }
            )
    write_jsonl(output_dir / "training_materialized/length_excluded.jsonl", excluded)
    return kept


def materialize_round1(args: argparse.Namespace) -> None:
    complete = args.run_root / "round_01/MATERIALIZED"
    if complete.exists() and args.resume:
        print("[INFO] Round-1 datasets already materialized", flush=True)
        return
    cross_public, cross_oracle = _load_candidate_pair(args.run_root, 1, "cross")
    repeat_public, repeat_oracle = _load_candidate_pair(args.run_root, 1, "repeat")
    cross_oracle_by_id = {str(row["candidate_id"]): row for row in cross_oracle}
    component_raw = _raw_map(args.run_root / "round_01/raw_predictions/components.jsonl", "component_id")
    direct_raw = _raw_map(args.run_root / "round_01/raw_predictions/direct_g4.jsonl", "candidate_id")

    decisions_by_condition: Dict[str, List[Dict[str, Any]]] = {
        "direct_g4": _condition_decisions("direct_g4", cross_public, raw_direct=direct_raw),
        "compose_g1": _condition_decisions("compose_g1", cross_public, raw_components=component_raw),
        "compose_g4": _condition_decisions("compose_g4", cross_public, raw_components=component_raw),
        "oracle": _condition_decisions(
            "oracle", cross_public, oracle_by_id=cross_oracle_by_id
        ),
    }
    summaries: Dict[str, Any] = {}
    records: Dict[str, List[Dict[str, Any]]] = {}
    for condition, decisions in decisions_by_condition.items():
        summaries[condition], records[condition] = _persist_decisions(
            _condition_dir(args.run_root, 1, condition),
            cross_public,
            cross_oracle,
            decisions,
            round_index=1,
            condition=condition,
            prediction_checkpoint="hidden_oracle" if condition == "oracle" else str(args.seed_adapter),
        )

    repeat_decisions = _condition_decisions(
        "compose_g4_repeat20", repeat_public, raw_components=component_raw
    )
    aux_candidates = [*cross_public, *repeat_public]
    aux_oracles = [*cross_oracle, *repeat_oracle]
    aux_decisions = [*decisions_by_condition["compose_g4"], *repeat_decisions]
    summaries["compose_g4_repeat20"], records["compose_g4_repeat20"] = _persist_decisions(
        _condition_dir(args.run_root, 1, "compose_g4_repeat20"),
        aux_candidates,
        aux_oracles,
        aux_decisions,
        round_index=1,
        condition="compose_g4_repeat20",
        prediction_checkpoint=str(args.seed_adapter),
    )
    repeat_ids = {str(row["candidate_id"]) for row in repeat_public}
    tokenizer = load_qwen_tokenizer(args.model_name)
    for condition in records:
        records[condition] = _filter_record_lengths(
            records[condition],
            tokenizer=tokenizer,
            output_dir=_condition_dir(args.run_root, 1, condition),
        )
    aux_cross_records = [row for row in records["compose_g4_repeat20"] if row["candidate_id"] not in repeat_ids]
    aux_repeat_records = [row for row in records["compose_g4_repeat20"] if row["candidate_id"] in repeat_ids]

    matched_new_count = min(
        NEW_EXAMPLE_CAP,
        *(len(records[condition]) for condition in MAIN_CONDITIONS),
    )
    if matched_new_count <= 0:
        raise ValueError("No matched Round-1 pseudo-label budget is available")
    atomic_train = read_examples(args.atomic_data_dir / "train.jsonl")
    mixes: Dict[str, Any] = {}
    for condition in MAIN_CONDITIONS:
        selected = records[condition][:matched_new_count]
        mixes[condition] = _materialize_training(
            run_root=args.run_root,
            round_index=1,
            condition=condition,
            selected_records=selected,
            atomic_train=atomic_train,
        )
    oracle_selected = records["oracle"][:matched_new_count]
    mixes["oracle"] = _materialize_training(
        run_root=args.run_root,
        round_index=1,
        condition="oracle",
        selected_records=oracle_selected,
        atomic_train=atomic_train,
    )
    aux_selected, aux_quota = _select_auxiliary_records(
        cross_records=aux_cross_records,
        repeat_records=aux_repeat_records,
        total=matched_new_count,
    )
    if len(aux_selected) != matched_new_count:
        raise ValueError("Auxiliary condition cannot meet the matched Round-1 budget")
    mixes["compose_g4_repeat20"] = {
        **_materialize_training(
            run_root=args.run_root,
            round_index=1,
            condition="compose_g4_repeat20",
            selected_records=aux_selected,
            atomic_train=atomic_train,
        ),
        "new_example_families": aux_quota,
    }
    write_json(
        args.run_root / "round_01/materialization.json",
        {
            "matched_new_count": matched_new_count,
            "guard_summaries": summaries,
            "training_mixes": mixes,
        },
    )
    complete.touch()
    print(json.dumps({"matched_new_count": matched_new_count, "mixes": mixes}, sort_keys=True), flush=True)


def materialize_round2(args: argparse.Namespace) -> None:
    complete = args.run_root / "round_02/MATERIALIZED"
    if complete.exists() and args.resume:
        print("[INFO] Round-2 datasets already materialized", flush=True)
        return
    cross_public, cross_oracle = _load_candidate_pair(args.run_root, 2, "cross")
    repeat_public, repeat_oracle = _load_candidate_pair(args.run_root, 2, "repeat")
    cross_oracle_by_id = {str(row["candidate_id"]): row for row in cross_oracle}
    decisions_by_condition: Dict[str, List[Dict[str, Any]]] = {}
    for condition in MAIN_CONDITIONS:
        raw_root = _condition_dir(args.run_root, 2, condition) / "raw_predictions"
        if condition == "direct_g4":
            direct_raw = _raw_map(raw_root / "direct.jsonl", "candidate_id")
            decisions_by_condition[condition] = _condition_decisions(
                condition, cross_public, raw_direct=direct_raw
            )
        else:
            component_raw = _raw_map(raw_root / "components.jsonl", "component_id")
            decisions_by_condition[condition] = _condition_decisions(
                condition, cross_public, raw_components=component_raw
            )
    decisions_by_condition["oracle"] = _condition_decisions(
        "oracle", cross_public, oracle_by_id=cross_oracle_by_id
    )

    summaries: Dict[str, Any] = {}
    records: Dict[str, List[Dict[str, Any]]] = {}
    for condition, decisions in decisions_by_condition.items():
        summaries[condition], records[condition] = _persist_decisions(
            _condition_dir(args.run_root, 2, condition),
            cross_public,
            cross_oracle,
            decisions,
            round_index=2,
            condition=condition,
            prediction_checkpoint=(
                "hidden_oracle"
                if condition == "oracle"
                else str(_condition_dir(args.run_root, 1, condition) / "adapter")
            ),
        )

    aux_raw = _raw_map(
        _condition_dir(args.run_root, 2, "compose_g4_repeat20") / "raw_predictions/components.jsonl",
        "component_id",
    )
    aux_cross_decisions = _condition_decisions(
        "compose_g4_repeat20", cross_public, raw_components=aux_raw
    )
    aux_repeat_decisions = _condition_decisions(
        "compose_g4_repeat20", repeat_public, raw_components=aux_raw
    )
    summaries["compose_g4_repeat20"], records["compose_g4_repeat20"] = _persist_decisions(
        _condition_dir(args.run_root, 2, "compose_g4_repeat20"),
        [*cross_public, *repeat_public],
        [*cross_oracle, *repeat_oracle],
        [*aux_cross_decisions, *aux_repeat_decisions],
        round_index=2,
        condition="compose_g4_repeat20",
        prediction_checkpoint=str(
            _condition_dir(args.run_root, 1, "compose_g4_repeat20") / "adapter"
        ),
    )

    tokenizer = load_qwen_tokenizer(args.model_name)
    for condition in records:
        records[condition] = _filter_record_lengths(
            records[condition],
            tokenizer=tokenizer,
            output_dir=_condition_dir(args.run_root, 2, condition),
        )

    matched_new_count = min(
        NEW_EXAMPLE_CAP,
        *(len(records[condition]) for condition in MAIN_CONDITIONS),
    )
    if matched_new_count <= 0:
        raise ValueError("No matched Round-2 pseudo-label budget is available")
    atomic_train = read_examples(args.atomic_data_dir / "train.jsonl")
    mixes: Dict[str, Any] = {}
    for condition in (*MAIN_CONDITIONS, "oracle"):
        previous = read_examples(
            _condition_dir(args.run_root, 1, condition) / "composed_unique/selected_new.jsonl"
        )
        mixes[condition] = _materialize_training(
            run_root=args.run_root,
            round_index=2,
            condition=condition,
            selected_records=records[condition][:matched_new_count],
            atomic_train=atomic_train,
            previous_new=previous,
        )
    repeat_ids = {str(row["candidate_id"]) for row in repeat_public}
    aux_cross_records = [row for row in records["compose_g4_repeat20"] if row["candidate_id"] not in repeat_ids]
    aux_repeat_records = [row for row in records["compose_g4_repeat20"] if row["candidate_id"] in repeat_ids]
    aux_selected, aux_quota = _select_auxiliary_records(
        cross_records=aux_cross_records,
        repeat_records=aux_repeat_records,
        total=matched_new_count,
    )
    if len(aux_selected) != matched_new_count:
        raise ValueError("Auxiliary condition cannot meet the matched Round-2 budget")
    previous_aux = read_examples(
        _condition_dir(args.run_root, 1, "compose_g4_repeat20") / "composed_unique/selected_new.jsonl"
    )
    mixes["compose_g4_repeat20"] = {
        **_materialize_training(
            run_root=args.run_root,
            round_index=2,
            condition="compose_g4_repeat20",
            selected_records=aux_selected,
            atomic_train=atomic_train,
            previous_new=previous_aux,
        ),
        "new_example_families": aux_quota,
    }
    write_json(
        args.run_root / "round_02/materialization.json",
        {
            "matched_new_count": matched_new_count,
            "guard_summaries": summaries,
            "training_mixes": mixes,
        },
    )
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.touch()
    print(json.dumps({"matched_new_count": matched_new_count, "mixes": mixes}, sort_keys=True), flush=True)


def _stratified_metrics(
    examples: Sequence[AtomicExample],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_count: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for example, row in zip(examples, rows):
        by_count[int(example.component_count)].append(row)
    return {
        str(component_count): {
            "count": len(group),
            "exact_accuracy": sum(bool(row["exact"]) for row in group) / max(len(group), 1),
            "format_accuracy": sum(bool(row["format_valid"]) for row in group) / max(len(group), 1),
            "behavior_valid_accuracy": sum(bool(row["behavior_valid"]) for row in group) / max(len(group), 1),
        }
        for component_count, group in sorted(by_count.items())
    }


def _evaluate_split(
    *,
    model: Any,
    tokenizer: Any,
    examples: Sequence[AtomicExample],
    batch_size: int,
    output_path: Path,
) -> Dict[str, Any]:
    largest = max((example.component_count for example in examples), default=1)
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max(128, 64 * largest + 32),
    )
    summary, raw_rows = evaluate_predictions(examples, predictions)
    rows: List[Dict[str, Any]] = []
    for example, row in zip(examples, raw_rows):
        rows.append(
            {
                **row,
                "source_component_ids": list(example.source_component_ids),
                "evaluation_track": example.evaluation_track,
                "composition_family": example.metadata.get("composition_family"),
                "join_template": example.metadata.get("join_template"),
            }
        )
    write_jsonl(output_path, rows)
    summary["by_component_count"] = _stratified_metrics(examples, rows)
    summary["distinct_source_count"] = len(
        {source_id for example in examples for source_id in example.source_component_ids}
    )
    return summary


def _evaluation_sets(run_root: Path) -> List[Tuple[str, List[AtomicExample]]]:
    sets_root = run_root / "data/evaluation/sets"
    return [(path.stem, read_examples(path)) for path in sorted(sets_root.glob("*.jsonl"))]


def _evaluate_all(
    *,
    model: Any,
    tokenizer: Any,
    run_root: Path,
    output_dir: Path,
    batch_size: int,
) -> Dict[str, Any]:
    summaries: Dict[str, Any] = {}
    for name, examples in _evaluation_sets(run_root):
        summaries[name] = _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            batch_size=batch_size,
            output_path=output_dir / "predictions" / f"{name}.jsonl",
        )
    write_json(output_dir / "summary.json", summaries)
    return summaries


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def train_and_evaluate(args: argparse.Namespace) -> None:
    output_dir = _condition_dir(args.run_root, args.round, args.condition)
    complete = output_dir / "TRAIN_EVAL_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Train/eval already complete: round={args.round} condition={args.condition}")
        return
    train_examples = read_examples(output_dir / "training_materialized/train.jsonl")
    starting_adapter = (
        args.seed_adapter
        if args.round == 1
        else _condition_dir(args.run_root, 1, args.condition) / "adapter"
    )
    max_steps = math.ceil(len(train_examples) / EFFECTIVE_BATCH_SIZE)
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
                examples=train_examples,
                output_dir=output_dir,
                max_length=MAX_LENGTH,
                max_steps=max_steps,
                learning_rate=LEARNING_RATE,
                micro_batch_size=micro_batch,
                effective_batch_size=EFFECTIVE_BATCH_SIZE,
                seed=TRAINING_SEED,
            )
            evaluation = _evaluate_all(
                model=model,
                tokenizer=tokenizer,
                run_root=args.run_root,
                output_dir=output_dir / "evaluation",
                batch_size=args.eval_batch_size,
            )
            metrics = {
                "round": args.round,
                "condition": args.condition,
                "starting_adapter": str(starting_adapter),
                "adapter": str(output_dir / "adapter"),
                "training_example_count": len(train_examples),
                "max_steps": max_steps,
                "learning_rate": LEARNING_RATE,
                "max_length": MAX_LENGTH,
                "training": training,
                "evaluation": evaluation,
                "parameters": parameters,
                "oom_attempts": attempts,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            write_json(output_dir / "metrics.json", metrics)
            complete.touch()
            print(
                json.dumps(
                    {"round": args.round, "condition": args.condition, "evaluation": evaluation},
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        except RuntimeError as exc:
            if not _is_cuda_oom(exc) or micro_batch == 1:
                raise
            attempts.append({"micro_batch_size": micro_batch, "error": str(exc)})
            micro_batch //= 2
            print(f"[WARN] CUDA OOM; retrying with micro_batch_size={micro_batch}", flush=True)
        finally:
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _frozen_predictions(
    *,
    examples: Sequence[AtomicExample],
    raw_components: Mapping[str, str],
    component_by_id: Mapping[str, AtomicExample],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    predictions: List[str] = []
    decisions: List[Dict[str, Any]] = []
    for example in examples:
        calls: List[Dict[str, Any]] = []
        reasons: List[str] = []
        for source_id in example.source_component_ids:
            source = component_by_id.get(source_id)
            raw = raw_components.get(source_id)
            if source is None or raw is None:
                reasons.append(f"{source_id}:missing_component")
                continue
            spec = {
                "expected_call_count": 1,
                "functions": source.evaluator["functions"],
                "allow_exact_duplicates": False,
            }
            decision = guard_prediction(raw, spec, level="g4")
            if decision["accepted"]:
                calls.extend(decision["parsed_calls"])
            else:
                reasons.extend(f"{source_id}:{reason}" for reason in decision["reasons"])
        accepted = not reasons and len(calls) == example.component_count
        prediction = canonical_json(calls) if accepted else "[]"
        predictions.append(prediction)
        decisions.append(
            {
                "source_id": example.source_id,
                "accepted": accepted,
                "reasons": reasons,
                "prediction": prediction,
            }
        )
    return predictions, decisions


def evaluate_seed_and_frozen(args: argparse.Namespace) -> None:
    seed_dir = _condition_dir(args.run_root, 0, "seed_direct")
    frozen_dir = _condition_dir(args.run_root, 0, "frozen_recursive")
    complete = args.run_root / "round_00/SEED_FROZEN_COMPLETE"
    if complete.exists() and args.resume:
        print("[INFO] Seed/frozen evaluation already complete", flush=True)
        return
    model, tokenizer = load_adapter_for_evaluation(args.model_name, args.seed_adapter)
    seed_summaries = _evaluate_all(
        model=model,
        tokenizer=tokenizer,
        run_root=args.run_root,
        output_dir=seed_dir / "evaluation",
        batch_size=args.eval_batch_size,
    )
    components = read_examples(args.run_root / "data/evaluation/components.jsonl")
    component_rows = _generate_rows(
        model=model,
        tokenizer=tokenizer,
        examples=components,
        identifiers=[example.source_id for example in components],
        batch_size=args.eval_batch_size,
        max_new_tokens=128,
        id_key="component_id",
    )
    write_jsonl(frozen_dir / "raw_predictions/components.jsonl", component_rows)
    raw_components = {str(row["component_id"]): str(row["prediction"]) for row in component_rows}
    component_by_id = {example.source_id: example for example in components}
    frozen_summaries: Dict[str, Any] = {}
    for name, examples in _evaluation_sets(args.run_root):
        if name.startswith("natural_"):
            frozen_summaries[name] = {"status": "not_decomposable", "count": len(examples)}
            continue
        predictions, decisions = _frozen_predictions(
            examples=examples,
            raw_components=raw_components,
            component_by_id=component_by_id,
        )
        summary, rows = evaluate_predictions(examples, predictions)
        summary["by_component_count"] = _stratified_metrics(examples, rows)
        summary["distinct_source_count"] = len(
            {source_id for example in examples for source_id in example.source_component_ids}
        )
        frozen_summaries[name] = summary
        write_jsonl(frozen_dir / "evaluation/predictions" / f"{name}.jsonl", rows)
        write_jsonl(frozen_dir / "evaluation/decisions" / f"{name}.jsonl", decisions)
    write_json(seed_dir / "metrics.json", {"condition": "seed_direct", "evaluation": seed_summaries})
    write_json(frozen_dir / "evaluation/summary.json", frozen_summaries)
    write_json(frozen_dir / "metrics.json", {"condition": "frozen_recursive", "evaluation": frozen_summaries})
    complete.parent.mkdir(parents=True, exist_ok=True)
    complete.touch()
    print(json.dumps({"seed": seed_summaries, "frozen": frozen_summaries}, sort_keys=True), flush=True)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _all_artifact_checksums(run_root: Path) -> Dict[str, str]:
    roots = [run_root / "data", run_root / "round_00", run_root / "round_01", run_root / "round_02"]
    checksums: Dict[str, str] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                checksums[str(path.relative_to(run_root))] = sha256_path(path)
    return checksums


def collect(args: argparse.Namespace) -> None:
    gate_path = args.run_root / "round_01/gate.json"
    legacy_gate = _read_json(gate_path) if gate_path.exists() else None
    completed_round = 2 if all(
        (_condition_dir(args.run_root, 2, condition) / "metrics.json").exists()
        for condition in TRAINED_CONDITIONS
    ) else 1
    rows: List[Dict[str, Any]] = []
    for round_index in (0, 1, 2):
        conditions = ("seed_direct", "frozen_recursive") if round_index == 0 else TRAINED_CONDITIONS
        for condition in conditions:
            metrics_path = _condition_dir(args.run_root, round_index, condition) / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _read_json(metrics_path)
            evaluation = metrics.get("evaluation", {})
            for dataset, summary in evaluation.items():
                if "exact_accuracy" not in summary:
                    continue
                rows.append(
                    {
                        "round": round_index,
                        "condition": condition,
                        "dataset": dataset,
                        "count": summary["count"],
                        "exact_accuracy": summary["exact_accuracy"],
                        "format_accuracy": summary["format_accuracy"],
                        "behavior_valid_accuracy": summary["behavior_valid_accuracy"],
                        "distinct_source_count": summary.get("distinct_source_count", ""),
                    }
                )
    summary = {
        "completed_round": completed_round,
        "curriculum_policy": "fixed_manual",
        "primary_condition": PRIMARY_CONDITION,
        "metric_rows": rows,
        "run_root": str(args.run_root),
    }
    if legacy_gate is not None:
        summary["legacy_round1_gate"] = legacy_gate
    write_json(args.run_root / "summary.json", summary)
    if rows:
        with (args.run_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(args.run_root / "checksums.json", _all_artifact_checksums(args.run_root))
    manifest = _read_json(args.run_root / "manifest.json")
    manifest["status"] = "complete" if completed_round == 2 else manifest.get("status", "round_01_complete")
    manifest["completed_round"] = completed_round
    manifest["collected_at_unix"] = time.time()
    write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def audit_records(args: argparse.Namespace) -> None:
    condition_dir = _condition_dir(args.run_root, args.round, args.condition)
    decisions = read_jsonl(condition_dir / "guard_decisions/all.jsonl")
    audits = read_jsonl(condition_dir / "pseudo_label_audit/all.jsonl")
    accepted = read_jsonl(condition_dir / "composed_unique/all_accepted.jsonl")
    audit_by_id = {str(row["candidate_id"]): row for row in audits}
    accepted_by_id = {str(row["candidate_id"]): row for row in accepted}
    selected: List[Dict[str, Any]] = []
    for decision in decisions:
        candidate_id = str(decision["candidate_id"])
        audit = audit_by_id[candidate_id]
        include = (
            args.kind == "all"
            or (args.kind == "accepted" and decision["accepted"])
            or (args.kind == "rejected" and not decision["accepted"])
            or (args.kind == "false_accept" and audit["false_accept"])
            or (args.kind == "false_reject" and audit["false_reject"])
        )
        if include:
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "decision": decision,
                    "audit": audit,
                    "accepted_record": accepted_by_id.get(candidate_id),
                }
            )
        if len(selected) >= args.limit:
            break
    print(json.dumps(selected, indent=2, ensure_ascii=False, sort_keys=True))


def _run_command(command: Sequence[str], *, dry_run: bool) -> Optional[str]:
    print(f"[INFO] Command: {shlex.join(list(command))}", flush=True)
    if dry_run:
        return None
    completed = subprocess.run(
        list(command),
        cwd=ROOT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr, flush=True)
    return completed.stdout.strip()


def _submit_job(
    *,
    command: Sequence[str],
    job_name: str,
    log_dir: Path,
    dependencies: Sequence[str],
    gpu: bool,
    dry_run: bool,
) -> Optional[str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    sbatch = [
        "sbatch",
        "--parsable",
        "--partition=ailab" if gpu else "--partition=cpu",
        "--cpus-per-task=8" if gpu else "--cpus-per-task=1",
        "--mem=48G" if gpu else "--mem=8G",
        "--time=01:00:00" if gpu else "--time=00:15:00",
        f"--job-name={job_name}",
        f"--output={log_dir / (job_name + '-%j.out')}",
        f"--error={log_dir / (job_name + '-%j.err')}",
    ]
    if gpu:
        sbatch.append("--gres=gpu:h200:1")
    if dependencies:
        sbatch.append(f"--dependency=afterok:{':'.join(dependencies)}")
    wrapped = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        *command,
    ]
    sbatch.append(f"--wrap={shlex.join(wrapped)}")
    output = _run_command(sbatch, dry_run=dry_run)
    if dry_run:
        return f"dryrun-{job_name}"
    return None if output is None else output.split(";")[0].strip()


def _common_cli(args: argparse.Namespace) -> List[str]:
    return [
        "--run-root", str(args.run_root),
        "--atomic-data-dir", str(args.atomic_data_dir),
        "--model-name", args.model_name,
        "--seed-adapter", str(args.seed_adapter),
    ]


def _module_command(args: argparse.Namespace, command: str, *extra: str) -> List[str]:
    return [
        str(args.python_bin),
        "-m", "self.experiments.bfcl_compositional_pilot",
        command,
        *_common_cli(args),
        *extra,
        "--resume",
    ]


def _submit_in_waves(
    *,
    specs: Sequence[Tuple[str, Sequence[str]]],
    args: argparse.Namespace,
    initial_dependencies: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    submitted: List[Dict[str, Any]] = []
    prior = list(initial_dependencies)
    for start in range(0, len(specs), 4):
        wave: List[str] = []
        for job_name, command in specs[start : start + 4]:
            job_id = _submit_job(
                command=command,
                job_name=job_name,
                log_dir=args.log_dir,
                dependencies=prior,
                gpu=True,
                dry_run=args.dry_run,
            )
            submitted.append({"job_name": job_name, "job_id": job_id, "command": list(command)})
            if job_id:
                wave.append(job_id)
        prior = wave
    return submitted, prior


def submit(args: argparse.Namespace) -> None:
    manifest = _read_json(args.run_root / "manifest.json")
    generate_id = _submit_job(
        command=_module_command(args, "generate-round1"),
        job_name="bfcl-csi-r1-generate",
        log_dir=args.log_dir,
        dependencies=(),
        gpu=True,
        dry_run=args.dry_run,
    )
    materialize_id = _submit_job(
        command=_module_command(args, "materialize-round1"),
        job_name="bfcl-csi-r1-materialize",
        log_dir=args.log_dir,
        dependencies=[generate_id] if generate_id else (),
        gpu=False,
        dry_run=args.dry_run,
    )
    specs: List[Tuple[str, Sequence[str]]] = [
        (
            f"bfcl-csi-r1-{condition}"[:120],
            _module_command(args, "train-evaluate", "--round", "1", "--condition", condition),
        )
        for condition in TRAINED_CONDITIONS
    ]
    specs.append(("bfcl-csi-seed-frozen", _module_command(args, "evaluate-seed")))
    jobs, last_wave = _submit_in_waves(
        specs=specs,
        args=args,
        initial_dependencies=[materialize_id] if materialize_id else (),
    )
    round1_jobs = {
        "round_01_generate": generate_id,
        "round_01_materialize": materialize_id,
        "round_01_train_eval": jobs,
    }
    round2_jobs = _submit_round2(args, initial_dependencies=last_wave)
    jobs_payload = {"round_01": round1_jobs, "round_02": round2_jobs}
    if not args.dry_run:
        manifest["jobs"].update(jobs_payload)
        manifest["status"] = "fixed_curriculum_submitted"
        write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(jobs_payload, indent=2, sort_keys=True), flush=True)


def _submit_round2(
    args: argparse.Namespace,
    *,
    initial_dependencies: Sequence[str],
) -> Dict[str, Any]:
    generation_specs = [
        (
            f"bfcl-csi-r2-gen-{condition}"[:120],
            _module_command(args, "generate-round2", "--condition", condition),
        )
        for condition in ("direct_g4", "compose_g1", "compose_g4", "compose_g4_repeat20")
    ]
    generation_jobs, generation_last = _submit_in_waves(
        specs=generation_specs,
        args=args,
        initial_dependencies=initial_dependencies,
    )
    materialize_id = _submit_job(
        command=_module_command(args, "materialize-round2"),
        job_name="bfcl-csi-r2-materialize",
        log_dir=args.log_dir,
        dependencies=generation_last,
        gpu=False,
        dry_run=args.dry_run,
    )
    training_specs = [
        (
            f"bfcl-csi-r2-{condition}"[:120],
            _module_command(args, "train-evaluate", "--round", "2", "--condition", condition),
        )
        for condition in TRAINED_CONDITIONS
    ]
    training_jobs, training_last = _submit_in_waves(
        specs=training_specs,
        args=args,
        initial_dependencies=[materialize_id] if materialize_id else (),
    )
    collect_id = _submit_job(
        command=_module_command(args, "collect"),
        job_name="bfcl-csi-collect",
        log_dir=args.log_dir,
        dependencies=training_last,
        gpu=False,
        dry_run=args.dry_run,
    )
    return {
        "generation": generation_jobs,
        "materialize": materialize_id,
        "train_eval": training_jobs,
        "collect": collect_id,
    }


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--atomic-data-dir", type=Path, default=DEFAULT_ATOMIC_DATA)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed-adapter", type=Path, default=DEFAULT_SEED_ADAPTER)
    parser.add_argument("--resume", action="store_true")


def _add_scheduler(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    _add_common(prepare_parser)
    prepare_parser.add_argument("--data-seed", type=int, default=EXPERIMENT_SEED)

    generate1 = subparsers.add_parser("generate-round1")
    _add_common(generate1)
    generate1.add_argument("--eval-batch-size", type=int, default=16)

    generate2 = subparsers.add_parser("generate-round2")
    _add_common(generate2)
    generate2.add_argument(
        "--condition",
        choices=("direct_g4", "compose_g1", "compose_g4", "compose_g4_repeat20"),
        required=True,
    )
    generate2.add_argument("--eval-batch-size", type=int, default=16)

    materialize1 = subparsers.add_parser("materialize-round1")
    _add_common(materialize1)

    materialize2 = subparsers.add_parser("materialize-round2")
    _add_common(materialize2)

    train = subparsers.add_parser("train-evaluate")
    _add_common(train)
    train.add_argument("--round", type=int, choices=(1, 2), required=True)
    train.add_argument("--condition", choices=TRAINED_CONDITIONS, required=True)
    train.add_argument("--micro-batch-size", type=int, default=8)
    train.add_argument("--eval-batch-size", type=int, default=16)

    seed = subparsers.add_parser("evaluate-seed")
    _add_common(seed)
    seed.add_argument("--eval-batch-size", type=int, default=16)

    collect_parser = subparsers.add_parser("collect")
    _add_common(collect_parser)

    audit = subparsers.add_parser("audit")
    _add_common(audit)
    audit.add_argument("--round", type=int, choices=(1, 2), required=True)
    audit.add_argument("--condition", choices=TRAINED_CONDITIONS, required=True)
    audit.add_argument(
        "--kind",
        choices=("all", "accepted", "rejected", "false_accept", "false_reject"),
        default="false_accept",
    )
    audit.add_argument("--limit", type=int, default=10)

    submit_parser = subparsers.add_parser("submit")
    _add_common(submit_parser)
    _add_scheduler(submit_parser)
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    args.run_root = args.run_root.resolve()
    args.atomic_data_dir = args.atomic_data_dir.resolve()
    args.seed_adapter = args.seed_adapter.resolve()
    if hasattr(args, "log_dir"):
        args.log_dir = (args.log_dir or (args.run_root / "logs")).resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _normalize_args(args)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate-round1":
        generate_round1(args)
    elif args.command == "generate-round2":
        generate_round2(args)
    elif args.command == "materialize-round1":
        materialize_round1(args)
    elif args.command == "materialize-round2":
        materialize_round2(args)
    elif args.command == "train-evaluate":
        train_and_evaluate(args)
    elif args.command == "evaluate-seed":
        evaluate_seed_and_frozen(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "audit":
        audit_records(args)
    elif args.command == "submit":
        submit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

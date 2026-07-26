#!/usr/bin/env python3
"""Round-1 BFCL diagnostic for canonical versus model-aligned oracle targets."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
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
    guard_direct_prediction,
    public_candidate_to_example,
    public_spec_to_example,
    read_jsonl,
    write_jsonl,
)
from self.coding.evaluation import evaluate_bfcl
from self.coding.training import (
    adapter_parameter_summary,
    chat_prefix_ids,
    load_adapter_for_training,
    load_qwen_tokenizer,
    train_lora,
)
from self.experiments.bfcl_compositional_pilot import (
    DEFAULT_ATOMIC_DATA,
    DEFAULT_MODEL,
    DEFAULT_PYTHON,
    DEFAULT_SEED_ADAPTER,
    EFFECTIVE_BATCH_SIZE,
    EXPERIMENT_SEED,
    MAX_LENGTH,
    TRAINING_SEED,
    _evaluate_all,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PARENT_RUN = ROOT_DIR / "artifacts/runs/bfcl_compositional_pilot_20260720_124355"
TARGET_STYLES = ("canonical", "aligned")
LEARNING_RATES = (5e-5, 1e-4, 2e-4)
CHECKPOINT_STEPS = (20, 50, 100)
NEW_EXAMPLE_COUNT = 1000
ATOMIC_REPLAY_COUNT = 1000
DEFAULT_MAX_CONCURRENT = 4


@dataclass(frozen=True)
class SweepCell:
    index: int
    target_style: str
    learning_rate: float
    max_steps: int

    @property
    def cell_id(self) -> str:
        return (
            f"{self.target_style}-lr{_float_slug(self.learning_rate)}"
            f"-s{self.max_steps:03d}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "cell_id": self.cell_id,
            "target_style": self.target_style,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
        }


def _float_slug(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{'m' if int(exponent) < 0 else 'p'}{abs(int(exponent)):02d}"


def sweep_cells() -> List[SweepCell]:
    cells: List[SweepCell] = []
    for target_style in TARGET_STYLES:
        for learning_rate in LEARNING_RATES:
            for max_steps in CHECKPOINT_STEPS:
                cells.append(
                    SweepCell(
                        index=len(cells),
                        target_style=target_style,
                        learning_rate=learning_rate,
                        max_steps=max_steps,
                    )
                )
    return cells


def _raw_map(path: Path, id_key: str) -> Dict[str, str]:
    return {
        str(row[id_key]): str(row["prediction"])
        for row in read_jsonl(path)
    }


def _component_example(
    spec: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> AtomicExample:
    public = public_spec_to_example(spec)
    return AtomicExample(
        **{
            **public.__dict__,
            "target": canonical_json(oracle["canonical_calls"]),
            "evaluator": {
                "functions": copy.deepcopy(spec["functions"]),
                "accepted_calls": copy.deepcopy(oracle["accepted_calls"]),
            },
        }
    )


def build_aligned_oracle_decision(
    candidate: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    raw_components: Mapping[str, str],
    raw_direct: Mapping[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Choose an evaluation-exact target as close as possible to seed behavior.

    An exact end-to-end seed prediction has first priority, preserving both valid
    aliases and call order. Otherwise, each exact component prediction is kept
    and only incorrect components fall back to the canonical hidden reference.
    """

    candidate_id = str(candidate["candidate_id"])
    direct_decision = guard_direct_prediction(candidate, raw_direct[candidate_id])
    direct_audit = audit_decision(candidate, oracle, direct_decision)
    component_trace: List[Dict[str, Any]] = []
    if direct_audit["oracle_exact"]:
        calls = copy.deepcopy(direct_decision["composed_calls"])
        alignment_source = "direct_exact"
    else:
        alignment_source = "component_or_canonical"
        component_oracle_by_id = {
            str(row["component_id"]): row
            for row in oracle["component_oracles"]
        }
        calls: List[Dict[str, Any]] = []
        for spec in candidate["component_specs"]:
            component_id = str(spec["component_id"])
            component_oracle = component_oracle_by_id[component_id]
            prediction = raw_components[component_id]
            evaluation = evaluate_bfcl(
                _component_example(spec, component_oracle),
                prediction,
            )
            if evaluation.exact:
                selected_calls = copy.deepcopy(evaluation.parsed_prediction)
                source = "component_exact"
            else:
                selected_calls = copy.deepcopy(component_oracle["canonical_calls"])
                source = "canonical_fallback"
            calls.extend(selected_calls)
            component_trace.append(
                {
                    "component_id": component_id,
                    "source": source,
                    "seed_exact": bool(evaluation.exact),
                    "seed_error": evaluation.error,
                }
            )

    decision = {
        "candidate_id": candidate_id,
        "accepted": True,
        "guard_level": "aligned_oracle",
        "reasons": [],
        "component_decisions": component_trace,
        "composed_calls": calls,
        "composed_target": canonical_json(calls),
    }
    audit = audit_decision(candidate, oracle, decision)
    if not audit["oracle_exact"]:
        raise AssertionError(f"Aligned target is not oracle-exact: {candidate_id}")
    trace = {
        "candidate_id": candidate_id,
        "alignment_source": alignment_source,
        "direct_seed_exact": bool(direct_audit["oracle_exact"]),
        "canonical_target": canonical_json(oracle["canonical_calls"]),
        "aligned_target": decision["composed_target"],
        "differs_from_canonical": (
            decision["composed_target"] != canonical_json(oracle["canonical_calls"])
        ),
        "component_trace": component_trace,
    }
    return decision, trace


def _canonical_oracle_decision(
    candidate: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> Dict[str, Any]:
    calls = copy.deepcopy(oracle["canonical_calls"])
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "accepted": True,
        "guard_level": "canonical_oracle",
        "reasons": [],
        "component_decisions": [],
        "composed_calls": calls,
        "composed_target": canonical_json(calls),
    }


def _training_example(
    candidate: Mapping[str, Any],
    target: str,
    *,
    target_style: str,
) -> AtomicExample:
    public = public_candidate_to_example(candidate, target=target)
    return AtomicExample(
        **{
            **public.__dict__,
            "metadata": {
                **public.metadata,
                "training_origin": "new_composed",
                "target_style": target_style,
                "round": 1,
            },
        }
    )


def _atomic_replay(examples: Sequence[AtomicExample], count: int) -> List[AtomicExample]:
    ordered = sorted(
        examples,
        key=lambda item: stable_hash(EXPERIMENT_SEED, "aligned-oracle-atomic", item.source_id),
    )
    replay: List[AtomicExample] = []
    for index in range(count):
        source = ordered[index % len(ordered)]
        replay.append(
            AtomicExample(
                **{
                    **source.__dict__,
                    "metadata": {
                        **copy.deepcopy(source.metadata),
                        "training_origin": "atomic_replay",
                        "replay_instance": index,
                    },
                }
            )
        )
    return replay


def _ordered_training_mix(
    new_examples: Sequence[AtomicExample],
    replay: Sequence[AtomicExample],
) -> List[AtomicExample]:
    mixed = [*new_examples, *replay]
    mixed.sort(
        key=lambda item: stable_hash(
            EXPERIMENT_SEED,
            "aligned-oracle-training-order",
            item.metadata.get("training_origin", ""),
            item.source_id,
            item.metadata.get("replay_instance", -1),
        )
    )
    return mixed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _length_summary(tokenizer: Any, examples: Sequence[AtomicExample]) -> Dict[str, Any]:
    lengths = [
        len(chat_prefix_ids(tokenizer, example.messages))
        + len(tokenizer.encode(example.target, add_special_tokens=False))
        + 1
        for example in examples
    ]
    ordered = sorted(lengths)
    percentile = lambda q: ordered[round((len(ordered) - 1) * q)]
    return {
        "count": len(ordered),
        "min": min(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": max(ordered),
        "over_max_length": sum(length > MAX_LENGTH for length in ordered),
    }


def prepare(args: argparse.Namespace) -> None:
    complete = args.run_root / "PREPARED"
    if complete.exists() and args.resume:
        print(f"[INFO] Sweep already prepared: {args.run_root}", flush=True)
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    public = read_jsonl(args.parent_run_root / "data/public_candidates/round_01_cross.jsonl")
    oracles = read_jsonl(args.parent_run_root / "data/oracle/round_01_cross.jsonl")
    candidate_by_id = {str(row["candidate_id"]): row for row in public}
    oracle_by_id = {str(row["candidate_id"]): row for row in oracles}
    selection = read_jsonl(
        args.parent_run_root
        / "round_01/conditions/oracle/training_materialized/selection.jsonl"
    )
    selected_ids = [str(row["candidate_id"]) for row in selection[:NEW_EXAMPLE_COUNT]]
    if len(selected_ids) != NEW_EXAMPLE_COUNT or len(set(selected_ids)) != NEW_EXAMPLE_COUNT:
        raise ValueError("Expected 1,000 unique matched Round-1 Oracle candidate IDs")

    raw_components = _raw_map(
        args.parent_run_root / "round_01/raw_predictions/components.jsonl",
        "component_id",
    )
    raw_direct = _raw_map(
        args.parent_run_root / "round_01/raw_predictions/direct_g4.jsonl",
        "candidate_id",
    )
    atomic_train = read_examples(args.atomic_data_dir / "train.jsonl")
    replay = _atomic_replay(atomic_train, ATOMIC_REPLAY_COUNT)
    tokenizer = load_qwen_tokenizer(args.model_name)

    training_by_style: Dict[str, List[AtomicExample]] = {style: [] for style in TARGET_STYLES}
    decisions_by_style: Dict[str, List[Dict[str, Any]]] = {style: [] for style in TARGET_STYLES}
    audits_by_style: Dict[str, List[Dict[str, Any]]] = {style: [] for style in TARGET_STYLES}
    alignment_traces: List[Dict[str, Any]] = []
    for candidate_id in selected_ids:
        candidate = candidate_by_id[candidate_id]
        oracle = oracle_by_id[candidate_id]
        canonical = _canonical_oracle_decision(candidate, oracle)
        aligned, trace = build_aligned_oracle_decision(
            candidate,
            oracle,
            raw_components=raw_components,
            raw_direct=raw_direct,
        )
        for style, decision in (("canonical", canonical), ("aligned", aligned)):
            audit = audit_decision(candidate, oracle, decision)
            if not audit["oracle_exact"]:
                raise AssertionError(f"{style} target is not exact for {candidate_id}")
            decisions_by_style[style].append(decision)
            audits_by_style[style].append(audit)
            training_by_style[style].append(
                _training_example(
                    candidate,
                    str(decision["composed_target"]),
                    target_style=style,
                )
            )
        alignment_traces.append(trace)

    style_summaries: Dict[str, Any] = {}
    model_sequences: Dict[str, List[Tuple[Any, ...]]] = {}
    for style in TARGET_STYLES:
        style_dir = args.run_root / "targets" / style
        mixed = _ordered_training_mix(training_by_style[style], replay)
        write_examples(style_dir / "selected_new.jsonl", training_by_style[style])
        write_examples(style_dir / "training_materialized/train.jsonl", mixed)
        write_jsonl(style_dir / "decisions.jsonl", decisions_by_style[style])
        write_jsonl(style_dir / "oracle_audit.jsonl", audits_by_style[style])
        length_summary = _length_summary(tokenizer, mixed)
        if length_summary["over_max_length"]:
            raise ValueError(f"{style} contains examples longer than {MAX_LENGTH}")
        style_summaries[style] = {
            "new": len(training_by_style[style]),
            "atomic_replay": len(replay),
            "total": len(mixed),
            "oracle_exact": sum(bool(row["oracle_exact"]) for row in audits_by_style[style]),
            "lengths": length_summary,
            "train_sha256": _sha256(style_dir / "training_materialized/train.jsonl"),
        }
        model_sequences[style] = [
            (
                example.source_id,
                example.messages,
                example.metadata.get("training_origin"),
                example.metadata.get("replay_instance", -1),
            )
            for example in mixed
        ]
    if model_sequences["canonical"] != model_sequences["aligned"]:
        raise AssertionError("Canonical and aligned conditions do not share input ordering")

    write_jsonl(args.run_root / "alignment/trace.jsonl", alignment_traces)
    alignment_summary = {
        "count": len(alignment_traces),
        "direct_exact_count": sum(row["direct_seed_exact"] for row in alignment_traces),
        "component_or_canonical_count": sum(
            row["alignment_source"] == "component_or_canonical"
            for row in alignment_traces
        ),
        "differs_from_canonical_count": sum(
            row["differs_from_canonical"] for row in alignment_traces
        ),
        "component_exact_count": sum(
            component["source"] == "component_exact"
            for row in alignment_traces
            for component in row["component_trace"]
        ),
        "canonical_fallback_count": sum(
            component["source"] == "canonical_fallback"
            for row in alignment_traces
            for component in row["component_trace"]
        ),
    }
    grid = [cell.to_dict() for cell in sweep_cells()]
    write_json(args.run_root / "grid.json", grid)
    checksum_paths = sorted(
        [
            *(path for path in (args.run_root / "targets").rglob("*") if path.is_file()),
            *(path for path in (args.run_root / "alignment").rglob("*") if path.is_file()),
            args.run_root / "grid.json",
        ]
    )
    write_json(
        args.run_root / "data_checksums.json",
        {
            str(path.relative_to(args.run_root)): _sha256(path)
            for path in checksum_paths
        },
    )
    manifest = {
        "experiment": "bfcl_oracle_alignment_sweep",
        "status": "prepared",
        "created_at_unix": time.time(),
        "parent_run_root": str(args.parent_run_root),
        "atomic_data_dir": str(args.atomic_data_dir),
        "seed_adapter": str(args.seed_adapter),
        "model_name": args.model_name,
        "target_styles": list(TARGET_STYLES),
        "learning_rates": list(LEARNING_RATES),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "new_example_count": NEW_EXAMPLE_COUNT,
        "atomic_replay_count": ATOMIC_REPLAY_COUNT,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "training_seed": TRAINING_SEED,
        "alignment_summary": alignment_summary,
        "style_summaries": style_summaries,
        "grid": grid,
        "jobs": {},
    }
    write_json(args.run_root / "manifest.json", manifest)
    complete.touch()
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _cell_from_index(run_root: Path, index: int) -> SweepCell:
    grid = _read_json(run_root / "grid.json")
    try:
        row = grid[index]
    except IndexError as exc:
        raise ValueError(f"Cell index {index} is outside the sweep grid") from exc
    if int(row["index"]) != index:
        raise ValueError("Sweep grid index is inconsistent")
    return SweepCell(
        index=index,
        target_style=str(row["target_style"]),
        learning_rate=float(row["learning_rate"]),
        max_steps=int(row["max_steps"]),
    )


def train_evaluate(args: argparse.Namespace) -> None:
    cell = _cell_from_index(args.run_root, args.cell_index)
    output_dir = args.run_root / "cells" / cell.cell_id
    complete = output_dir / "TRAIN_EVAL_COMPLETE"
    if complete.exists() and args.resume:
        print(f"[INFO] Cell already complete: {cell.cell_id}", flush=True)
        return
    examples = read_examples(
        args.run_root / "targets" / cell.target_style / "training_materialized/train.jsonl"
    )
    model = None
    try:
        torch.manual_seed(TRAINING_SEED)
        torch.cuda.manual_seed_all(TRAINING_SEED)
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load_adapter_for_training(args.model_name, args.seed_adapter)
        parameters = adapter_parameter_summary(model)
        training = train_lora(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            output_dir=output_dir,
            max_length=MAX_LENGTH,
            max_steps=cell.max_steps,
            learning_rate=cell.learning_rate,
            micro_batch_size=args.micro_batch_size,
            effective_batch_size=EFFECTIVE_BATCH_SIZE,
            seed=TRAINING_SEED,
        )
        evaluation = _evaluate_all(
            model=model,
            tokenizer=tokenizer,
            run_root=args.parent_run_root,
            output_dir=output_dir / "evaluation",
            batch_size=args.eval_batch_size,
        )
        metrics = {
            **cell.to_dict(),
            "parent_run_root": str(args.parent_run_root),
            "starting_adapter": str(args.seed_adapter),
            "adapter": str(output_dir / "adapter"),
            "training_example_count": len(examples),
            "training_data": str(
                args.run_root
                / "targets"
                / cell.target_style
                / "training_materialized/train.jsonl"
            ),
            "training": training,
            "evaluation": evaluation,
            "parameters": parameters,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        }
        write_json(output_dir / "metrics.json", metrics)
        complete.touch()
        print(json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def collect(args: argparse.Namespace) -> None:
    rows: List[Dict[str, Any]] = []
    cells: List[Dict[str, Any]] = []
    missing: List[str] = []
    for cell in sweep_cells():
        path = args.run_root / "cells" / cell.cell_id / "metrics.json"
        if not path.exists():
            missing.append(cell.cell_id)
            continue
        metrics = _read_json(path)
        cells.append(metrics)
        for dataset, summary in metrics["evaluation"].items():
            if "exact_accuracy" not in summary:
                continue
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "target_style": cell.target_style,
                    "learning_rate": cell.learning_rate,
                    "max_steps": cell.max_steps,
                    "dataset": dataset,
                    "count": summary["count"],
                    "exact_accuracy": summary["exact_accuracy"],
                    "format_accuracy": summary["format_accuracy"],
                    "behavior_valid_accuracy": summary["behavior_valid_accuracy"],
                }
            )
    summary = {
        "experiment": "bfcl_oracle_alignment_sweep",
        "status": "complete" if not missing else "partial",
        "completed_cell_count": len(cells),
        "expected_cell_count": len(sweep_cells()),
        "missing_cells": missing,
        "cells": cells,
        "metric_rows": rows,
    }
    write_json(args.run_root / "summary.json", summary)
    with (args.run_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "cell_id",
            "target_style",
            "learning_rate",
            "max_steps",
            "dataset",
            "count",
            "exact_accuracy",
            "format_accuracy",
            "behavior_valid_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = _read_json(args.run_root / "manifest.json")
    manifest["status"] = summary["status"]
    manifest["collected_at_unix"] = time.time()
    manifest["completed_cell_count"] = len(cells)
    write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


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
    return completed.stdout.strip().split(";")[0]


def _common_cli(args: argparse.Namespace) -> List[str]:
    return [
        "--run-root", str(args.run_root),
        "--parent-run-root", str(args.parent_run_root),
        "--atomic-data-dir", str(args.atomic_data_dir),
        "--model-name", args.model_name,
        "--seed-adapter", str(args.seed_adapter),
    ]


def submit(args: argparse.Namespace) -> None:
    cells = sweep_cells()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    base_command = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        "TRANSFORMERS_OFFLINE=1",
        "HF_HUB_OFFLINE=1",
        str(args.python_bin),
        "-m", "self.experiments.bfcl_oracle_alignment_sweep",
        "train-evaluate",
        *_common_cli(args),
        "--resume",
    ]
    wrapped = shlex.join(base_command) + ' --cell-index="${SLURM_ARRAY_TASK_ID}"'
    array_command = [
        "sbatch",
        "--parsable",
        "--partition=ailab",
        "--cpus-per-task=8",
        "--mem=48G",
        "--time=00:45:00",
        "--gres=gpu:h200:1",
        "--job-name=bfcl-oracle-align",
        f"--array=0-{len(cells) - 1}%{args.max_concurrent}",
        f"--output={args.log_dir / 'bfcl-oracle-align-%A_%a.out'}",
        f"--error={args.log_dir / 'bfcl-oracle-align-%A_%a.err'}",
        f"--wrap={wrapped}",
    ]
    array_job = _run_command(array_command, dry_run=args.dry_run)
    if args.dry_run:
        array_job = "dryrun-array"

    collect_base = [
        "env",
        "TOKENIZERS_PARALLELISM=false",
        str(args.python_bin),
        "-m", "self.experiments.bfcl_oracle_alignment_sweep",
        "collect",
        *_common_cli(args),
    ]
    collect_command = [
        "sbatch",
        "--parsable",
        "--partition=cpu",
        "--cpus-per-task=1",
        "--mem=8G",
        "--time=00:15:00",
        "--job-name=bfcl-oracle-align-collect",
        f"--output={args.log_dir / 'bfcl-oracle-align-collect-%j.out'}",
        f"--error={args.log_dir / 'bfcl-oracle-align-collect-%j.err'}",
        f"--dependency=afterany:{array_job}",
        f"--wrap={shlex.join(collect_base)}",
    ]
    collect_job = _run_command(collect_command, dry_run=args.dry_run)
    if args.dry_run:
        collect_job = "dryrun-collect"
    jobs = {
        "array_job_id": array_job,
        "collect_job_id": collect_job,
        "array": f"0-{len(cells) - 1}%{args.max_concurrent}",
        "gpu_wall_time": "00:45:00",
        "cells": [cell.to_dict() for cell in cells],
    }
    if not args.dry_run:
        manifest = _read_json(args.run_root / "manifest.json")
        manifest["status"] = "submitted"
        manifest["jobs"] = jobs
        write_json(args.run_root / "manifest.json", manifest)
    print(json.dumps(jobs, indent=2, sort_keys=True), flush=True)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-run-root", type=Path, default=DEFAULT_PARENT_RUN)
    parser.add_argument("--atomic-data-dir", type=Path, default=DEFAULT_ATOMIC_DATA)
    parser.add_argument("--model-name", default=str(DEFAULT_MODEL))
    parser.add_argument("--seed-adapter", type=Path, default=DEFAULT_SEED_ADAPTER)
    parser.add_argument("--resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    _add_common(prepare_parser)

    train_parser = subparsers.add_parser("train-evaluate")
    _add_common(train_parser)
    train_parser.add_argument("--cell-index", type=int, required=True)
    train_parser.add_argument("--micro-batch-size", type=int, default=8)
    train_parser.add_argument("--eval-batch-size", type=int, default=16)

    collect_parser = subparsers.add_parser("collect")
    _add_common(collect_parser)

    submit_parser = subparsers.add_parser("submit")
    _add_common(submit_parser)
    submit_parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    submit_parser.add_argument("--log-dir", type=Path)
    submit_parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    submit_parser.add_argument("--dry-run", action="store_true")
    return parser


def _normalize_args(args: argparse.Namespace) -> None:
    args.run_root = args.run_root.resolve()
    args.parent_run_root = args.parent_run_root.resolve()
    args.atomic_data_dir = args.atomic_data_dir.resolve()
    args.seed_adapter = args.seed_adapter.resolve()
    if hasattr(args, "python_bin"):
        args.python_bin = args.python_bin.resolve()
        args.log_dir = (args.log_dir or (args.run_root / "logs")).resolve()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _normalize_args(args)
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train-evaluate":
        train_evaluate(args)
    elif args.command == "collect":
        collect(args)
    elif args.command == "submit":
        submit(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

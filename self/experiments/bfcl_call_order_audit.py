#!/usr/bin/env python3
"""CPU-only audit of clause/call/schema ordering in BFCL composed candidates.

Composed supervision is only valid as an instance of ``f*(x1 o x2) = f*(x1) <>
f*(x2)`` when input composition and output composition use the same
permutation.  The cumulative sweep rendered the joined request from a fresh
shuffle of the leaf clauses while concatenating the target calls per component,
so at four and eight calls the target was an unpredictable permutation of the
request.  This audit quantifies that defect on the persisted run artifacts,
measures the model's own ordering convention, and re-checks the same statistics
on candidates rebuilt with the corrected builders.

Nothing here touches a GPU or a checkpoint; it reads persisted artifacts only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from self.coding.atomic_data import read_examples
from self.coding.bfcl_composition import (
    build_hierarchical_cross_candidates,
    build_mutated_atomic_variants,
    build_round1_repeat_candidates,
    read_jsonl,
)
from self.coding.evaluation import parse_strict_json_array


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RUN_ROOT = ROOT_DIR / "artifacts/runs/bfcl_cumulative_size_sweep_20260721_132230"
DEFAULT_ATOMIC_DATA = ROOT_DIR / "artifacts/runs/coding_atomic_sweep_20260718_014707/data/bfcl"
DEFAULT_OUTPUT = ROOT_DIR / "reports/bfcl_call_order_audit"
DEFAULT_DATA_SEED = 20260721
CANDIDATE_FILES = (
    ("2", "cross"),
    ("4", "cross"),
    ("8", "cross"),
    ("2", "repeat"),
    ("4", "repeat"),
    ("8", "repeat"),
)


def _strip_terminal(text: str) -> str:
    return text.strip().rstrip(".!?; ")


def leaf_question_index(
    atomic_data_dir: Path,
    *,
    data_seed: int,
    max_variants_per_source: int = 4,
) -> Dict[str, str]:
    """Map every leaf source ID used by composition to its clause text.

    Repeat-family leaves are deterministic scalar mutations, so they are
    regenerated from the same pool and seed rather than read from disk.
    """

    index: Dict[str, str] = {}
    pool = [
        *read_examples(atomic_data_dir / "train.jsonl"),
        *read_examples(atomic_data_dir / "hidden_composition.jsonl"),
    ]
    for example in [*pool, *read_examples(atomic_data_dir / "test.jsonl")]:
        index[example.source_id] = str(example.metadata["question"])
    for example in pool:
        for variant in build_mutated_atomic_variants(
            example,
            split="hidden_composition",
            max_variants=max_variants_per_source,
            seed=data_seed,
        ):
            index[variant.source_id] = str(variant.metadata["question"])
    return index


def clause_units(
    candidate: Mapping[str, Any],
    question_by_source: Mapping[str, str],
) -> Optional[List[str]]:
    """Return one clause text per emitted call, in target-serialization order."""

    units: List[str] = []
    for spec in candidate["component_specs"]:
        clauses = spec.get("clause_questions")
        if clauses is None:
            sources = [str(source) for source in spec["source_component_ids"]]
            if not all(source in question_by_source for source in sources):
                return None
            clauses = [question_by_source[source] for source in sources]
        units.extend(str(clause) for clause in clauses)
    return units


def clause_offsets(question: str, units: Sequence[str]) -> Optional[List[int]]:
    offsets: List[int] = []
    for unit in units:
        offset = question.find(_strip_terminal(unit))
        if offset < 0:
            return None
        offsets.append(offset)
    return offsets


def _ranks(values: Sequence[int]) -> List[int]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _dedupe(names: Sequence[str]) -> List[str]:
    seen: List[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def candidate_order_report(
    candidate: Mapping[str, Any],
    oracle: Mapping[str, Any],
    question_by_source: Mapping[str, str],
) -> Optional[Dict[str, Any]]:
    """Compare target order, request clause order, and schema listing order."""

    units = clause_units(candidate, question_by_source)
    calls = oracle["canonical_calls"]
    if units is None or len(units) != len(calls):
        return None
    offsets = clause_offsets(str(candidate["question"]), units)
    if offsets is None:
        return None
    ranks = _ranks(offsets)
    clause_sequence = [calls[index]["name"] for index in sorted(range(len(calls)), key=lambda i: offsets[i])]
    schema_names = [str(function["name"]) for function in candidate["functions"]]
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "call_count": len(calls),
        "target_in_clause_order": ranks == list(range(len(ranks))),
        "displaced_call_fraction": sum(
            rank != index for index, rank in enumerate(ranks)
        ) / len(ranks),
        "schema_in_clause_order": schema_names == _dedupe(clause_sequence),
    }


def prediction_order_report(
    candidate: Mapping[str, Any],
    oracle: Mapping[str, Any],
    prediction: str,
    question_by_source: Mapping[str, str],
) -> Optional[bool]:
    """Did the model emit its calls in request-clause order?"""

    units = clause_units(candidate, question_by_source)
    calls = oracle["canonical_calls"]
    if units is None or len(units) != len(calls):
        return None
    offsets = clause_offsets(str(candidate["question"]), units)
    if offsets is None:
        return None
    offset_by_name: Dict[str, int] = {}
    for call, offset in zip(calls, offsets):
        name = str(call["name"])
        if name in offset_by_name:
            return None  # ambiguous: the same function answers several clauses
        offset_by_name[name] = offset
    parsed, _error = parse_strict_json_array(prediction)
    if parsed is None or len(parsed) != len(calls):
        return None
    predicted: List[int] = []
    for call in parsed:
        if not isinstance(call, dict) or str(call.get("name")) not in offset_by_name:
            return None
        predicted.append(offset_by_name[str(call["name"])])
    return predicted == sorted(predicted)


def _summarize(reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    resolved = [report for report in reports if report is not None]
    if not resolved:
        return {"resolved_count": 0}
    return {
        "resolved_count": len(resolved),
        "target_in_clause_order": sum(
            bool(report["target_in_clause_order"]) for report in resolved
        ) / len(resolved),
        "mean_displaced_call_fraction": sum(
            float(report["displaced_call_fraction"]) for report in resolved
        ) / len(resolved),
        "schema_in_clause_order": sum(
            bool(report["schema_in_clause_order"]) for report in resolved
        ) / len(resolved),
    }


def audit_persisted_candidates(
    run_root: Path,
    question_by_source: Mapping[str, str],
    *,
    sample: int,
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for calls, family in CANDIDATE_FILES:
        stem = f"calls_{calls}_{family}.jsonl"
        public_path = run_root / "data/public_candidates" / stem
        oracle_path = run_root / "data/oracle" / stem
        if not public_path.exists():
            continue
        public = read_jsonl(public_path)[:sample]
        oracle_by_id = {
            str(row["candidate_id"]): row for row in read_jsonl(oracle_path)
        }
        reports = [
            candidate_order_report(
                candidate, oracle_by_id[str(candidate["candidate_id"])], question_by_source
            )
            for candidate in public
        ]
        output[f"calls_{calls}_{family}"] = {
            "sampled_count": len(public),
            "unresolved_count": sum(report is None for report in reports),
            **_summarize(reports),
        }
    return output


def audit_model_order_convention(
    run_root: Path,
    question_by_source: Mapping[str, str],
    *,
    sample: int,
) -> Dict[str, Any]:
    """Measure the ordering convention in the model's own direct predictions."""

    sources: List[Tuple[str, int, Path]] = [
        ("seed", 2, run_root / "shared/round_01/cross/direct.jsonl"),
        (
            "round_02_direct_g4",
            4,
            run_root
            / "cells/n1000-direct_g4/round_02/regimes/calls_4/cross/raw_predictions/direct.jsonl",
        ),
        (
            "round_03_direct_g4",
            8,
            run_root
            / "cells/n1000-direct_g4/round_03/regimes/calls_8/cross/raw_predictions/direct.jsonl",
        ),
    ]
    output: Dict[str, Any] = {}
    for label, calls, path in sources:
        if not path.exists():
            continue
        stem = f"calls_{calls}_cross.jsonl"
        candidate_by_id = {
            str(row["candidate_id"]): row
            for row in read_jsonl(run_root / "data/public_candidates" / stem)
        }
        oracle_by_id = {
            str(row["candidate_id"]): row
            for row in read_jsonl(run_root / "data/oracle" / stem)
        }
        verdicts: List[bool] = []
        for row in read_jsonl(path)[:sample]:
            candidate_id = str(row["candidate_id"])
            if candidate_id not in candidate_by_id:
                continue
            verdict = prediction_order_report(
                candidate_by_id[candidate_id],
                oracle_by_id[candidate_id],
                str(row["prediction"]),
                question_by_source,
            )
            if verdict is not None:
                verdicts.append(verdict)
        if verdicts:
            output[label] = {
                "call_count": calls,
                "resolved_count": len(verdicts),
                "prediction_in_clause_order": sum(verdicts) / len(verdicts),
            }
    return output


def audit_training_targets(
    run_root: Path,
    question_by_source: Mapping[str, str],
    *,
    round_index: int,
    sample: int,
) -> Dict[str, Any]:
    """Fraction of materialized SFT targets whose calls follow clause order."""

    cells_root = run_root / "cells"
    if not cells_root.exists():
        return {}
    candidate_cache: Dict[str, Dict[str, Mapping[str, Any]]] = {}

    def _lookup(calls: int) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        if str(calls) not in candidate_cache:
            merged_public: Dict[str, Mapping[str, Any]] = {}
            merged_oracle: Dict[str, Mapping[str, Any]] = {}
            for family in ("cross", "repeat"):
                stem = f"calls_{calls}_{family}.jsonl"
                public_path = run_root / "data/public_candidates" / stem
                if not public_path.exists():
                    continue
                for row in read_jsonl(public_path):
                    merged_public[str(row["candidate_id"])] = row
                for row in read_jsonl(run_root / "data/oracle" / stem):
                    merged_oracle[str(row["candidate_id"])] = row
            candidate_cache[str(calls)] = merged_public
            candidate_cache[f"{calls}-oracle"] = merged_oracle
        return candidate_cache[str(calls)], candidate_cache[f"{calls}-oracle"]

    output: Dict[str, Any] = {}
    for cell_dir in sorted(cells_root.iterdir()):
        selected_root = cell_dir / f"round_{round_index:02d}/selected"
        if not selected_root.exists():
            continue
        per_regime: Dict[str, Any] = {}
        for path in sorted(selected_root.glob("calls_*.jsonl")):
            calls = int(path.stem.rsplit("_", 1)[1])
            public_by_id, oracle_by_id = _lookup(calls)
            verdicts: List[bool] = []
            for example in read_examples(path)[:sample]:
                candidate = public_by_id.get(example.source_id)
                oracle = oracle_by_id.get(example.source_id)
                if candidate is None or oracle is None:
                    continue
                verdict = prediction_order_report(
                    candidate, oracle, example.target, question_by_source
                )
                if verdict is not None:
                    verdicts.append(verdict)
            if verdicts:
                per_regime[f"calls_{calls}"] = {
                    "resolved_count": len(verdicts),
                    "target_in_clause_order": sum(verdicts) / len(verdicts),
                }
        if per_regime:
            output[cell_dir.name] = per_regime
    return output


def audit_rebuilt_candidates(
    atomic_data_dir: Path,
    *,
    data_seed: int,
    sample: int,
) -> Dict[str, Any]:
    """Re-run the same statistics on the corrected builders."""

    pool = [
        *read_examples(atomic_data_dir / "train.jsonl"),
        *read_examples(atomic_data_dir / "hidden_composition.jsonl"),
    ]
    question_by_source = {
        example.source_id: str(example.metadata["question"]) for example in pool
    }
    output: Dict[str, Any] = {}
    for calls in (2, 4, 8):
        public, oracle = build_hierarchical_cross_candidates(
            pool, component_count=calls, count=sample, seed=data_seed
        )
        reports = [
            candidate_order_report(candidate, hidden, question_by_source)
            for candidate, hidden in zip(public, oracle)
        ]
        output[f"calls_{calls}_cross"] = {
            "sampled_count": len(public),
            "unresolved_count": sum(report is None for report in reports),
            **_summarize(reports),
        }
    repeat_public, repeat_oracle, _audit = build_round1_repeat_candidates(
        pool,
        split="hidden_composition",
        seed=data_seed,
        max_variants_per_source=4,
        template_partition="train",
        renders_per_pair=2,
    )
    repeat_questions = dict(question_by_source)
    for example in pool:
        for variant in build_mutated_atomic_variants(
            example, split="hidden_composition", max_variants=4, seed=data_seed
        ):
            repeat_questions[variant.source_id] = str(variant.metadata["question"])
    reports = [
        candidate_order_report(candidate, hidden, repeat_questions)
        for candidate, hidden in zip(repeat_public[:sample], repeat_oracle[:sample])
    ]
    output["calls_2_repeat"] = {
        "sampled_count": len(reports),
        "unresolved_count": sum(report is None for report in reports),
        **_summarize(reports),
    }
    return output


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else "n/a"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# BFCL call-order audit",
        "",
        f"Run: `{report['run_root']}`",
        "",
        "Composed supervision is a valid instance of `f*(x1 o x2) = f*(x1) <> f*(x2)`",
        "only when input composition and output composition use the same permutation.",
        "The persisted sweep rendered the joined request from an independent shuffle of",
        "the leaf clauses while concatenating target calls per component, so clause `k`",
        "does not answer call `k` above two calls.",
        "",
        "## 1. Persisted candidates (as trained on)",
        "",
        "| candidate file | sampled | target in clause order | mean displaced calls | schema listed in clause order |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["persisted_candidates"].items():
        lines.append(
            f"| `{name}` | {row['sampled_count']} | {_percent(row.get('target_in_clause_order'))} "
            f"| {_percent(row.get('mean_displaced_call_fraction'))} "
            f"| {_percent(row.get('schema_in_clause_order'))} |"
        )
    lines += [
        "",
        "## 2. The model's own ordering convention",
        "",
        "| direct predictions | calls | resolved | emitted in clause order |",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["model_order_convention"].items():
        lines.append(
            f"| `{name}` | {row['call_count']} | {row['resolved_count']} "
            f"| {_percent(row['prediction_in_clause_order'])} |"
        )
    lines += [
        "",
        "The model emits calls in request order essentially always, so a target in any",
        "other order is unlearnable structure rather than a stylistic difference.",
        "",
        "## 3. Materialized training targets by condition",
        "",
        "| cell | regime | resolved | target in clause order |",
        "|---|---|---:|---:|",
    ]
    for cell, regimes in report["training_targets"].items():
        for regime, row in regimes.items():
            lines.append(
                f"| `{cell}` | {regime} | {row['resolved_count']} "
                f"| {_percent(row['target_in_clause_order'])} |"
            )
    lines += [
        "",
        "`direct_g4` targets are the model's own generations and stay in clause order;",
        "every composed and Oracle target is permuted at four and eight calls.  The",
        "confound therefore penalizes exactly the arms the study is about.",
        "",
        "## 4. Candidates rebuilt with the corrected builders",
        "",
        "| candidate file | sampled | target in clause order | mean displaced calls | schema listed in clause order |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in report["rebuilt_candidates"].items():
        lines.append(
            f"| `{name}` | {row['sampled_count']} | {_percent(row.get('target_in_clause_order'))} "
            f"| {_percent(row.get('mean_displaced_call_fraction'))} "
            f"| {_percent(row.get('schema_in_clause_order'))} |"
        )
    lines += [
        "",
        "Clause and call order now agree by construction, and schema listing order is",
        "an independent deterministic shuffle, so the positional shortcut documented in",
        "the plan's schema-permutation audit is removed at construction time.",
        "",
        "Chance agreement between schema order and clause order is `1/k!` for `k`",
        "distinct functions: 50% at two calls, 4.2% at four, 0.002% at eight.  The",
        "rebuilt columns sit at chance.  The repeat family reads 100% trivially because",
        "all of its calls share one function name, so its schema list has one entry.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m self.experiments.bfcl_call_order_audit --sample 200",
        "```",
        "",
        "## What this does and does not establish",
        "",
        "It establishes that every composed and Oracle training target above two calls",
        "asked for a permutation the model had no way to predict, and that the Direct",
        "arm was exempt.  It does not by itself prove the permutation caused the",
        "four/eight-call regressions; that requires retraining on corrected data with a",
        "fixed update budget.  Until then, no conclusion about the four- or eight-call",
        "frontier from the cumulative sweep should be treated as informative.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--atomic-data-dir", type=Path, default=DEFAULT_ATOMIC_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-seed", type=int, default=DEFAULT_DATA_SEED)
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--training-round", type=int, default=3)
    args = parser.parse_args(argv)

    question_by_source = leaf_question_index(
        args.atomic_data_dir, data_seed=args.data_seed
    )
    report = {
        "run_root": str(args.run_root),
        "atomic_data_dir": str(args.atomic_data_dir),
        "data_seed": args.data_seed,
        "sample": args.sample,
        "leaf_question_count": len(question_by_source),
        "persisted_candidates": audit_persisted_candidates(
            args.run_root, question_by_source, sample=args.sample
        ),
        "model_order_convention": audit_model_order_convention(
            args.run_root, question_by_source, sample=args.sample
        ),
        "training_targets": audit_training_targets(
            args.run_root,
            question_by_source,
            round_index=args.training_round,
            sample=args.sample,
        ),
        "rebuilt_candidates": audit_rebuilt_candidates(
            args.atomic_data_dir, data_seed=args.data_seed, sample=args.sample
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "call_order_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "call_order_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(
        {
            "persisted_candidates": report["persisted_candidates"],
            "model_order_convention": report["model_order_convention"],
            "rebuilt_candidates": report["rebuilt_candidates"],
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()

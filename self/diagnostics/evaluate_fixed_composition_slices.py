#!/usr/bin/env python3
"""Evaluate saved fixed-binary checkpoints on boundary-event slices.

This is an offline diagnostic: it does not retrain or regenerate pseudo-labels.
It builds targeted held-out examples for the fixed binary split and evaluates
each saved round model on the same slice set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from core.addition_pipeline import (
    ADDITION_WIDTH_EXACT_DIGITS,
    AdditionExample,
    compose_examples,
    example_key,
    generate_addition_pair,
    has_component_boundary_carry,
)
from self.core.data_io import sanitize_json_value
from self.core.evaluation import extract_numeric_answer, evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.recipe_models import (
    build_recipe_tokenizer,
    load_recipe_model,
)
from self.core.recipe_presets import (
    resolve_self_improvement_recipe,
)
from self.tasks.bit_parsing import (
    RUN_LENGTH_ALPHABET_SYMBOLS,
    parse_run_length_prediction,
)
from self.tasks.run_length_data import (
    RunLengthExample,
    run_length_key,
)
from self.tasks.run_length_logic import compute_run_stats


JsonDict = Dict[str, Any]


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "self").exists() and (candidate / "artifacts").exists():
            return candidate
    raise RuntimeError("Could not locate repository root.")


def load_json(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_result_rows(path: Path) -> List[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return list(payload)
    return list(payload.get("rounds", []))


def available_round_dirs(run_root: Path, requested: Optional[str]) -> List[Tuple[int, Path]]:
    selected: Optional[set[int]] = None
    if requested and requested != "all":
        selected = {int(part.strip()) for part in requested.split(",") if part.strip()}

    rounds: List[Tuple[int, Path]] = []
    for round_dir in sorted(run_root.glob("round_*")):
        if not round_dir.is_dir():
            continue
        suffix = round_dir.name.removeprefix("round_")
        if not suffix.isdigit():
            continue
        round_idx = int(suffix)
        if selected is not None and round_idx not in selected:
            continue
        if (round_dir / "model.safetensors").exists() or (round_dir / "pytorch_model.bin").exists():
            rounds.append((round_idx, round_dir))
    if not rounds:
        raise FileNotFoundError(f"No saved round model directories found under {run_root}")
    return rounds


def infer_final_size(run_root: Path, task: str) -> int:
    results_path = run_root / "self_improvement_results.json"
    if results_path.exists():
        rows = load_result_rows(results_path)
        if rows:
            last = rows[-1]
            for key in ("max_size", "max_bits", "max_digits"):
                if last.get(key) is not None:
                    return int(last[key])
    config = load_json(run_root / "config_args.json")
    if task == "addition":
        return int(config["initial_max_digits"]) + int(config["num_expand_rounds"]) * int(config["expand_num_digits"])
    frontier_min = config.get("frontier_min_bits")
    if frontier_min is None:
        frontier_min = int(config["initial_max_bits"]) + 1
    return int(frontier_min) + int(config["num_expand_rounds"]) * int(config["expand_num_bits"]) - 1


def addition_split_sizes(total_digits: int) -> Tuple[int, int]:
    left_digits = total_digits // 2
    right_digits = total_digits - left_digits
    if left_digits <= 0 or right_digits <= 0:
        raise ValueError(f"Cannot fixed-binary compose {total_digits} digits.")
    return left_digits, right_digits


def build_addition_fixed_binary_slice(
    *,
    min_digits: int,
    max_digits: int,
    per_size: int,
    want_middle_carry: bool,
    rng: random.Random,
) -> Tuple[List[AdditionExample], Dict[Tuple[int, int, int], List[Tuple[int, int, int]]]]:
    examples: List[AdditionExample] = []
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    occupied: set[Tuple[int, int, int]] = set()
    for digits in range(min_digits, max_digits + 1):
        left_digits, right_digits = addition_split_sizes(digits)
        attempts = 0
        count = 0
        while count < per_size and attempts < max(10_000, per_size * 2_000):
            attempts += 1
            left = generate_addition_pair(
                left_digits,
                allow_carry=True,
                rng=rng,
                addition_width_mode=ADDITION_WIDTH_EXACT_DIGITS,
            )
            right = generate_addition_pair(
                right_digits,
                allow_carry=True,
                rng=rng,
                addition_width_mode=ADDITION_WIDTH_EXACT_DIGITS,
            )
            composed = compose_examples(left, right)
            key = example_key(composed)
            if key in occupied:
                continue
            has_middle_carry = has_component_boundary_carry(composed, [left_digits, right_digits])
            if has_middle_carry != want_middle_carry:
                continue
            occupied.add(key)
            examples.append(composed)
            component_map[key] = [example_key(left), example_key(right)]
            count += 1
        if count < per_size:
            raise RuntimeError(
                f"Only generated {count}/{per_size} addition examples for digits={digits} "
                f"middle_carry={want_middle_carry}."
            )
    return examples, component_map


def run_length_example(bitstring: str, *, format_version: str, target_mode: str) -> RunLengthExample:
    max_run, prefix, suffix = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=len(bitstring),
        max_run=max_run,
        prefix_run=prefix,
        suffix_run=suffix,
        format_version=format_version,
        target_mode=target_mode,
    )


def random_string_without_long_run(
    length: int,
    *,
    alphabet: str,
    rng: random.Random,
    max_run_exclusive: int,
    force_first_not: Optional[str] = None,
    force_last_not: Optional[str] = None,
) -> str:
    for _ in range(20_000):
        value = "".join(rng.choice(alphabet) for _ in range(length))
        if force_first_not is not None and value and value[0] == force_first_not:
            continue
        if force_last_not is not None and value and value[-1] == force_last_not:
            continue
        max_run, _, _ = compute_run_stats(value)
        if max_run < max_run_exclusive:
            return value
    raise RuntimeError(f"Failed to sample a length-{length} string with max run < {max_run_exclusive}.")


def build_run_length_middle_critical_example(
    *,
    total_bits: int,
    alphabet: str,
    rng: random.Random,
    format_version: str,
    target_mode: str,
) -> Tuple[RunLengthExample, List[RunLengthExample]]:
    left_bits = total_bits // 2
    right_bits = total_bits - left_bits
    if left_bits < 3 or right_bits < 3:
        raise ValueError(f"Need both fixed-binary halves to have at least 3 symbols, got {left_bits}+{right_bits}.")

    boundary_symbol = rng.choice(alphabet)
    # A 2+2 continuation yields a boundary run of length 4; keep every child
    # below 4 so the middle run strictly determines the answer.
    left_body = random_string_without_long_run(
        left_bits - 2,
        alphabet=alphabet,
        rng=rng,
        max_run_exclusive=4,
        force_last_not=boundary_symbol,
    )
    right_body = random_string_without_long_run(
        right_bits - 2,
        alphabet=alphabet,
        rng=rng,
        max_run_exclusive=4,
        force_first_not=boundary_symbol,
    )
    left = run_length_example(left_body + boundary_symbol * 2, format_version=format_version, target_mode=target_mode)
    right = run_length_example(boundary_symbol * 2 + right_body, format_version=format_version, target_mode=target_mode)
    composed = run_length_example(left.bitstring + right.bitstring, format_version=format_version, target_mode=target_mode)
    if composed.max_run <= max(left.max_run, right.max_run):
        raise AssertionError("Constructed run-length example was not boundary-critical.")
    return composed, [left, right]


def build_run_length_no_continue_example(
    *,
    total_bits: int,
    alphabet: str,
    rng: random.Random,
    format_version: str,
    target_mode: str,
) -> Tuple[RunLengthExample, List[RunLengthExample]]:
    left_bits = total_bits // 2
    right_bits = total_bits - left_bits
    for _ in range(20_000):
        left = run_length_example(
            "".join(rng.choice(alphabet) for _ in range(left_bits)),
            format_version=format_version,
            target_mode=target_mode,
        )
        right = run_length_example(
            "".join(rng.choice(alphabet) for _ in range(right_bits)),
            format_version=format_version,
            target_mode=target_mode,
        )
        if left.bitstring[-1] != right.bitstring[0]:
            return (
                run_length_example(left.bitstring + right.bitstring, format_version=format_version, target_mode=target_mode),
                [left, right],
            )
    raise RuntimeError(f"Failed to sample no-continuation run-length example for bits={total_bits}.")


def build_run_length_fixed_binary_slice(
    *,
    min_bits: int,
    max_bits: int,
    per_size: int,
    slice_name: str,
    alphabet: str,
    rng: random.Random,
    format_version: str,
    target_mode: str,
) -> Tuple[List[RunLengthExample], Dict[Tuple[int, str], List[Tuple[int, str]]]]:
    examples: List[RunLengthExample] = []
    component_map: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
    occupied: set[Tuple[int, str]] = set()
    for bits in range(min_bits, max_bits + 1):
        count = 0
        attempts = 0
        while count < per_size and attempts < max(10_000, per_size * 2_000):
            attempts += 1
            if slice_name == "middle_continue_answer_relevant":
                example, children = build_run_length_middle_critical_example(
                    total_bits=bits,
                    alphabet=alphabet,
                    rng=rng,
                    format_version=format_version,
                    target_mode=target_mode,
                )
            elif slice_name == "no_middle_continue":
                example, children = build_run_length_no_continue_example(
                    total_bits=bits,
                    alphabet=alphabet,
                    rng=rng,
                    format_version=format_version,
                    target_mode=target_mode,
                )
            else:
                raise ValueError(f"Unsupported run-length slice {slice_name!r}.")
            key = run_length_key(example)
            if key in occupied:
                continue
            occupied.add(key)
            examples.append(example)
            component_map[key] = [run_length_key(child) for child in children]
            count += 1
        if count < per_size:
            raise RuntimeError(f"Only generated {count}/{per_size} run-length examples for bits={bits} slice={slice_name}.")
    return examples, component_map


def evaluate_slices(
    *,
    run_root: Path,
    recipe_name: str,
    slices: Dict[str, Sequence[Any]],
    task: str,
    batch_size: int,
    max_new_tokens: int,
    rounds: List[Tuple[int, Path]],
    bf16: bool,
    fp16: bool,
) -> List[JsonDict]:
    preset = resolve_self_improvement_recipe(recipe_name)
    tokenizer = build_recipe_tokenizer(preset)
    tokenizer.padding_side = "left"

    if task == "addition":
        parser = extract_numeric_answer
        size_getter = lambda example: int(example.digits)
    elif task == "run_length":
        parser = parse_run_length_prediction
        size_getter = lambda example: int(example.bits)
    else:
        raise ValueError(f"Unsupported task={task!r}")

    rows: List[JsonDict] = []
    for round_idx, round_dir in rounds:
        print(f"[INFO] Loading round {round_idx}: {round_dir}", flush=True)
        model = load_recipe_model(round_dir, tokenizer, bf16=bf16, fp16=fp16)
        round_payload: JsonDict = {
            "round": int(round_idx),
            "model_dir": str(round_dir),
            "slices": {},
        }
        for slice_name, examples in slices.items():
            decode_tokens = resolve_max_new_tokens(examples, max_new_tokens)
            accuracy, per_size = evaluate_accuracy_with_breakdown(
                model,
                tokenizer,
                examples,
                batch_size=batch_size,
                max_new_tokens=decode_tokens,
                size_getter=size_getter,
                prediction_parser=parser,
            )
            round_payload["slices"][slice_name] = {
                "accuracy": None if math.isnan(accuracy) else accuracy,
                "count": len(examples),
                "per_size_accuracy": {str(k): v for k, v in sorted(per_size.items())},
            }
            print(
                f"[INFO] round={round_idx} slice={slice_name} "
                f"accuracy={accuracy:.4f} count={len(examples)}",
                flush=True,
            )
        rows.append(round_payload)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def frontier_size_range_for_round(*, task: str, config: JsonDict, round_idx: int) -> Tuple[int, int]:
    """Return the newly introduced frontier band for a saved round model.

    Round 0 is the seed model, so we evaluate it on the first expansion frontier
    as a pre-self-improvement baseline. Round r>=1 is evaluated on the band that
    was introduced/trained during that round.
    """
    effective_round = max(1, int(round_idx))
    if task == "addition":
        initial_max = int(config.get("initial_max_digits", 7))
        step = int(config.get("expand_num_digits", config.get("expand_num_size", 1)))
        start = initial_max + (effective_round - 1) * step + 1
        return start, start + step - 1

    frontier_min = config.get("frontier_min_bits")
    if frontier_min is None:
        frontier_min = int(config.get("initial_max_bits", 0)) + 1
    step = int(config.get("expand_num_bits", config.get("expand_num_size", 1)))
    start = int(frontier_min) + (effective_round - 1) * step
    return start, start + step - 1


def evaluate_frontier_slices(
    *,
    run_root: Path,
    recipe_name: str,
    config: JsonDict,
    task: str,
    batch_size: int,
    max_new_tokens: int,
    rounds: List[Tuple[int, Path]],
    per_size: int,
    rng: random.Random,
    bf16: bool,
    fp16: bool,
    symbol_alphabet_size: int = 10,
    target_mode: str = "symbol_run_pair",
    format_version: str = "legacy",
) -> List[JsonDict]:
    preset = resolve_self_improvement_recipe(recipe_name)
    tokenizer = build_recipe_tokenizer(preset)
    tokenizer.padding_side = "left"

    if task == "addition":
        parser = extract_numeric_answer
        size_getter = lambda example: int(example.digits)
    elif task == "run_length":
        parser = parse_run_length_prediction
        size_getter = lambda example: int(example.bits)
        alphabet = RUN_LENGTH_ALPHABET_SYMBOLS[:symbol_alphabet_size]
    else:
        raise ValueError(f"Unsupported task={task!r}")

    rows: List[JsonDict] = []
    for round_idx, round_dir in rounds:
        min_size, max_size = frontier_size_range_for_round(task=task, config=config, round_idx=round_idx)
        if task == "addition":
            middle_carry, middle_carry_map = build_addition_fixed_binary_slice(
                min_digits=min_size,
                max_digits=max_size,
                per_size=per_size,
                want_middle_carry=True,
                rng=rng,
            )
            no_middle_carry, no_middle_carry_map = build_addition_fixed_binary_slice(
                min_digits=min_size,
                max_digits=max_size,
                per_size=per_size,
                want_middle_carry=False,
                rng=rng,
            )
            slices: Dict[str, Sequence[Any]] = {
                "middle_carry": middle_carry,
                "no_middle_carry": no_middle_carry,
            }
            component_counts = {
                "middle_carry": len(middle_carry_map),
                "no_middle_carry": len(no_middle_carry_map),
            }
        else:
            no_middle_continue, no_middle_map = build_run_length_fixed_binary_slice(
                min_bits=min_size,
                max_bits=max_size,
                per_size=per_size,
                slice_name="no_middle_continue",
                alphabet=alphabet,
                rng=rng,
                format_version=format_version,
                target_mode=target_mode,
            )
            middle_critical, middle_map = build_run_length_fixed_binary_slice(
                min_bits=min_size,
                max_bits=max_size,
                per_size=per_size,
                slice_name="middle_continue_answer_relevant",
                alphabet=alphabet,
                rng=rng,
                format_version=format_version,
                target_mode=target_mode,
            )
            slices = {
                "no_middle_continue": no_middle_continue,
                "middle_continue_answer_relevant": middle_critical,
            }
            component_counts = {
                "no_middle_continue": len(no_middle_map),
                "middle_continue_answer_relevant": len(middle_map),
            }

        print(
            f"[INFO] Loading round {round_idx}: {round_dir} "
            f"(frontier sizes {min_size}..{max_size})",
            flush=True,
        )
        model = load_recipe_model(round_dir, tokenizer, bf16=bf16, fp16=fp16)
        round_payload: JsonDict = {
            "round": int(round_idx),
            "model_dir": str(round_dir),
            "min_size": min_size,
            "max_size": max_size,
            "slice_counts": {name: len(examples) for name, examples in slices.items()},
            "component_counts": component_counts,
            "slices": {},
        }
        for slice_name, examples in slices.items():
            decode_tokens = resolve_max_new_tokens(examples, max_new_tokens)
            accuracy, per_size_breakdown = evaluate_accuracy_with_breakdown(
                model,
                tokenizer,
                examples,
                batch_size=batch_size,
                max_new_tokens=decode_tokens,
                size_getter=size_getter,
                prediction_parser=parser,
            )
            round_payload["slices"][slice_name] = {
                "accuracy": None if math.isnan(accuracy) else accuracy,
                "count": len(examples),
                "per_size_accuracy": {str(k): v for k, v in sorted(per_size_breakdown.items())},
            }
            print(
                f"[INFO] round={round_idx} sizes={min_size}..{max_size} "
                f"slice={slice_name} accuracy={accuracy:.4f} count={len(examples)}",
                flush=True,
            )
        rows.append(round_payload)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def write_outputs(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2)
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "min_size", "max_size", "slice", "accuracy", "count"])
        writer.writeheader()
        for row in payload["rounds"]:
            for slice_name, metrics in row["slices"].items():
                writer.writerow(
                    {
                        "round": row["round"],
                        "min_size": row.get("min_size", payload.get("min_size")),
                        "max_size": row.get("max_size", payload.get("max_size")),
                        "slice": slice_name,
                        "accuracy": metrics["accuracy"],
                        "count": metrics["count"],
                    }
                )
    print(f"[INFO] Wrote {path}", flush=True)
    print(f"[INFO] Wrote {csv_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("addition", "run_length"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=str, default="all")
    parser.add_argument("--per-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--max-size", type=int, default=None)
    parser.add_argument(
        "--frontier-only",
        action="store_true",
        help=(
            "Evaluate each round only on its newly introduced frontier band. "
            "Round 0 is evaluated on the first expansion frontier as a seed baseline."
        ),
    )
    parser.add_argument("--recipe", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--symbol-alphabet-size", type=int, default=None)
    parser.add_argument("--target-mode", type=str, default=None)
    parser.add_argument("--format-version", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = find_repo_root()
    run_root = (repo / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root.resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    config = load_json(run_root / "config_args.json") if (run_root / "config_args.json").exists() else {}
    recipe_name = args.recipe or str(config.get("recipe") or ("arithmetic_self_improve_v1" if args.task == "addition" else "algorithmic_self_improve_v1"))
    max_new_tokens = int(args.max_new_tokens or config.get("decode_max_new_tokens") or (48 if args.task == "addition" else 16))
    min_size = int(args.min_size or config.get("frontier_min_bits") or (int(config.get("initial_max_digits", config.get("initial_max_bits", 0))) + 1))
    if args.task == "addition" and args.min_size is None:
        min_size = int(config.get("initial_max_digits", 7)) + 1
    max_size = int(args.max_size or infer_final_size(run_root, args.task))
    rounds = available_round_dirs(run_root, args.rounds)
    rng = random.Random(args.seed)

    if args.frontier_only:
        symbol_alphabet_size = int(args.symbol_alphabet_size or config.get("symbol_alphabet_size") or 10)
        target_mode = str(args.target_mode or config.get("target_mode") or "symbol_run_pair")
        format_version = str(args.format_version or config.get("format_version") or "legacy")
        print(
            f"[INFO] Evaluating {args.task} fixed-binary frontier-only slices "
            f"on {len(rounds)} rounds, per_size={args.per_size}",
            flush=True,
        )
        round_rows = evaluate_frontier_slices(
            run_root=run_root,
            recipe_name=recipe_name,
            config=config,
            task=args.task,
            batch_size=args.batch_size,
            max_new_tokens=max_new_tokens,
            rounds=rounds,
            per_size=args.per_size,
            rng=rng,
            bf16=bool(args.bf16),
            fp16=bool(args.fp16),
            symbol_alphabet_size=symbol_alphabet_size,
            target_mode=target_mode,
            format_version=format_version,
        )
        payload = {
            "task": args.task,
            "run_root": str(run_root),
            "recipe": recipe_name,
            "frontier_only": True,
            "per_size": args.per_size,
            "rounds": round_rows,
        }
        output_path = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
        write_outputs(output_path, payload)
        return

    if args.task == "addition":
        middle_carry, middle_carry_map = build_addition_fixed_binary_slice(
            min_digits=min_size,
            max_digits=max_size,
            per_size=args.per_size,
            want_middle_carry=True,
            rng=rng,
        )
        no_middle_carry, no_middle_carry_map = build_addition_fixed_binary_slice(
            min_digits=min_size,
            max_digits=max_size,
            per_size=args.per_size,
            want_middle_carry=False,
            rng=rng,
        )
        slices: Dict[str, Sequence[Any]] = {
            "middle_carry": middle_carry,
            "no_middle_carry": no_middle_carry,
        }
        component_counts = {
            "middle_carry": len(middle_carry_map),
            "no_middle_carry": len(no_middle_carry_map),
        }
    else:
        symbol_alphabet_size = int(args.symbol_alphabet_size or config.get("symbol_alphabet_size") or 10)
        alphabet = RUN_LENGTH_ALPHABET_SYMBOLS[:symbol_alphabet_size]
        target_mode = str(args.target_mode or config.get("target_mode") or "symbol_run_pair")
        format_version = str(args.format_version or config.get("format_version") or "legacy")
        no_middle_continue, no_middle_map = build_run_length_fixed_binary_slice(
            min_bits=min_size,
            max_bits=max_size,
            per_size=args.per_size,
            slice_name="no_middle_continue",
            alphabet=alphabet,
            rng=rng,
            format_version=format_version,
            target_mode=target_mode,
        )
        middle_critical, middle_map = build_run_length_fixed_binary_slice(
            min_bits=min_size,
            max_bits=max_size,
            per_size=args.per_size,
            slice_name="middle_continue_answer_relevant",
            alphabet=alphabet,
            rng=rng,
            format_version=format_version,
            target_mode=target_mode,
        )
        slices = {
            "no_middle_continue": no_middle_continue,
            "middle_continue_answer_relevant": middle_critical,
        }
        component_counts = {
            "no_middle_continue": len(no_middle_map),
            "middle_continue_answer_relevant": len(middle_map),
        }

    print(
        f"[INFO] Evaluating {args.task} fixed-binary slices on {len(rounds)} rounds, "
        f"sizes={min_size}..{max_size}, per_size={args.per_size}",
        flush=True,
    )
    round_rows = evaluate_slices(
        run_root=run_root,
        recipe_name=recipe_name,
        slices=slices,
        task=args.task,
        batch_size=args.batch_size,
        max_new_tokens=max_new_tokens,
        rounds=rounds,
        bf16=bool(args.bf16),
        fp16=bool(args.fp16),
    )
    payload = {
        "task": args.task,
        "run_root": str(run_root),
        "recipe": recipe_name,
        "min_size": min_size,
        "max_size": max_size,
        "per_size": args.per_size,
        "slice_counts": {name: len(examples) for name, examples in slices.items()},
        "component_counts": component_counts,
        "rounds": round_rows,
    }
    output_path = (repo / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    write_outputs(output_path, payload)


if __name__ == "__main__":
    main()

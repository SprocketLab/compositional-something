"""Exact-pair composition and pseudolabel generation helpers."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.addition_pipeline import (
    AdditionExample,
    bucket_by_digits,
    compose_examples,
    example_key,
    has_component_boundary_carry,
)
from self.core.composition_pseudolabels import (
    compose_addition_pseudo_examples,
    compose_program_pseudo_examples,
    compose_pseudo_examples,
    compose_run_length_pseudo_examples,
    target_pattern_for_task,
)
from self.core.models import ExactPairDataset
from self.core.proposal_config_schema import ConfigProposal
from self.tasks.run_length_data import (
    RunLengthExample,
    bucket_run_length_by_bits,
    run_length_key,
)
from self.tasks.run_length_logic import compute_run_stats


def _passes_addition_guard(example: AdditionExample, component_digits: Sequence[int], guard: str) -> bool:
    if guard == "none":
        return True
    if guard == "reject_boundary_carry":
        return not has_component_boundary_carry(example, component_digits)
    raise ValueError(f"Unsupported addition guard={guard!r}.")


def build_exact_pair_addition_dataset(
    *,
    source_examples: Sequence[AdditionExample],
    proposal: ConfigProposal,
    per_size_count: int,
    rng: random.Random,
    exclude_keys: Optional[set[Tuple[int, int, int]]] = None,
    progress_name: str = "candidate",
    max_attempts: int = 10000,
) -> ExactPairDataset:
    if per_size_count <= 0:
        return ExactPairDataset([], {}, set(), {"requested": per_size_count, "retained": 0})
    buckets = bucket_by_digits(source_examples)
    if not buckets.get(proposal.left) or not buckets.get(proposal.right):
        raise ValueError(
            f"Missing source bucket for proposal {proposal.left}+{proposal.right}; "
            f"available={sorted(buckets)}."
        )

    occupied = set(exclude_keys or set())
    generated: List[AdditionExample] = []
    component_map: Dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
    new_keys: set[Tuple[int, int, int]] = set()
    attempts = 0
    duplicates_allowed = False
    rejected_by_guard = 0
    duplicate_count = 0
    while len(generated) < per_size_count:
        attempts += 1
        left_example = rng.choice(buckets[proposal.left])
        right_example = rng.choice(buckets[proposal.right])
        composed = compose_examples(left_example, right_example)
        component_digits = [proposal.left, proposal.right]
        if not _passes_addition_guard(composed, component_digits, proposal.guard):
            rejected_by_guard += 1
            if attempts < max_attempts:
                continue
            raise RuntimeError(
                f"Unable to sample addition examples satisfying guard={proposal.guard!r} "
                f"for pair {proposal.left}+{proposal.right} after {max_attempts} attempts."
            )
        key = example_key(composed)
        is_duplicate = key in occupied
        if is_duplicate and not duplicates_allowed:
            if attempts < max_attempts:
                continue
            print(
                f"[WARN] Exhausted unique {progress_name} addition sampling for target={proposal.target}; "
                "allowing duplicates.",
                flush=True,
            )
            duplicates_allowed = True
        if is_duplicate:
            duplicate_count += 1
        else:
            occupied.add(key)
            new_keys.add(key)
        generated.append(composed)
        component_map[key] = [example_key(left_example), example_key(right_example)]
        attempts = 0
    return ExactPairDataset(
        examples=generated,
        component_map=component_map,
        keys=new_keys,
        diagnostics={
            "task": "addition",
            "left": proposal.left,
            "right": proposal.right,
            "target": proposal.target,
            "guard": proposal.guard,
            "requested": per_size_count,
            "retained": len(generated),
            "rejected_by_guard": rejected_by_guard,
            "duplicate_count": duplicate_count,
        },
    )


def _passes_run_length_guard(left: RunLengthExample, right: RunLengthExample, guard: str) -> bool:
    if guard == "none":
        return True
    boundary_continue = bool(left.bitstring and right.bitstring and left.bitstring[-1] == right.bitstring[0])
    if guard == "reject_boundary_continue":
        return not boundary_continue
    if guard == "require_boundary_continue":
        return boundary_continue
    raise ValueError(f"Unsupported run_length guard={guard!r}.")


def merge_run_length_examples(left: RunLengthExample, right: RunLengthExample) -> RunLengthExample:
    bitstring = left.bitstring + right.bitstring
    max_run, prefix_run, suffix_run = compute_run_stats(bitstring)
    return RunLengthExample(
        bitstring=bitstring,
        bits=left.bits + right.bits,
        max_run=max_run,
        prefix_run=prefix_run,
        suffix_run=suffix_run,
        format_version=left.format_version,
        target_mode=left.target_mode,
    )


def build_exact_pair_run_length_dataset(
    *,
    source_examples: Sequence[RunLengthExample],
    proposal: ConfigProposal,
    per_size_count: int,
    rng: random.Random,
    exclude_keys: Optional[set[Tuple[int, str]]] = None,
    progress_name: str = "candidate",
    max_attempts: int = 10000,
) -> ExactPairDataset:
    if per_size_count <= 0:
        return ExactPairDataset([], {}, set(), {"requested": per_size_count, "retained": 0})
    buckets = bucket_run_length_by_bits(source_examples)
    if not buckets.get(proposal.left) or not buckets.get(proposal.right):
        raise ValueError(
            f"Missing source bucket for proposal {proposal.left}+{proposal.right}; "
            f"available={sorted(buckets)}."
        )

    occupied = set(exclude_keys or set())
    generated: List[RunLengthExample] = []
    component_map: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
    new_keys: set[Tuple[int, str]] = set()
    attempts = 0
    duplicates_allowed = False
    rejected_by_guard = 0
    duplicate_count = 0
    while len(generated) < per_size_count:
        attempts += 1
        left_example = rng.choice(buckets[proposal.left])
        right_example = rng.choice(buckets[proposal.right])
        if not _passes_run_length_guard(left_example, right_example, proposal.guard):
            rejected_by_guard += 1
            if attempts < max_attempts:
                continue
            raise RuntimeError(
                f"Unable to sample run_length examples satisfying guard={proposal.guard!r} "
                f"for pair {proposal.left}+{proposal.right} after {max_attempts} attempts."
            )
        composed = merge_run_length_examples(left_example, right_example)
        key = run_length_key(composed)
        is_duplicate = key in occupied
        if is_duplicate and not duplicates_allowed:
            if attempts < max_attempts:
                continue
            print(
                f"[WARN] Exhausted unique {progress_name} run-length sampling for target={proposal.target}; "
                "allowing duplicates.",
                flush=True,
            )
            duplicates_allowed = True
        if is_duplicate:
            duplicate_count += 1
        else:
            occupied.add(key)
            new_keys.add(key)
        generated.append(composed)
        component_map[key] = [run_length_key(left_example), run_length_key(right_example)]
        attempts = 0
    return ExactPairDataset(
        examples=generated,
        component_map=component_map,
        keys=new_keys,
        diagnostics={
            "task": "run_length",
            "left": proposal.left,
            "right": proposal.right,
            "target": proposal.target,
            "guard": proposal.guard,
            "requested": per_size_count,
            "retained": len(generated),
            "rejected_by_guard": rejected_by_guard,
            "duplicate_count": duplicate_count,
        },
    )


def build_exact_pair_dataset(
    *,
    task_name: str,
    source_examples: Sequence[Any],
    proposal: ConfigProposal,
    per_size_count: int,
    rng: random.Random,
    exclude_keys: Optional[set[Any]] = None,
    progress_name: str = "candidate",
) -> ExactPairDataset:
    if task_name == "addition":
        return build_exact_pair_addition_dataset(
            source_examples=source_examples,
            proposal=proposal,
            per_size_count=per_size_count,
            rng=rng,
            exclude_keys=exclude_keys,
            progress_name=progress_name,
        )
    if task_name == "run_length":
        return build_exact_pair_run_length_dataset(
            source_examples=source_examples,
            proposal=proposal,
            per_size_count=per_size_count,
            rng=rng,
            exclude_keys=exclude_keys,
            progress_name=progress_name,
        )
    raise ValueError(f"Unsupported task={task_name!r}.")

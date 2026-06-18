"""Task-specific property-test cases for sandboxed composition programs."""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

from self.core.program_sandbox_models import SandboxCase


def _run_length_state(bitstring: str) -> Tuple[int, str, int, str, int]:
    if not bitstring:
        return 0, "", 0, "", 0
    max_run = 1
    current = 1
    for prev, ch in zip(bitstring, bitstring[1:]):
        if prev == ch:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    prefix_symbol = bitstring[0]
    prefix_run = 0
    for ch in bitstring:
        if ch != prefix_symbol:
            break
        prefix_run += 1
    suffix_symbol = bitstring[-1]
    suffix_run = 0
    for ch in reversed(bitstring):
        if ch != suffix_symbol:
            break
        suffix_run += 1
    return max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run


def _format_run_length_state(bitstring: str) -> str:
    max_run, prefix_symbol, prefix_run, suffix_symbol, suffix_run = _run_length_state(bitstring)
    return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"


def build_run_length_program_cases(*, random_seed: int = 0, random_count: int = 8) -> List[SandboxCase]:
    cases: List[SandboxCase] = []

    def add_case(name: str, parts: Sequence[str]) -> None:
        components = [
            {
                "size": len(part),
                "input_id": f"{len(part)}:{part}",
                "prediction": _format_run_length_state(part),
                "metadata": {"part_index": index},
            }
            for index, part in enumerate(parts)
        ]
        joined = "".join(parts)
        cases.append(
            SandboxCase(
                name=name,
                components=components,
                metadata={"component_count": len(parts)},
                expected_accept=True,
                expected_target=_format_run_length_state(joined),
                target_pattern=r"\d+\|[0-9A-Z]\|\d+\|[0-9A-Z]\|\d+",
            )
        )

    add_case("same_boundary", ["0011", "11100"])
    add_case("different_boundary", ["0011", "22200"])
    add_case("all_same", ["111", "11"])
    add_case("multi_part", ["0100", "000", "2222"])
    cases.append(
        SandboxCase(
            name="empty_components",
            components=[],
            metadata={},
            expected_accept=False,
        )
    )
    cases.append(
        SandboxCase(
            name="malformed_prediction",
            components=[{"size": 3, "input_id": "bad", "prediction": "not|state", "metadata": {}}],
            metadata={},
            expected_accept=False,
        )
    )

    rng = random.Random(random_seed)
    alphabet = "012"
    for index in range(random_count):
        part_count = rng.randint(2, 4)
        parts = []
        for _ in range(part_count):
            size = rng.randint(1, 6)
            parts.append("".join(rng.choice(alphabet) for _ in range(size)))
        add_case(f"random_{index}", parts)
    return cases


def build_addition_program_cases() -> List[SandboxCase]:
    return [
        SandboxCase(
            name="concat_no_carry",
            components=[
                {"size": 2, "input_id": "2:12+34", "prediction": "46", "metadata": {}},
                {"size": 2, "input_id": "2:21+43", "prediction": "64", "metadata": {}},
            ],
            metadata={"component_count": 2},
            expected_accept=True,
            expected_target="4664",
            target_pattern=r"-?\d+",
        ),
        SandboxCase(
            name="malformed_prediction",
            components=[{"size": 2, "input_id": "bad", "prediction": "x", "metadata": {}}],
            metadata={},
            expected_accept=False,
        ),
    ]

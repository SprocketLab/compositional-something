#!/usr/bin/env python3
"""Multiplication example container, keys, and compatibility reexports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from self.tasks.bit_parsing import format_multiplication_target


MultiplicationKey = Tuple[int, int, int]


@dataclass(frozen=True)
class MultiplicationExample:
    a: int
    b: int
    digits: int
    result: int
    operand_width: int
    format_version: str = "legacy"
    target_override: Optional[str] = None

    def prompt(self) -> str:
        if self.format_version == "symbolic_v1":
            return f"{self.a:0{self.operand_width}d}×{self.b:0{self.operand_width}d}="
        return f"Q: {self.a:0{self.operand_width}d} * {self.b:0{self.operand_width}d} = ?\nA:"

    def target(self) -> str:
        if self.target_override is not None:
            return self.target_override
        return format_multiplication_target(self.result, self.digits, self.format_version)

    def target_prefix(self) -> str:
        return "" if self.format_version == "symbolic_v1" else " "


def multiplication_key(example: MultiplicationExample) -> MultiplicationKey:
    return example.digits, example.a, example.b


def encode_multiplication_key(key: MultiplicationKey) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def decode_multiplication_key(value: str) -> MultiplicationKey:
    digits, a, b = value.split("|", 2)
    return int(digits), int(a), int(b)


def clone_multiplication_with_override(
    example: MultiplicationExample,
    override: Optional[str],
) -> MultiplicationExample:
    if override is None:
        return example
    return MultiplicationExample(
        a=example.a,
        b=example.b,
        digits=example.digits,
        result=example.result,
        operand_width=example.operand_width,
        format_version=example.format_version,
        target_override=override,
    )


from self.tasks.multiplication_sampling import (  # noqa: E402
    analyze_partial_products,
    build_multiplication_component_payload,
    build_multiplication_long_dataset,
    build_multiplication_seed_dataset,
    generate_long_multiplication_example,
    generate_multiplication_seed_example,
    get_multiplication_slice_name,
    iter_multiplication_sizes,
    random_int_with_exact_digits,
    split_value_into_blocks,
)

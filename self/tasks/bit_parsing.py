#!/usr/bin/env python3
"""Prediction parsing and target formatting for bit-string tasks."""

from __future__ import annotations

import re
from typing import Any, Optional

from self.core.evaluation import extract_numeric_answer


INTEGER_PATTERN = re.compile(r"[-+]?\d+")
RUN_LENGTH_FORMATS = {"legacy", "symbolic_v1"}
MULTIPLICATION_FORMATS = {"legacy", "symbolic_v1"}
RUN_LENGTH_TARGET_RUN_STATE = "run_state"
RUN_LENGTH_ALPHABET_SYMBOLS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

SYMBOL_RUN_PAIR_PATTERN = re.compile(
    rf"([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])\s*(?:\||,|:|\s+)\s*([-+]?\d+)"
)

RUN_LENGTH_RUN_STATE_PATTERN = re.compile(
    rf"([-+]?\d+)\s*(?:\||,|:|\s+)\s*([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])"
    rf"\s*(?:\||,|:|\s+)\s*([-+]?\d+)\s*(?:\||,|:|\s+)\s*"
    rf"([{re.escape(RUN_LENGTH_ALPHABET_SYMBOLS)}])\s*(?:\||,|:|\s+)\s*([-+]?\d+)"
)


def format_multiplication_target(value: int, digits: int, format_version: str) -> str:
    if format_version == "symbolic_v1":
        return f"{value:0{digits * 2}d}"
    return str(value)


def parse_multiplication_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    value = extract_numeric_answer(text)
    if value is None:
        return None
    if example is None or getattr(example, "format_version", "legacy") != "symbolic_v1":
        return value
    return format_multiplication_target(int(value), int(example.digits), str(example.format_version))


def parse_run_length_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    matches = INTEGER_PATTERN.findall(text)
    target_mode = getattr(example, "target_mode", "default") if example is not None else "default"
    if target_mode == "plain_output":
        if not matches:
            return None
        value = int(matches[-1])
        if value < 0:
            return None
        return str(value)
    if target_mode == "symbol_run_pair":
        return parse_run_length_symbol_pair_prediction(text, example)
    if target_mode == RUN_LENGTH_TARGET_RUN_STATE:
        return parse_run_length_run_state_prediction(text, example)
    if len(matches) < 3:
        return None
    max_run = int(matches[0])
    prefix = int(matches[1])
    suffix = int(matches[2])
    return f"{max_run}|{prefix}|{suffix}"


def parse_run_length_symbol_pair_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    allowed_symbols = set(RUN_LENGTH_ALPHABET_SYMBOLS)
    if example is not None and getattr(example, "bitstring", None):
        allowed_symbols = set(str(example.bitstring))
    for match in SYMBOL_RUN_PAIR_PATTERN.finditer(text):
        symbol = match.group(1)
        if symbol not in allowed_symbols:
            continue
        value = int(match.group(2))
        if value < 0:
            continue
        return f"{symbol}|{value}"
    return None


def parse_run_length_run_state_prediction(text: str, example: Optional[Any] = None) -> Optional[str]:
    allowed_symbols = set(RUN_LENGTH_ALPHABET_SYMBOLS)
    bits = None
    if example is not None:
        if getattr(example, "bitstring", None):
            allowed_symbols = set(str(example.bitstring))
        bits = int(getattr(example, "bits", 0) or 0)
    for match in RUN_LENGTH_RUN_STATE_PATTERN.finditer(text):
        max_run = int(match.group(1))
        prefix_symbol = match.group(2)
        prefix_run = int(match.group(3))
        suffix_symbol = match.group(4)
        suffix_run = int(match.group(5))
        if prefix_symbol not in allowed_symbols or suffix_symbol not in allowed_symbols:
            continue
        if max_run < 0 or prefix_run < 0 or suffix_run < 0:
            continue
        if bits and (max_run > bits or prefix_run > bits or suffix_run > bits):
            continue
        return f"{max_run}|{prefix_symbol}|{prefix_run}|{suffix_symbol}|{suffix_run}"
    return None

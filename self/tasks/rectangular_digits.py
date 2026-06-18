"""Digit-order and reverse-CoT helpers for rectangular multiplication."""

from __future__ import annotations

import re
from typing import List, Optional


ALLOWED_COT_TRACE_CHARS = set("0123456789+=()")


def split_digits_lsd_first(value: int) -> List[int]:
    if value < 0:
        raise ValueError("Negative values are not supported.")
    if value == 0:
        return [0]
    digits: List[int] = []
    remaining = value
    while remaining > 0:
        digits.append(remaining % 10)
        remaining //= 10
    return digits


def reverse_digit_text(value: int) -> str:
    return "".join(str(digit) for digit in split_digits_lsd_first(value))


def format_cot_reverse_prompt(a: int, b: int) -> str:
    return f"{reverse_digit_text(a)}*{reverse_digit_text(b)}="


def format_cot_reverse_target(a: int, b: int) -> str:
    a_digits = split_digits_lsd_first(a)
    b_digits = split_digits_lsd_first(b)
    len_a = len(a_digits)
    len_b = len(b_digits)

    cumulative = 0
    chunks: List[str] = []
    for index, digit in enumerate(b_digits):
        shifted_partial = digit * a * (10**index)
        cumulative += shifted_partial
        partial_text = reverse_digit_text(shifted_partial).ljust(index + len_a + 1, "0")
        if index == 0:
            chunks.append(partial_text)
            continue
        chunks.append("+" + partial_text)
        if index < len_b - 1:
            cumulative_text = reverse_digit_text(cumulative).ljust(index + len_a + 1, "0")
            chunks.append("(" + cumulative_text + ")")

    final_answer = reverse_digit_text(a * b).ljust(len_a + len_b, "0")
    chunks.append("=" + final_answer)
    return "".join(chunks)


def normalize_cot_reverse_prediction_for_training(text: str) -> Optional[str]:
    compact = "".join(char for char in text if char not in {" ", "\n", "\r", "\t"})
    filtered = "".join(char for char in compact if char in ALLOWED_COT_TRACE_CHARS)
    return filtered or None


def extract_cot_reverse_final_digits(text: str, total_digits: int) -> Optional[str]:
    if total_digits <= 0:
        raise ValueError("total_digits must be positive.")
    compact = "".join(char for char in text if char not in {" ", "\n", "\r", "\t"})
    tail = compact.rsplit("=", 1)[-1]
    match = re.search(r"\d+", tail)
    if match is None:
        return None
    reversed_digits = match.group(0)
    if len(reversed_digits) < total_digits:
        reversed_digits = reversed_digits.ljust(total_digits, "0")
    else:
        reversed_digits = reversed_digits[:total_digits]
    return reversed_digits


def parse_cot_reverse_final_value(text: str, total_digits: int) -> Optional[int]:
    reversed_digits = extract_cot_reverse_final_digits(text, total_digits)
    if reversed_digits is None:
        return None
    return int(reversed_digits[::-1])

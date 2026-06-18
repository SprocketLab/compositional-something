#!/usr/bin/env python3
"""Run addition self-improvement with controlled composition-error rates."""

from __future__ import annotations

import argparse
from typing import List, Optional, Sequence

from self.self_improvement import main as self_improvement_main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch self_improvement.py with a specified percentage of boundary-carry composition errors retained "
            "in the pseudo labels."
        )
    )
    parser.add_argument(
        "--composition-error-percent",
        type=float,
        default=0.0,
        help="Percentage (0-100) of boundary-carry pseudo labels to retain when stitching with carry filtering.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to self_improvement.py (use '--' to separate).",
    )
    return parser.parse_args(argv)


def build_forward_args(args: argparse.Namespace) -> List[str]:
    composition_error_percent = args.composition_error_percent
    if composition_error_percent < 0.0 or composition_error_percent > 100.0:
        raise ValueError("composition-error-percent must be between 0 and 100.")

    forwarded = list(args.extra_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forwarded.append(f"--composition-error-percent={composition_error_percent}")
    has_composed_strategy = False
    for token in forwarded:
        if token == "--composed-strategy":
            has_composed_strategy = True
            break
        if token.startswith("--composed-strategy="):
            has_composed_strategy = True
            break
    if not has_composed_strategy:
        forwarded.append("--composed-strategy=with_carry_filtered")
    return forwarded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    forwarded_args = build_forward_args(args)
    self_improvement_main(forwarded_args)


__all__ = [
    "build_forward_args",
    "main",
    "parse_args",
    "self_improvement_main",
]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run-length-specific entrypoint for the shared self-improvement scaffold."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from self.legacy.bit_task_self_improvement import build_bit_task_parser, normalize_bit_task_args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_bit_task_parser(
        description="Self-improvement run-length experiment (resumable)",
        default_output_dir="artifacts/runs/self_improvement_run_length",
    )
    return parser.parse_args(argv)


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    return normalize_bit_task_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    from self.self_improvement_core import run_self_improvement
    from self.self_improvement_tasks import RunLengthTask

    args = normalize_args(parse_args(argv))
    run_self_improvement(args, RunLengthTask())


if __name__ == "__main__":
    main()

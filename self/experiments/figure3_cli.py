"""Shared CLI path and argument helpers for Figure 3 experiment scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from self.experiments.figure3_common import DEFAULT_LOG_DIR


def default_manifest(out_root: Path) -> Path:
    return out_root / "manifest.json"


def default_selection(out_root: Path) -> Path:
    return out_root / "selection.json"


def default_summary(out_root: Path) -> Path:
    return out_root / "summary.json"


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_log_dir: Path = DEFAULT_LOG_DIR,
    default_python_bin: str = sys.executable,
) -> None:
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=default_log_dir)
    parser.add_argument("--python-bin", type=str, default=default_python_bin)
    parser.add_argument("--dry-run", action="store_true")

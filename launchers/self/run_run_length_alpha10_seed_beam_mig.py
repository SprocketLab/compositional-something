#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.run_length_alpha10_seed_beam`."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from self.experiments.run_length_alpha10_seed_beam import main  # noqa: E402


if __name__ == "__main__":
    main()

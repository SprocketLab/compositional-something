#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.analysis.summarize_seed_fit_grid`."""

from __future__ import annotations

from self.analysis.summarize_seed_fit_grid import *  # noqa: F401,F403
from self.analysis.summarize_seed_fit_grid import main as _main


if __name__ == "__main__":
    _main()

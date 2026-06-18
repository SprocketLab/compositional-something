#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.figure2_condition_sweep`."""

from __future__ import annotations

from self.experiments.figure2_condition_sweep import *  # noqa: F401,F403
from self.experiments.figure2_condition_sweep import main as _main


if __name__ == "__main__":
    _main()

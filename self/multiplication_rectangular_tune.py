#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.multiplication_rectangular_tune`."""

from __future__ import annotations

from self.experiments.multiplication_rectangular_tune import *  # noqa: F401,F403
from self.experiments.multiplication_rectangular_tune import main as _main


if __name__ == "__main__":
    _main()

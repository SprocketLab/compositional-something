#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.seed_fit_experiment`."""

from __future__ import annotations

from self.experiments.seed_fit_experiment import *  # noqa: F401,F403
from self.experiments.seed_fit_experiment import main as _main


if __name__ == "__main__":
    _main()

#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.diagnostics.analyze_symbolic_training_dynamics`."""

from __future__ import annotations

from self.diagnostics.analyze_symbolic_training_dynamics import *  # noqa: F401,F403
from self.diagnostics.analyze_symbolic_training_dynamics import main as _main


if __name__ == "__main__":
    _main()

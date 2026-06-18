#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.diagnostics.evaluate_fixed_composition_slices`."""

from __future__ import annotations

from self.diagnostics.evaluate_fixed_composition_slices import *  # noqa: F401,F403
from self.diagnostics.evaluate_fixed_composition_slices import main as _main


if __name__ == "__main__":
    _main()

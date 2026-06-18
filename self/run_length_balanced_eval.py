#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.diagnostics.run_length_balanced_eval`."""

from __future__ import annotations

from self.diagnostics.run_length_balanced_eval import *  # noqa: F401,F403
from self.diagnostics.run_length_balanced_eval import main as _main


if __name__ == "__main__":
    _main()

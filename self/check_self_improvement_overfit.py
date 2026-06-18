#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.diagnostics.check_self_improvement_overfit`."""

from __future__ import annotations

from self.diagnostics.check_self_improvement_overfit import *  # noqa: F401,F403
from self.diagnostics.check_self_improvement_overfit import main as _main


if __name__ == "__main__":
    _main()

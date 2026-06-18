#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.legacy.multiplication_cot_pseudo_addition`."""

from __future__ import annotations

from self.legacy.multiplication_cot_pseudo_addition import *  # noqa: F401,F403
from self.legacy.multiplication_cot_pseudo_addition import main as _main


if __name__ == "__main__":
    _main()

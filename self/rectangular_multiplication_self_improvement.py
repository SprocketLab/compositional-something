#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.rectangular_multiplication_self_improvement`."""

from __future__ import annotations

from self.experiments.rectangular_multiplication_self_improvement import *  # noqa: F401,F403
from self.experiments.rectangular_multiplication_self_improvement import main as _main


if __name__ == "__main__":
    _main()

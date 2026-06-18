#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.figure2_paper_retune`."""

from __future__ import annotations

from self.experiments.figure2_paper_retune import *  # noqa: F401,F403
from self.experiments.figure2_paper_retune import main as _main


if __name__ == "__main__":
    _main()

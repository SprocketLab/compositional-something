#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.analysis.plot_appendix_baseline_heatmaps`."""

from __future__ import annotations

from self.analysis.plot_appendix_baseline_heatmaps import *  # noqa: F401,F403
from self.analysis.plot_appendix_baseline_heatmaps import main as _main


if __name__ == "__main__":
    _main()

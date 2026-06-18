#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.analysis.plot_self_improvement_figure`."""

from __future__ import annotations

from self.analysis.plot_self_improvement_figure import *  # noqa: F401,F403
from self.analysis.plot_self_improvement_figure import main as _main


if __name__ == "__main__":
    _main()

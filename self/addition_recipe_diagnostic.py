#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.diagnostics.addition_recipe_diagnostic`."""

from __future__ import annotations

from self.diagnostics.addition_recipe_diagnostic import *  # noqa: F401,F403
from self.diagnostics.addition_recipe_diagnostic import main as _main


if __name__ == "__main__":
    _main()

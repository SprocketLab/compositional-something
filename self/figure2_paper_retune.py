#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.figure2_paper_retune`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy, module_star_export_names
from self.experiments import figure2_paper_retune as _impl

install_module_proxy(__name__, _impl, export_names=module_star_export_names(_impl))


if __name__ == "__main__":
    _impl.main()

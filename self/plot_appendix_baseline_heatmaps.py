#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.analysis.plot_appendix_baseline_heatmaps`."""

from __future__ import annotations

from self.analysis import plot_appendix_baseline_heatmaps as _impl
from self.core.module_proxy import install_module_proxy, module_star_export_names

install_module_proxy(__name__, _impl, export_names=module_star_export_names(_impl))


if __name__ == "__main__":
    _impl.main()

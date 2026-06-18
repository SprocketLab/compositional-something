"""Compatibility wrapper for :mod:`self.analysis.training_curve_notebook_utils`."""

from __future__ import annotations

from self.analysis import training_curve_notebook_utils as _impl
from self.core.module_proxy import install_module_proxy, module_star_export_names

install_module_proxy(__name__, _impl, export_names=module_star_export_names(_impl))

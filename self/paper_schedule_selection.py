"""Compatibility wrapper for :mod:`self.experiments.paper_schedule_selection`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy, module_star_export_names
from self.experiments import paper_schedule_selection as _impl

install_module_proxy(__name__, _impl, export_names=module_star_export_names(_impl))

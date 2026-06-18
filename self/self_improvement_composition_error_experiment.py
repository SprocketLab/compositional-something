#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.composition_error_sweep`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy
from self.experiments import composition_error_sweep as _impl

install_module_proxy(__name__, _impl, export_names=_impl.__all__)


if __name__ == "__main__":
    _impl.main()

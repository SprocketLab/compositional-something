#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.tasks.rectangular_multiplication`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy
from self.tasks import rectangular_multiplication as _impl

install_module_proxy(__name__, _impl)

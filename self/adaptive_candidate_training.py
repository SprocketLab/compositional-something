#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.adaptive.run.driver`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy
from self.adaptive.run import driver as _impl

install_module_proxy(__name__, _impl)


if __name__ == "__main__":
    _impl.main()

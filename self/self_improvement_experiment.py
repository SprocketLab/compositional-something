#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.legacy.addition_self_improvement`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy
from self.legacy import addition_self_improvement as _impl

install_module_proxy(__name__, _impl)


if __name__ == "__main__":
    _impl.main()

#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`self.experiments.composition_error_sweep`."""

from __future__ import annotations

import sys as _sys
import types as _types

from self.experiments import composition_error_sweep as _impl


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))


class _ModuleProxy(_types.ModuleType):
    def __getattr__(self, name: str):
        return getattr(_impl, name)

    def __setattr__(self, name: str, value):
        if not name.startswith("__"):
            setattr(_impl, name, value)
        super().__setattr__(name, value)


__all__ = list(_impl.__all__)
_sys.modules[__name__].__class__ = _ModuleProxy


if __name__ == "__main__":
    _impl.main()

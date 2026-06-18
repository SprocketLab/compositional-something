"""Compatibility helpers for old module paths that proxy canonical modules."""

from __future__ import annotations

import sys
import types
from types import ModuleType
from typing import Iterable, Optional


_INTERNAL_PROXY_NAMES = {"_compat_impl", "_compat_export_names"}


class ModuleProxy(types.ModuleType):
    """Module proxy that forwards missing reads and monkeypatch writes."""

    def __getattr__(self, name: str):
        return getattr(self.__dict__["_compat_impl"], name)

    def __setattr__(self, name: str, value):
        if not name.startswith("__") and name not in _INTERNAL_PROXY_NAMES:
            setattr(self.__dict__["_compat_impl"], name, value)
        super().__setattr__(name, value)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(dir(self.__dict__["_compat_impl"])))


def install_module_proxy(
    module_name: str,
    impl: ModuleType,
    *,
    export_names: Optional[Iterable[str]] = None,
) -> None:
    """Make ``module_name`` behave as a monkeypatch-friendly proxy for ``impl``."""

    module = sys.modules[module_name]
    names = (
        list(export_names)
        if export_names is not None
        else [name for name in dir(impl) if not name.startswith("__")]
    )

    def _compat_getattr(name: str):
        return getattr(impl, name)

    def _compat_dir():
        return sorted(set(module.__dict__) | set(dir(impl)))

    module.__dict__["_compat_impl"] = impl
    module.__dict__["_compat_export_names"] = tuple(names)
    module.__dict__["__all__"] = names
    module.__dict__["__getattr__"] = _compat_getattr
    module.__dict__["__dir__"] = _compat_dir
    module.__class__ = ModuleProxy


def module_star_export_names(impl: ModuleType) -> list[str]:
    """Return names exported by ``from impl import *``."""

    if hasattr(impl, "__all__"):
        return list(impl.__all__)
    return [name for name in dir(impl) if not name.startswith("_")]

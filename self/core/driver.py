#!/usr/bin/env python3
"""Candidate-training loop for adaptive config self-improvement."""

from __future__ import annotations

import argparse
import sys
from typing import Any, List, Optional, Sequence

from self.core import driver_default_bindings, driver_wiring
from self.core.driver_compat_manifest import COMPAT_EXPORT_NAMES
from self.core.driver_public_api import install_driver_wiring_delegates

_DEFAULT_BINDING_NAME_SET = frozenset(driver_default_bindings.DEFAULT_BINDING_NAMES)
_COMPAT_EXPORT_NAME_SET = frozenset(COMPAT_EXPORT_NAMES)


def _bindings() -> Any:
    return sys.modules[__name__]


def __getattr__(name: str) -> Any:
    if name in _DEFAULT_BINDING_NAME_SET:
        return getattr(driver_default_bindings, name)
    if name in _COMPAT_EXPORT_NAME_SET:
        from self.core import driver_compat_exports

        return getattr(driver_compat_exports, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | _DEFAULT_BINDING_NAME_SET | _COMPAT_EXPORT_NAME_SET)


install_driver_wiring_delegates(globals(), driver_wiring=driver_wiring, get_bindings=_bindings)


def _default_bf16_on_cuda(args: argparse.Namespace, label: str) -> None:
    return driver_default_bindings._default_bf16_on_cuda(args, label)


def main(argv: Optional[Sequence[str]] = None) -> None:
    return driver_wiring.main(_bindings(), argv)


__all__ = [name for name in __dir__() if not name.startswith("__")]


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compatibility facade for iterative compositional self-improvement."""

from __future__ import annotations

import sys
from typing import Any

from self.nonadaptive import nonadaptive_facade_exports as _facade_exports
from self.nonadaptive.nonadaptive_compat import (
    NONADAPTIVE_PATCHABLE_NAMES,
    sync_nonadaptive_loop_globals,
)
from self.nonadaptive.nonadaptive_facade_exports import NONADAPTIVE_FACADE_EXPORT_NAMES

__all__ = list(NONADAPTIVE_FACADE_EXPORT_NAMES)

_NONADAPTIVE_PATCHABLE_NAMES = NONADAPTIVE_PATCHABLE_NAMES
_NONADAPTIVE_FACADE_EXPORT_NAME_SET = frozenset(NONADAPTIVE_FACADE_EXPORT_NAMES)


def __getattr__(name: str) -> Any:
    if name == "run_self_improvement":
        return run_self_improvement
    if name not in _NONADAPTIVE_FACADE_EXPORT_NAME_SET:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_facade_exports, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NONADAPTIVE_FACADE_EXPORT_NAME_SET)


def _ensure_patchable_defaults() -> None:
    for name in _NONADAPTIVE_PATCHABLE_NAMES:
        if name not in globals():
            globals()[name] = getattr(_facade_exports, name)


def run_self_improvement(args: Any, task: "SelfImprovementTask") -> None:
    from self.nonadaptive import nonadaptive_loop as _nonadaptive_loop

    _ensure_patchable_defaults()
    sync_nonadaptive_loop_globals(
        source_globals=sys.modules[__name__].__dict__,
        target_module=_nonadaptive_loop,
        names=_NONADAPTIVE_PATCHABLE_NAMES,
    )
    return _nonadaptive_loop.run_self_improvement(args, task)

#!/usr/bin/env python3
"""Task-agnostic scaffold for iterative compositional self-improvement."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from self.core import nonadaptive_loop as _nonadaptive_loop
from self.core.nonadaptive_facade_exports import *  # noqa: F401,F403
from self.core.nonadaptive_facade_exports import NONADAPTIVE_FACADE_EXPORT_NAMES
from self.core.nonadaptive_compat import (
    NONADAPTIVE_PATCHABLE_NAMES,
    sync_nonadaptive_loop_globals,
)
from self.core.task_protocols import (
    SelfImprovementTask,
)

__all__ = list(NONADAPTIVE_FACADE_EXPORT_NAMES)

_NONADAPTIVE_PATCHABLE_NAMES = NONADAPTIVE_PATCHABLE_NAMES


def run_self_improvement(args: Any, task: SelfImprovementTask) -> None:
    sync_nonadaptive_loop_globals(
        source_globals=globals(),
        target_module=_nonadaptive_loop,
        names=_NONADAPTIVE_PATCHABLE_NAMES,
    )
    return _nonadaptive_loop.run_self_improvement(args, task)

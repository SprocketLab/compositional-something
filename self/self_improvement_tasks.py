#!/usr/bin/env python3
"""Compatibility facade for task-specific self-improvement adapters.

Canonical task implementations live in :mod:`self.tasks`. This module keeps
the historical import path available for legacy scripts, notebooks, and tests.
"""

from __future__ import annotations

from self.tasks.compat_exports import *  # noqa: F401,F403
from self.tasks.compat_exports import TASK_COMPAT_EXPORT_NAMES

__all__ = list(TASK_COMPAT_EXPORT_NAMES)

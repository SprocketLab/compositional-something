"""Lazy compatibility exports for :mod:`self.core.driver`.

The adaptive driver used to import a broad surface of task helpers, proposal
helpers, and data classes directly. Keep those old attribute/import paths
available here without importing the full runtime stack during module import.
"""

from __future__ import annotations

from typing import Any

from self.core.driver_compat_manifest import COMPAT_EXPORT_NAMES
from self.core.driver_compat_targets import DRIVER_COMPAT_EXPORT_TARGETS
from self.core.lazy_exports import (
    lazy_export_dir,
    resolve_lazy_export,
    validate_lazy_export_targets,
)

validate_lazy_export_targets(
    export_names=COMPAT_EXPORT_NAMES,
    targets=DRIVER_COMPAT_EXPORT_TARGETS,
    label="driver compatibility",
)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        targets=DRIVER_COMPAT_EXPORT_TARGETS,
        module_globals=globals(),
    )


def __dir__() -> list[str]:
    return lazy_export_dir(globals(), COMPAT_EXPORT_NAMES)


__all__ = list(COMPAT_EXPORT_NAMES)

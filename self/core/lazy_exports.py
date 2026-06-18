"""Shared helpers for lightweight compatibility export modules."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

LazyExportTarget = tuple[str, str | None]


def validate_lazy_export_targets(
    *,
    export_names: Sequence[str],
    targets: Mapping[str, LazyExportTarget],
    label: str,
) -> None:
    """Fail fast when a compatibility manifest has no canonical target."""

    missing_targets = set(export_names) - set(targets)
    if missing_targets:
        missing = ", ".join(sorted(missing_targets))
        raise RuntimeError(f"Missing lazy {label} targets: {missing}")


def resolve_lazy_export(
    name: str,
    *,
    module_name: str,
    targets: Mapping[str, LazyExportTarget],
    module_globals: dict[str, Any],
) -> Any:
    """Resolve and cache one compatibility export."""

    target = targets.get(name)
    if target is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    target_module_name, attr_name = target
    target_module = importlib.import_module(target_module_name)
    value = target_module if attr_name is None else getattr(target_module, attr_name)
    module_globals[name] = value
    return value


def lazy_export_dir(module_globals: Mapping[str, Any], export_names: Sequence[str]) -> list[str]:
    """Return a stable ``dir()`` for a lazy compatibility module."""

    return sorted(set(module_globals) | set(export_names))

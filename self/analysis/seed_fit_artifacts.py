"""Seed-fit artifact discovery and loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from self.analysis.artifact_io import JsonDict, natural_sort_key, read_json


SEED_FIT_RESULTS_FILE = "seed_fit_results.json"


def is_seed_fit_run_dir(path: Path | str) -> bool:
    resolved = Path(path)
    return resolved.is_dir() and (resolved / SEED_FIT_RESULTS_FILE).exists()


def resolve_seed_fit_results_path(path: Path | str) -> Path:
    resolved = Path(path)
    if resolved.is_dir():
        resolved = resolved / SEED_FIT_RESULTS_FILE
    if not resolved.exists():
        raise FileNotFoundError(f"Could not find seed-fit results at {resolved}.")
    if resolved.name != SEED_FIT_RESULTS_FILE:
        raise ValueError(f"Expected {SEED_FIT_RESULTS_FILE}, got {resolved.name}.")
    return resolved


def discover_seed_fit_results(root: Path | str) -> list[Path]:
    resolved = Path(root)
    if resolved.is_file():
        return [resolve_seed_fit_results_path(resolved)]
    if is_seed_fit_run_dir(resolved):
        return [resolved / SEED_FIT_RESULTS_FILE]
    if not resolved.exists():
        return []
    return sorted(resolved.rglob(SEED_FIT_RESULTS_FILE), key=natural_sort_key)


def load_seed_fit_result(path: Path | str) -> JsonDict:
    results_path = resolve_seed_fit_results_path(path)
    payload = read_json(results_path, {})
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected object payload in {results_path}.")
    return dict(payload)


def load_seed_fit_results(root: Path | str) -> list[tuple[Path, JsonDict]]:
    return [(path, load_seed_fit_result(path)) for path in discover_seed_fit_results(root)]

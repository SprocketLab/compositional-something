"""Shared IO helpers for analysis artifact loaders."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

ADAPTIVE_RESULTS_FILE = "adaptive_candidate_training_results.json"
SELF_IMPROVEMENT_RESULTS_FILE = "self_improvement_results.json"


def read_json(path: Path | str, default: Any = None) -> Any:
    resolved = Path(path)
    if not resolved.exists():
        return default
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path | str) -> list[JsonDict]:
    resolved = Path(path)
    if not resolved.exists():
        return []
    rows: list[JsonDict] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for chunk in re.split(r"(\d+)", str(path)):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk)
    return tuple(parts)

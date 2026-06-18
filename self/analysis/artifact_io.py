"""Shared IO helpers for analysis artifact loaders."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
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


def read_round_summaries(path: Path | str, default: Any = None) -> list[JsonDict]:
    """Load self-improvement round records from supported results JSON shapes."""

    resolved = Path(path)
    if not resolved.exists():
        if default is None:
            raise FileNotFoundError(f"Could not find round-summary results at {resolved}.")
        payload = default
    else:
        payload = read_json(resolved)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping) and isinstance(payload.get("rounds"), list):
        rows = payload["rounds"]
    else:
        raise ValueError(f"Expected list of round records or object with rounds list in {resolved}.")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for chunk in re.split(r"(\d+)", str(path)):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            parts.append(chunk)
    return tuple(parts)

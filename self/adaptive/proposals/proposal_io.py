"""Proposal fixture and trace JSONL IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from self.core.data_io import sanitize_json_value


JsonDict = Dict[str, Any]


def load_fixture_proposals(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(payload)
    return rows


def build_trace_row(
    *,
    round_index: int,
    task: str,
    condition: str,
    reward: float,
    frontier_delta: float,
    final_accuracy: float,
    prompt: str,
    completion: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> JsonDict:
    row: JsonDict = {
        "round": round_index,
        "task": task,
        "condition": condition,
        "reward": reward,
        "frontier_delta": frontier_delta,
        "final_accuracy": final_accuracy,
        "prompt": prompt,
        "completion": completion,
    }
    if extra_metadata:
        row["metadata"] = dict(extra_metadata)
    return sanitize_json_value(row)


def write_trace_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(sanitize_json_value(dict(row)), handle, sort_keys=True)
            handle.write("\n")

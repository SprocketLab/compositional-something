#!/usr/bin/env python3
"""JSON/spec path helpers for adaptive candidate workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from self.core.data_io import sanitize_json_value


PATH_ARG_NAMES = {
    "output_dir",
    "proposal_fixture_jsonl",
    "plan_log_path",
    "controller_worker_sbatch_script",
    "controller_worker_spec",
    "candidate_worker_spec",
    "candidate_worker_pack_spec",
}


def json_ready_args(args: argparse.Namespace) -> dict[str, Any]:
    return sanitize_json_value(vars(args))


def namespace_from_json_args(payload: Mapping[str, Any]) -> argparse.Namespace:
    args_payload = dict(payload)
    for name in PATH_ARG_NAMES:
        value = args_payload.get(name)
        if value is not None:
            args_payload[name] = Path(str(value))
    return argparse.Namespace(**args_payload)


def json_ready_key(key: Any) -> Any:
    if isinstance(key, tuple):
        return [json_ready_key(value) for value in key]
    if isinstance(key, list):
        return [json_ready_key(value) for value in key]
    return key


def key_from_json(payload: Any) -> Any:
    if isinstance(payload, list):
        return tuple(key_from_json(value) for value in payload)
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_key_set(path: Path, keys: Iterable[Any]) -> None:
    write_json(path, [json_ready_key(key) for key in sorted(keys, key=repr)])


def load_key_set(path: Path) -> set[Any]:
    return {key_from_json(value) for value in load_json(path)}


def controller_worker_failure_path(worker_dir: Path) -> Path:
    return worker_dir / "worker_failure.json"


def controller_worker_output_path(worker_dir: Path) -> Path:
    return worker_dir / "worker_output.json"


def candidate_metric_path(round_dir: Path, candidate_index: int) -> Path:
    return round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "candidate_metrics.json"


def candidate_worker_failure_path(round_dir: Path, candidate_index: int) -> Path:
    return round_dir / "candidates" / f"candidate_{candidate_index:02d}" / "worker_failure.json"


def clear_worker_entry_flags(args_payload: dict[str, Any]) -> dict[str, Any]:
    args_payload["run_candidate_worker"] = False
    args_payload["candidate_worker_spec"] = None
    args_payload["run_candidate_pack_worker"] = False
    args_payload["candidate_worker_pack_spec"] = None
    args_payload["run_controller_worker"] = False
    args_payload["controller_worker_spec"] = None
    return args_payload

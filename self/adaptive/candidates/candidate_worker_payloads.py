"""Candidate work-item payload conversion at worker boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from self.core import worker_io
from self.core.data_io import load_examples, sanitize_json_value
from self.core.models import CandidateWorkItem, ExactPairDataset, proposal_from_payload


JsonDict = Dict[str, Any]


def candidate_payload_from_work_item(item: CandidateWorkItem) -> JsonDict:
    """Return the candidate block embedded in worker specs."""
    return sanitize_json_value(
        {
            "index": item.index,
            "row_id": item.row_id,
            "proposal": item.proposal.to_json_dict(),
            "completion": item.completion,
            "raw_output": item.raw_output,
            "proposal_prediction": item.proposal_prediction,
            "pseudo_diagnostics": item.pseudo_diagnostics,
        }
    )


def candidate_payload_to_work_item(
    *,
    payload: Mapping[str, Any],
    pseudo_examples: Sequence[Any],
    composed_keys: Sequence[Any] | set[Any] | None = None,
) -> CandidateWorkItem:
    """Rebuild a candidate work item from a serialized candidate block."""
    return CandidateWorkItem(
        index=int(payload["index"]),
        row_id=payload.get("row_id"),
        proposal=proposal_from_payload(dict(payload["proposal"])),
        completion=str(payload.get("completion", "")),
        raw_output=payload.get("raw_output"),
        composed=ExactPairDataset(
            examples=[],
            component_map={},
            keys=set(composed_keys or ()),
            diagnostics={},
        ),
        pseudo_examples=list(pseudo_examples),
        pseudo_diagnostics=dict(payload.get("pseudo_diagnostics") or {}),
        proposal_prediction=dict(payload.get("proposal_prediction") or {}),
    )


def work_item_to_worker_payload(
    *,
    item: CandidateWorkItem,
    round_dir: Path,
) -> JsonDict:
    candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
    payload = candidate_payload_from_work_item(item)
    payload.update(
        {
            "pseudo_examples_path": str(candidate_dir / "pseudo_examples.jsonl"),
            "pseudo_count": len(item.pseudo_examples),
            "composed_keys": [
                worker_io.json_ready_key(key)
                for key in sorted(item.composed.keys, key=repr)
            ],
            "composed_count": len(item.composed.examples),
        }
    )
    return sanitize_json_value(payload)


def work_item_from_worker_payload(
    *,
    payload: Mapping[str, Any],
    task: Any,
) -> CandidateWorkItem:
    pseudo_path = Path(str(payload["pseudo_examples_path"]))
    pseudo_examples = load_examples(pseudo_path, task.deserialize_example)
    composed_keys = {worker_io.key_from_json(key) for key in payload.get("composed_keys", [])}
    return candidate_payload_to_work_item(
        payload=payload,
        pseudo_examples=pseudo_examples,
        composed_keys=composed_keys,
    )


def candidate_worker_failure_payload(spec_path: Path, exc: Exception) -> JsonDict:
    return {
        "spec_path": str(spec_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def candidate_worker_failure_path_from_payload(payload: Mapping[str, Any]) -> Path:
    candidate_index = int(payload["candidate_index"])
    round_dir = Path(str(payload["round_dir"]))
    return worker_io.candidate_worker_failure_path(round_dir, candidate_index)


def write_candidate_worker_failure(
    *,
    spec_path: Path,
    spec_payload: Mapping[str, Any],
    exc: Exception,
    write_json_fn: Callable[[Path, Any], None],
) -> JsonDict:
    failure_payload = candidate_worker_failure_payload(spec_path, exc)
    write_json_fn(candidate_worker_failure_path_from_payload(spec_payload), failure_payload)
    return failure_payload


def write_candidate_worker_failure_from_spec(
    spec_path: Path,
    exc: Exception,
    *,
    load_json_fn: Callable[[Path], Any],
    write_json_fn: Callable[[Path, Any], None],
) -> JsonDict:
    spec_payload = load_json_fn(spec_path)
    return write_candidate_worker_failure(
        spec_path=spec_path,
        spec_payload=spec_payload,
        exc=exc,
        write_json_fn=write_json_fn,
    )

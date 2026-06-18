"""Candidate worker spec and pack-spec artifact generation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from self.core import worker_io
from self.core.candidate_local_workers import write_json
from self.core.data_io import ensure_dir, save_examples
from self.core.proposals import PromptBundle, write_trace_jsonl


JsonDict = Dict[str, Any]


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


def prepare_candidate_worker_specs(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[Any],
    outcome_trace_buffer: Sequence[Any],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[Any],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
) -> List[Path]:
    job_dir = round_dir / "candidate_jobs"
    input_dir = job_dir / "inputs"
    spec_dir = job_dir / "specs"
    ensure_dir(input_dir)
    ensure_dir(spec_dir)
    source_examples_path = input_dir / "source_examples.jsonl"
    eval_examples_path = input_dir / "eval_examples.jsonl"
    proposal_trace_path = input_dir / "proposal_trace_buffer.jsonl"
    outcome_trace_path = input_dir / "outcome_trace_buffer.jsonl"
    prompt_path = input_dir / "proposal_prompt.json"
    save_examples(source_examples_path, source_examples, task.serialize_example)
    save_examples(eval_examples_path, eval_examples, task.serialize_example)
    write_trace_jsonl(proposal_trace_path, [trace.to_json_dict() for trace in proposal_trace_buffer])
    write_trace_jsonl(outcome_trace_path, [trace.to_json_dict() for trace in outcome_trace_buffer])
    write_json(prompt_path, {"system": proposal_prompt.system, "user": proposal_prompt.user})

    spec_paths: List[Path] = []
    manifest: List[JsonDict] = []
    args_payload = worker_io.clear_worker_entry_flags(worker_io.json_ready_args(args))
    for array_index, item in enumerate(work_items):
        candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
        pseudo_examples_path = candidate_dir / "pseudo_examples.jsonl"
        spec_path = spec_dir / f"candidate_{array_index}.json"
        payload: JsonDict = {
            "args": args_payload,
            "array_index": array_index,
            "candidate_index": item.index,
            "round_index": round_index,
            "attempt_index": attempt_index,
            "current_checkpoint": current_checkpoint,
            "round_dir": str(round_dir),
            "source_examples_path": str(source_examples_path),
            "eval_examples_path": str(eval_examples_path),
            "proposal_trace_buffer_path": str(proposal_trace_path),
            "outcome_trace_buffer_path": str(outcome_trace_path),
            "proposal_prompt_path": str(prompt_path),
            "pseudo_examples_path": str(pseudo_examples_path),
            "current_final_accuracy": current_final_accuracy,
            "current_per_size_accuracy": {str(size): score for size, score in current_per_size_accuracy.items()},
            "init_final_accuracy": init_final_accuracy,
            "seed": args.seed + attempt_index * 1009 + item.index,
            "candidate": {
                "index": item.index,
                "row_id": item.row_id,
                "proposal": item.proposal.to_json_dict(),
                "completion": item.completion,
                "raw_output": item.raw_output,
                "proposal_prediction": item.proposal_prediction,
                "pseudo_diagnostics": item.pseudo_diagnostics,
            },
        }
        write_json(spec_path, payload)
        spec_paths.append(spec_path)
        manifest.append(
            {
                "array_index": array_index,
                "candidate_index": item.index,
                "spec_path": str(spec_path),
                "metrics_path": str(candidate_metric_path(round_dir, item)),
                "worker_failure_path": str(candidate_worker_failure_path(round_dir, item)),
            }
        )
    write_json(job_dir / "manifest.json", manifest)
    return spec_paths


def prepare_candidate_worker_pack_specs(
    *,
    round_dir: Path,
    work_items: Sequence[Any],
    spec_paths: Sequence[Path],
    pack_size: int,
) -> List[Tuple[int, List[Any], Path]]:
    if pack_size < 1:
        raise ValueError("pack_size must be positive.")
    pack_dir = round_dir / "candidate_jobs" / "pack_specs"
    ensure_dir(pack_dir)
    packs: List[Tuple[int, List[Any], Path]] = []
    manifest: List[JsonDict] = []
    pairs = list(zip(work_items, spec_paths))
    for pack_index, start in enumerate(range(0, len(pairs), pack_size)):
        chunk = pairs[start : start + pack_size]
        chunk_items = [item for item, _ in chunk]
        chunk_spec_paths = [spec_path for _, spec_path in chunk]
        pack_path = pack_dir / f"pack_{pack_index}.json"
        payload = {
            "pack_index": pack_index,
            "spec_paths": [str(spec_path) for spec_path in chunk_spec_paths],
            "candidates": [
                {
                    "candidate_index": item.index,
                    "spec_path": str(spec_path),
                    "metrics_path": str(candidate_metric_path(round_dir, item)),
                    "worker_failure_path": str(candidate_worker_failure_path(round_dir, item)),
                }
                for item, spec_path in chunk
            ],
        }
        write_json(pack_path, payload)
        packs.append((pack_index, chunk_items, pack_path))
        manifest.append(
            {
                "pack_index": pack_index,
                "pack_spec_path": str(pack_path),
                "candidate_indices": [item.index for item in chunk_items],
                "spec_paths": [str(spec_path) for spec_path in chunk_spec_paths],
            }
        )
    write_json(round_dir / "candidate_jobs" / "pack_manifest.json", manifest)
    return packs

"""Candidate dataset construction and pseudo-label attachment."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from self.core.composition import build_exact_pair_dataset, compose_pseudo_examples
from self.core.models import CandidateWorkItem, proposal_from_payload
from self.core.worker_io import write_json
from self.core.data_io import ensure_dir, save_examples
from self.core.evaluation import generate_prediction_map, resolve_max_new_tokens
from self.core.training import TrainingConfig

JsonDict = Dict[str, Any]


def examples_by_key(task: Any, examples: Sequence[Any]) -> dict[Any, Any]:
    by_key: dict[Any, Any] = {}
    for example in examples:
        by_key.setdefault(task.key_for_example(example), example)
    return by_key


def build_candidate_work_items(
    *,
    args: argparse.Namespace,
    task: Any,
    round_dir: Path,
    proposal_results: Sequence[Mapping[str, Any]],
    source_examples: Sequence[Any],
    exclude_keys: set[Any],
    rng: random.Random,
) -> List[CandidateWorkItem]:
    work_items: List[CandidateWorkItem] = []
    data_build_failures: List[JsonDict] = []
    for result in proposal_results:
        if not result.get("valid"):
            continue
        proposal_payload = result.get("parsed_proposal")
        if not isinstance(proposal_payload, dict):
            continue
        proposal = proposal_from_payload(proposal_payload)
        candidate_dir = round_dir / "candidates" / f"candidate_{int(result['proposal_index']):02d}"
        ensure_dir(candidate_dir)
        try:
            composed = build_exact_pair_dataset(
                task_name=args.task,
                source_examples=source_examples,
                proposal=proposal,
                per_size_count=args.candidate_train_per_size,
                rng=rng,
                exclude_keys=exclude_keys,
                progress_name=f"round_{round_dir.name}_{result['proposal_index']}",
            )
        except Exception as exc:
            failure = {
                "proposal_index": int(result["proposal_index"]),
                "id": result.get("id"),
                "parsed_proposal": proposal.to_json_dict(),
                "failure_reason": str(exc),
            }
            data_build_failures.append(failure)
            write_json(candidate_dir / "data_build_failure.json", failure)
            continue
        save_examples(candidate_dir / "composed_raw.jsonl", composed.examples, task.serialize_example)
        task.save_component_map(candidate_dir / "component_map.json", composed.component_map)
        write_json(candidate_dir / "composed_diagnostics.json", composed.diagnostics)
        work_items.append(
            CandidateWorkItem(
                index=int(result["proposal_index"]),
                row_id=result.get("id"),
                proposal=proposal,
                completion=str(result.get("completion", "")),
                raw_output=result.get("raw_output"),
                composed=composed,
                pseudo_examples=[],
                pseudo_diagnostics={},
                proposal_prediction=dict(result.get("parsed_prediction") or {}),
            )
        )
    if data_build_failures:
        write_json(round_dir / "data_build_failures.json", data_build_failures)
    return work_items


def attach_pseudo_labels(
    *,
    args: argparse.Namespace,
    task: Any,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    source_examples: Sequence[Any],
    current_model: Any,
    current_tokenizer: Any,
    config: TrainingConfig,
) -> List[CandidateWorkItem]:
    if not work_items:
        return []
    source_by_key = examples_by_key(task, source_examples)
    needed_keys: set[Any] = set()
    for item in work_items:
        for children in item.composed.component_map.values():
            needed_keys.update(children)
    missing_source = sorted((key for key in needed_keys if key not in source_by_key), key=repr)
    if missing_source:
        raise RuntimeError(f"Missing source examples for component keys: {missing_source[:5]}")
    component_examples = [source_by_key[key] for key in sorted(needed_keys, key=repr)]
    max_tokens = resolve_max_new_tokens(component_examples, config.decode_max_new_tokens)
    component_predictions = generate_prediction_map(
        model=current_model,
        tokenizer=current_tokenizer,
        examples=component_examples,
        batch_size=config.per_device_eval_batch_size,
        max_new_tokens=max_tokens,
        key_getter=task.key_for_example,
        prediction_parser=task.prediction_parser,
    )
    write_json(
        round_dir / "component_prediction_summary.json",
        {
            "component_example_count": len(component_examples),
            "prediction_count": len(component_predictions),
            "missing_count": len(component_examples) - len(component_predictions),
        },
    )

    updated: List[CandidateWorkItem] = []
    for item in work_items:
        pseudo_examples, pseudo_diagnostics = compose_pseudo_examples(
            task_name=args.task,
            task=task,
            proposal=item.proposal,
            composed_examples=item.composed.examples,
            component_map=item.composed.component_map,
            component_predictions=component_predictions,
            args=args,
        )
        candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
        save_examples(candidate_dir / "pseudo_examples.jsonl", pseudo_examples, task.serialize_example)
        write_json(candidate_dir / "pseudo_diagnostics.json", pseudo_diagnostics)
        updated.append(
            CandidateWorkItem(
                index=item.index,
                row_id=item.row_id,
                proposal=item.proposal,
                completion=item.completion,
                raw_output=item.raw_output,
                composed=item.composed,
                pseudo_examples=pseudo_examples,
                pseudo_diagnostics=pseudo_diagnostics,
                proposal_prediction=dict(item.proposal_prediction),
            )
        )
    return updated

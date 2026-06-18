"""Adaptive run output, dataset, and source-pool initialization."""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from self.core.training import TrainingConfig


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CheckpointManager:
    output_dir: Path
    keep_candidate_models: bool = False
    keep_proposal_grpo_checkpoints: bool = False

    def cleanup_unselected_candidates(
        self,
        *,
        metrics: Sequence[Any],
        selected: Optional[Any],
    ) -> None:
        if self.keep_candidate_models:
            return
        selected_dir = selected.model_dir if selected is not None else None
        for metric in metrics:
            model_dir = metric.model_dir
            if model_dir is None or model_dir == selected_dir:
                continue
            parent = model_dir.parent
            if parent.exists():
                shutil.rmtree(parent, ignore_errors=True)

    def cleanup_replaced_checkpoint(
        self,
        *,
        old_checkpoint: str,
        new_checkpoint: str,
    ) -> List[str]:
        if old_checkpoint == new_checkpoint:
            return []
        old_model_dir = Path(old_checkpoint)
        new_model_dir = Path(new_checkpoint)
        if old_model_dir.name != "model":
            return []
        if not old_model_dir.exists() or not new_model_dir.exists():
            return []
        try:
            old_model_dir.resolve().relative_to(self.output_dir.resolve())
        except (ValueError, OSError):
            return []
        if old_model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in old_model_dir.parts and self.keep_candidate_models:
            return []
        shutil.rmtree(old_model_dir, ignore_errors=True)
        return [str(old_model_dir)]


@dataclass(frozen=True)
class RunInitializationDeps:
    ensure_dir: Callable[[Path], None]
    make_config: Callable[[argparse.Namespace], TrainingConfig]
    prepare_datasets: Callable[..., tuple[Dict[str, list[Any]], Dict[str, set[Any]], list[Any], set[Any]]]
    save_examples: Callable[[Path, Sequence[Any], Callable[[Any], JsonDict]], None]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class RunInitializationResult:
    output_dir: Path
    data_dir: Path
    checkpoint_manager: CheckpointManager
    config: TrainingConfig
    source_examples: list[Any]
    source_sizes: set[int]
    exclude_keys: set[Any]
    eval_examples: list[Any]


def initialize_adaptive_run(
    *,
    args: argparse.Namespace,
    task: Any,
    rng: random.Random,
    deps: RunInitializationDeps,
) -> RunInitializationResult:
    output_dir = args.output_dir
    deps.ensure_dir(output_dir)
    data_dir = output_dir / "data"
    deps.ensure_dir(data_dir)
    checkpoint_manager = CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=args.keep_all_candidate_models,
        keep_proposal_grpo_checkpoints=args.keep_all_proposal_grpo_checkpoints,
    )

    config = deps.make_config(args)
    base_splits, base_records, eval_examples, eval_keys = deps.prepare_datasets(args, task, rng)
    deps.save_examples(data_dir / "initial_train.jsonl", base_splits["train"], task.serialize_example)
    deps.save_examples(data_dir / "initial_validation.jsonl", base_splits["validation"], task.serialize_example)
    deps.save_examples(data_dir / "initial_test.jsonl", base_splits["test"], task.serialize_example)
    deps.save_examples(data_dir / "evaluation.jsonl", eval_examples, task.serialize_example)
    deps.write_json(
        data_dir / "metadata.json",
        {
            "task": args.task,
            "initial_min_size": args.initial_min_size,
            "initial_max_size": args.initial_max_size,
            "frontier_min_size": args.frontier_min_size,
            "frontier_max_size": args.frontier_max_size,
            "initial_train_per_size": args.initial_train_per_size,
            "candidate_train_per_size": args.candidate_train_per_size,
            "eval_per_size": args.eval_per_size,
            "seed": args.seed,
        },
    )

    source_examples = list(base_splits["train"])
    source_sizes = set(range(args.initial_min_size, args.initial_max_size + 1))
    exclude_keys = set().union(*base_records.values())
    exclude_keys.update(eval_keys)

    return RunInitializationResult(
        output_dir=output_dir,
        data_dir=data_dir,
        checkpoint_manager=checkpoint_manager,
        config=config,
        source_examples=source_examples,
        source_sizes=source_sizes,
        exclude_keys=exclude_keys,
        eval_examples=eval_examples,
    )

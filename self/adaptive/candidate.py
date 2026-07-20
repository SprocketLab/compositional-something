#!/usr/bin/env python3
"""Candidate data, training, workers, and dispatch."""

from __future__ import annotations


# --- from candidate_training.py ---
"""Candidate data construction, training, scoring, and selection."""

# --- from checkpoints.py ---
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

from self.core.models import CandidateMetrics


@dataclass(frozen=True)
class CheckpointManager:
    output_dir: Path
    keep_candidate_models: bool = False
    keep_proposal_grpo_checkpoints: bool = False

    def _is_protected_checkpoint(self, model_dir: Path, protected_checkpoints: Sequence[str] = ()) -> bool:
        try:
            resolved_model_dir = model_dir.resolve()
        except OSError:
            return False
        for checkpoint in protected_checkpoints:
            if not checkpoint:
                continue
            try:
                if Path(checkpoint).resolve() == resolved_model_dir:
                    return True
            except OSError:
                continue
        return False

    def _cleanup_model_dir(self, model_dir: Path) -> List[str]:
        if model_dir.name != "model":
            return []
        if not model_dir.exists():
            return []
        try:
            model_dir.resolve().relative_to(self.output_dir.resolve())
        except (ValueError, OSError):
            return []
        shutil.rmtree(model_dir, ignore_errors=True)
        return [str(model_dir)]

    def cleanup_unselected_candidates(
        self,
        *,
        metrics: Sequence[CandidateMetrics],
        selected: Optional[CandidateMetrics],
    ) -> List[str]:
        if self.keep_candidate_models:
            return []
        deleted: List[str] = []
        selected_dir = selected.model_dir if selected is not None else None
        for metric in metrics:
            model_dir = metric.model_dir
            if model_dir is None or model_dir == selected_dir:
                continue
            deleted.extend(self._cleanup_model_dir(model_dir))
        return deleted

    def cleanup_unselectable_candidate(
        self,
        *,
        metric: CandidateMetrics,
        min_reward: float,
    ) -> List[str]:
        if self.keep_candidate_models:
            return []
        if metric.model_dir is None:
            return []
        if metric.valid and metric.reward >= min_reward:
            return []
        return self._cleanup_model_dir(metric.model_dir)

    def cleanup_final_checkpoint(
        self,
        *,
        checkpoint: str,
        keep_final: bool,
    ) -> List[str]:
        if keep_final:
            return []
        model_dir = Path(checkpoint)
        if model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in model_dir.parts and self.keep_candidate_models:
            return []
        return self._cleanup_model_dir(model_dir)

    def cleanup_replaced_checkpoint(
        self,
        *,
        old_checkpoint: str,
        new_checkpoint: str,
        protected_checkpoints: Sequence[str] = (),
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
        except ValueError:
            return []
        except OSError:
            return []
        if old_model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in old_model_dir.parts and self.keep_candidate_models:
            return []
        if self._is_protected_checkpoint(old_model_dir, protected_checkpoints):
            return []
        return self._cleanup_model_dir(old_model_dir)


def cleanup_unselected_models(
    *,
    metrics: Sequence[CandidateMetrics],
    selected: Optional[CandidateMetrics],
    keep_all: bool,
) -> None:
    CheckpointManager(output_dir=Path("."), keep_candidate_models=keep_all).cleanup_unselected_candidates(
        metrics=metrics,
        selected=selected,
    )


def cleanup_replaced_model_checkpoint(
    *,
    old_checkpoint: str,
    new_checkpoint: str,
    output_dir: Path,
    keep_candidate_models: bool,
    keep_proposal_grpo_checkpoints: bool,
    protected_checkpoints: Sequence[str] = (),
) -> List[str]:
    return CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=keep_candidate_models,
        keep_proposal_grpo_checkpoints=keep_proposal_grpo_checkpoints,
    ).cleanup_replaced_checkpoint(
        old_checkpoint=old_checkpoint,
        new_checkpoint=new_checkpoint,
        protected_checkpoints=protected_checkpoints,
    )


# --- from candidate_data.py ---
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from self.core.composition import build_exact_pair_dataset, compose_pseudo_examples
from self.core.models import CandidateWorkItem, ExactPairDataset, proposal_from_payload
from self.core.worker_io import write_json
from self.core.data_io import ensure_dir, save_examples
from self.core.evaluation import generate_prediction_map, resolve_max_new_tokens
from self.core.training import TrainingConfig

JsonDict = Dict[str, Any]


def candidate_action_key(proposal: Any) -> tuple[Any, ...]:
    left = int(getattr(proposal, "left"))
    right = int(getattr(proposal, "right"))
    guard = str(getattr(proposal, "guard", "none"))
    target = int(getattr(proposal, "target", left + right))
    code = getattr(proposal, "code", None)
    if code is not None:
        return (
            "executable",
            left,
            right,
            guard,
            target,
            str(getattr(proposal, "condition", "config")),
            str(code),
        )
    return ("config", left, right, guard, target)


def examples_by_key(task: Any, examples: Sequence[Any]) -> dict[Any, Any]:
    by_key: dict[Any, Any] = {}
    for example in examples:
        by_key.setdefault(task.key_for_example(example), example)
    return by_key


def stratified_eval_subset(
    *,
    task: Any,
    examples: Sequence[Any],
    per_size: int,
) -> List[Any]:
    """Return a stable per-size prefix so screening is nested in full eval."""

    counts: dict[int, int] = {}
    subset: List[Any] = []
    for example in examples:
        size = int(task.size_of(example))
        count = counts.get(size, 0)
        if count >= per_size:
            continue
        subset.append(example)
        counts[size] = count + 1
    return subset


def _limited_composed_dataset(
    *,
    task: Any,
    composed: ExactPairDataset,
    example_limit: int | None,
) -> ExactPairDataset:
    if example_limit is None or example_limit >= len(composed.examples):
        return composed
    examples = list(composed.examples[: max(0, example_limit)])
    keys = {task.key_for_example(example) for example in examples}
    component_map = {
        key: children
        for key, children in composed.component_map.items()
        if key in keys
    }
    diagnostics = dict(composed.diagnostics)
    diagnostics.update(
        {
            "full_retained": len(composed.examples),
            "rollout_retained": len(examples),
        }
    )
    return ExactPairDataset(
        examples=examples,
        component_map=component_map,
        keys=keys,
        diagnostics=diagnostics,
    )


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
    skipped_duplicates: List[JsonDict] = []
    kept_by_action: dict[tuple[Any, ...], JsonDict] = {}
    for result in proposal_results:
        if not result.get("valid"):
            continue
        proposal_payload = result.get("parsed_proposal")
        if not isinstance(proposal_payload, dict):
            continue
        proposal = proposal_from_payload(proposal_payload)
        action_key = candidate_action_key(proposal)
        action_key_payload = list(action_key)
        try:
            proposal_index = int(result["proposal_index"])
        except (KeyError, TypeError, ValueError):
            proposal_index = len(work_items) + len(skipped_duplicates)
        if isinstance(result, dict):
            result["candidate_dedup_action_key"] = action_key_payload
        kept = kept_by_action.get(action_key)
        if kept is not None:
            skipped = {
                "proposal_index": proposal_index,
                "id": result.get("id"),
                "kept_proposal_index": kept["proposal_index"],
                "kept_id": kept.get("id"),
                "action_key": action_key_payload,
                "parsed_proposal": proposal.to_json_dict(),
            }
            skipped_duplicates.append(skipped)
            if isinstance(result, dict):
                result["candidate_dedup_skipped"] = True
                result["candidate_dedup_reason"] = "duplicate_action"
                result["candidate_dedup_kept_proposal_index"] = kept["proposal_index"]
            continue
        kept_by_action[action_key] = {
            "proposal_index": proposal_index,
            "id": result.get("id"),
        }
        if isinstance(result, dict):
            result["candidate_dedup_skipped"] = False
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
    write_json(
        round_dir / "candidate_action_dedup.json",
        {
            "enabled": True,
            "valid_proposal_count": len(kept_by_action) + len(skipped_duplicates),
            "unique_action_count": len(kept_by_action),
            "skipped_duplicate_count": len(skipped_duplicates),
            "skipped_duplicates": skipped_duplicates,
        },
    )
    write_json(round_dir / "proposal_results.json", proposal_results)
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
    example_limit: int | None = None,
    fidelity: str = "candidate",
) -> List[CandidateWorkItem]:
    if not work_items:
        return []
    if example_limit is None and getattr(args, "candidate_training_mode", "single_stage") == "two_stage":
        example_limit = int(args.rollout_train_per_size)
    composed_views = {
        item.index: _limited_composed_dataset(
            task=task,
            composed=item.composed,
            example_limit=example_limit,
        )
        for item in work_items
    }
    source_by_key = examples_by_key(task, source_examples)
    needed_keys: set[Any] = set()
    for item in work_items:
        for children in composed_views[item.index].component_map.values():
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
            "fidelity": fidelity,
            "example_limit": example_limit,
        },
    )

    updated: List[CandidateWorkItem] = []
    for item in work_items:
        composed_view = composed_views[item.index]
        pseudo_examples, pseudo_diagnostics = compose_pseudo_examples(
            task_name=args.task,
            task=task,
            proposal=item.proposal,
            composed_examples=composed_view.examples,
            component_map=composed_view.component_map,
            component_predictions=component_predictions,
            args=args,
        )
        pseudo_diagnostics = {
            **dict(pseudo_diagnostics),
            "fidelity": fidelity,
            "full_composed_count": len(item.composed.examples),
            "composed_count_used": len(composed_view.examples),
        }
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


# --- from candidate_rewards.py ---
import argparse
import math
from pathlib import Path
from typing import Mapping, Sequence

from self.core.models import CandidateMetrics, CandidateWorkItem


def static_frontier_sizes(args: argparse.Namespace) -> list[int]:
    return list(range(int(args.frontier_min_size), int(args.frontier_max_size) + 1))


def mean_accuracy_for_sizes(per_size_accuracy: Mapping[int, float], sizes: Sequence[int]) -> float:
    if not sizes:
        return math.nan
    total = 0.0
    for size in sizes:
        value = per_size_accuracy.get(int(size), 0.0)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if not math.isfinite(numeric):
            numeric = 0.0
        total += numeric
    return total / len(sizes)


def build_no_pseudo_candidate_metrics(
    *,
    args: argparse.Namespace,
    item: CandidateWorkItem,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
) -> CandidateMetrics:
    current_target_accuracy = float(current_per_size_accuracy.get(item.proposal.target, 0.0))
    current_frontier_accuracy = mean_accuracy_for_sizes(current_per_size_accuracy, static_frontier_sizes(args))
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=False,
        reward=float("-inf"),
        frontier_delta=float("-inf"),
        target_accuracy=math.nan,
        current_target_accuracy=current_target_accuracy,
        final_accuracy=math.nan,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=math.nan,
        per_size_accuracy={},
        pseudo_count=0,
        model_dir=None,
        failure_reason="no pseudo labels retained",
        proposal_trace_replay_count=0,
        candidate_proposal_trace_count=0,
        outcome_trace_replay_count=0,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=math.nan,
        target_delta=math.nan,
        frontier_accuracy=math.nan,
        current_frontier_accuracy=current_frontier_accuracy,
        proposal_prediction=dict(item.proposal_prediction),
    )


def build_trained_candidate_metrics(
    *,
    args: argparse.Namespace,
    item: CandidateWorkItem,
    final_accuracy: float,
    per_size_accuracy: Mapping[int, float],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    model_dir: Path,
    proposal_trace_replay_count: int,
    candidate_proposal_trace_count: int,
    outcome_trace_replay_count: int,
) -> CandidateMetrics:
    target_accuracy = float(per_size_accuracy.get(item.proposal.target, 0.0))
    current_target_accuracy = float(current_per_size_accuracy.get(item.proposal.target, 0.0))
    target_delta = target_accuracy - current_target_accuracy
    frontier_sizes = static_frontier_sizes(args)
    frontier_accuracy = mean_accuracy_for_sizes(per_size_accuracy, frontier_sizes)
    current_frontier_accuracy = mean_accuracy_for_sizes(current_per_size_accuracy, frontier_sizes)
    frontier_delta = frontier_accuracy - current_frontier_accuracy
    final_accuracy_delta = final_accuracy - init_final_accuracy
    final_accuracy_delta_from_current = final_accuracy - current_final_accuracy
    per_size_delta = {
        int(size): float(per_size_accuracy.get(int(size), 0.0)) - float(current_per_size_accuracy.get(int(size), 0.0))
        for size in sorted({int(size) for size in per_size_accuracy} | {int(size) for size in current_per_size_accuracy})
    }
    reward = final_accuracy_delta_from_current
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=True,
        reward=reward,
        frontier_delta=frontier_delta,
        frontier_accuracy=frontier_accuracy,
        current_frontier_accuracy=current_frontier_accuracy,
        target_accuracy=target_accuracy,
        current_target_accuracy=current_target_accuracy,
        target_delta=target_delta,
        final_accuracy=final_accuracy,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=final_accuracy_delta,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=final_accuracy_delta_from_current,
        per_size_accuracy={int(size): float(value) for size, value in per_size_accuracy.items()},
        per_size_delta=per_size_delta,
        pseudo_count=len(item.pseudo_examples),
        model_dir=model_dir,
        proposal_trace_replay_count=proposal_trace_replay_count,
        candidate_proposal_trace_count=candidate_proposal_trace_count,
        outcome_trace_replay_count=outcome_trace_replay_count,
        proposal_prediction=dict(item.proposal_prediction),
    )


# --- from training.py ---
import argparse
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import torch

from self.core import worker_io
from self.core.evaluation import evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.model_io import instantiate_model_and_tokenizer
from self.core.training import (
    CausalLMDataCollator,
    TrainingConfig,
    TokenizedPromptTargetDataset,
    build_trainer,
    make_training_args,
)
from self.core.recipes import recipe_enabled


def make_config(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )


def train_checkpoint(
    *,
    source_checkpoint: str,
    train_examples: Sequence[Any],
    output_dir: Path,
    task: Any,
    args: argparse.Namespace,
    config: TrainingConfig,
    seed: int,
    recipe_phase_name: str = "self_improve",
    model_bootstrap_cache: ModelBootstrapCache | None = None,
) -> Tuple[Any, Any, Path]:
    model, tokenizer = instantiate_model_and_tokenizer(
        source_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=args.init_from_scratch if source_checkpoint == args.model_name else False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
        bootstrap_cache=model_bootstrap_cache,
    )
    use_recipe = recipe_enabled(str(args.recipe))
    data_collator = CausalLMDataCollator(tokenizer)
    if use_recipe:
        from self.core.recipes import PaddingAwareCausalLMDataCollator

        data_collator = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")
    train_dataset = TokenizedPromptTargetDataset(train_examples, tokenizer)

    def size_for_training_batch(example: Any) -> int:
        size_for_batching = getattr(example, "size_for_batching", None)
        if callable(size_for_batching):
            return int(size_for_batching())
        return int(task.size_of(example))

    training_args = make_training_args(
        output_dir,
        config,
        bf16=args.bf16,
        fp16=args.fp16,
        skip_save=True,
        keep_checkpoints=False,
        seed=seed,
        recipe=args.recipe,
        recipe_phase_name=recipe_phase_name,
    )
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        seed=seed,
        size_getter=size_for_training_batch,
        bucket_train_batches_by_size=args.bucket_train_batches_by_size,
        recipe=args.recipe,
        recipe_phase_name=recipe_phase_name,
    )
    trainer.train()
    model = trainer.model
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    return model, tokenizer, model_dir


def evaluate_model(
    *,
    model: Any,
    tokenizer: Any,
    task: Any,
    examples: Sequence[Any],
    batch_size: int,
    decode_max_new_tokens: int,
) -> Tuple[float, Dict[int, float]]:
    max_tokens = resolve_max_new_tokens(examples, decode_max_new_tokens)
    if model is None or tokenizer is None:
        raise ValueError("model and tokenizer are required for candidate evaluation.")
    return evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=batch_size,
        max_new_tokens=max_tokens,
        size_getter=task.size_of,
        prediction_parser=task.prediction_parser,
    )


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- from candidate_training_mix.py ---
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Sequence

from self.core import worker_io
from self.core.data_io import save_examples
from self.adaptive.traces import (
    OutcomeTraceExample,
    ProposalTraceExample,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.adaptive.traces import build_candidate_proposal_trace_example
from self.core.models import CandidateWorkItem
from self.adaptive.proposal import write_trace_jsonl
from self.adaptive.proposal import PromptBundle


@dataclass(frozen=True)
class CandidateTrainingMix:
    task_train_examples: List[Any]
    outcome_replay_examples: List[OutcomeTraceExample]
    candidate_trace_examples: List[ProposalTraceExample]
    mixed_proposal_replay_examples: List[ProposalTraceExample]
    mixed_candidate_trace_examples: List[ProposalTraceExample]
    train_examples: List[Any]

    @property
    def summary_counts(self) -> dict[str, int]:
        return {
            "task_train_examples": len(self.task_train_examples),
            "outcome_trace_replay_examples": len(self.outcome_replay_examples),
            "proposal_trace_replay_examples": len(self.mixed_proposal_replay_examples),
            "candidate_proposal_trace_examples": len(self.candidate_trace_examples),
            "mixed_candidate_proposal_trace_examples": len(self.mixed_candidate_trace_examples),
            "total_train_examples": len(self.train_examples),
        }


def build_candidate_training_mix(
    *,
    args: argparse.Namespace,
    source_examples: Sequence[Any],
    item: CandidateWorkItem,
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    seed: int,
    random_cls: Callable[[int], random.Random] = random.Random,
) -> CandidateTrainingMix:
    # Candidate task updates must train only on self-labeled composed data.
    # The source pool is used to build compositions, not replayed as supervised
    # labels after seed training.
    task_train_examples = list(item.pseudo_examples)
    two_stage = getattr(args, "candidate_training_mode", "single_stage") == "two_stage"
    outcome_replay_examples = [] if two_stage else sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcome_trace_buffer,
        task_train_count=len(task_train_examples),
        rng=random_cls(seed + 6151),
    )
    candidate_trace_examples: List[ProposalTraceExample] = []
    if not two_stage and item.completion and args.proposal_trace_replay_ratio > 0.0:
        candidate_trace_examples.append(
            build_candidate_proposal_trace_example(
                task_name=args.task,
                condition=args.condition,
                round_index=round_index,
                prompt=proposal_prompt,
                item=item,
            )
        )
    mixed_proposal_replay_examples: List[ProposalTraceExample] = []
    mixed_proposal_replay_examples = [] if two_stage else sample_proposal_trace_replay(
        args=args,
        trace_buffer=proposal_trace_buffer,
        task_train_count=len(task_train_examples),
        rng=random_cls(seed + 7919),
    )
    mixed_candidate_trace_examples = list(candidate_trace_examples)
    train_examples = (
        task_train_examples
        + list(outcome_replay_examples)
        + mixed_proposal_replay_examples
        + mixed_candidate_trace_examples
    )
    return CandidateTrainingMix(
        task_train_examples=task_train_examples,
        outcome_replay_examples=list(outcome_replay_examples),
        candidate_trace_examples=candidate_trace_examples,
        mixed_proposal_replay_examples=mixed_proposal_replay_examples,
        mixed_candidate_trace_examples=mixed_candidate_trace_examples,
        train_examples=train_examples,
    )


def write_candidate_training_mix_artifacts(
    *,
    candidate_dir: Path,
    task: Any,
    args: argparse.Namespace,
    source_examples: Sequence[Any],
    item: CandidateWorkItem,
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    mix: CandidateTrainingMix,
    save_examples_fn: Callable[..., None] = save_examples,
    write_trace_jsonl_fn: Callable[..., None] = write_trace_jsonl,
    write_json_fn: Callable[[Path, Any], None] = worker_io.write_json,
) -> None:
    save_examples_fn(candidate_dir / "train_examples.jsonl", mix.task_train_examples, task.serialize_example)
    if mix.outcome_replay_examples:
        write_trace_jsonl_fn(
            candidate_dir / "outcome_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in mix.outcome_replay_examples],
        )
    if mix.mixed_proposal_replay_examples:
        write_trace_jsonl_fn(
            candidate_dir / "proposal_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in mix.mixed_proposal_replay_examples],
        )
    if mix.candidate_trace_examples:
        write_trace_jsonl_fn(
            candidate_dir / "candidate_proposal_trace_example.jsonl",
            [example.to_json_dict() for example in mix.candidate_trace_examples],
        )
    write_json_fn(
        candidate_dir / "train_mix_summary.json",
        {
            **mix.summary_counts,
            "source_examples": len(source_examples),
            "source_examples_used_for_task_training": 0,
            "task_supervision_source": "pseudo_examples_only",
            "pseudo_examples": len(item.pseudo_examples),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "outcome_trace_target_mode": args.outcome_trace_target_mode,
            "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
            "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
            "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
        },
    )


# --- from candidate_scoring.py ---
import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from self.core import worker_io
from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle
from self.core.training import TrainingConfig


def train_and_score_candidate(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    item: CandidateWorkItem,
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    seed: int,
    model_bootstrap_cache: ModelBootstrapCache | None = None,
) -> CandidateMetrics:
    candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
    if not item.pseudo_examples:
        metrics = build_no_pseudo_candidate_metrics(
            args=args,
            item=item,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
        )
        worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
        return metrics

    training_mix = build_candidate_training_mix(
        args=args,
        source_examples=source_examples,
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        seed=seed,
    )
    write_candidate_training_mix_artifacts(
        candidate_dir=candidate_dir,
        task=task,
        args=args,
        source_examples=source_examples,
        item=item,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        mix=training_mix,
    )
    model, tokenizer, task_model_dir = train_checkpoint(
        source_checkpoint=current_checkpoint,
        train_examples=training_mix.train_examples,
        output_dir=candidate_dir / "training",
        task=task,
        args=args,
        config=config,
        seed=seed,
        recipe_phase_name="self_improve",
        model_bootstrap_cache=model_bootstrap_cache,
    )
    model_dir = task_model_dir
    eval_start = time.monotonic()
    final_accuracy, per_size_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=eval_examples,
        batch_size=config.per_device_eval_batch_size,
        decode_max_new_tokens=config.decode_max_new_tokens,
    )
    worker_io.write_json(
        candidate_dir / "candidate_eval_summary.json",
        {
            "backend": "transformers",
            "examples": len(eval_examples),
            "batch_size": config.per_device_eval_batch_size,
            "decode_max_new_tokens": config.decode_max_new_tokens,
            "resolved_max_new_tokens": resolve_max_new_tokens(eval_examples, config.decode_max_new_tokens),
            "runtime_seconds": time.monotonic() - eval_start,
            "model_dir": str(model_dir),
        },
    )
    metrics = build_trained_candidate_metrics(
        args=args,
        item=item,
        final_accuracy=final_accuracy,
        per_size_accuracy=per_size_accuracy,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        model_dir=model_dir,
        proposal_trace_replay_count=len(training_mix.mixed_proposal_replay_examples),
        candidate_proposal_trace_count=len(training_mix.candidate_trace_examples),
        outcome_trace_replay_count=len(training_mix.outcome_replay_examples),
    )
    worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
    deleted_after_scoring = CheckpointManager(
        output_dir=round_dir.parent,
        keep_candidate_models=bool(getattr(args, "keep_all_candidate_models", False)),
        keep_proposal_grpo_checkpoints=bool(getattr(args, "keep_all_proposal_grpo_checkpoints", False)),
    ).cleanup_unselectable_candidate(
        metric=metrics,
        min_reward=float(getattr(args, "selection_min_reward", 0.0)),
    )
    if deleted_after_scoring:
        worker_io.write_json(
            candidate_dir / "deleted_model_dirs.json",
            {
                "stage": "after_candidate_scoring",
                "reason": "unselectable",
                "selection_min_reward": float(getattr(args, "selection_min_reward", 0.0)),
                "deleted_model_dirs": deleted_after_scoring,
            },
        )
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    clear_cuda_cache()
    return metrics


@dataclass(frozen=True)
class SelectedConfirmationResult:
    """Full-fidelity result for a provisionally selected screen candidate."""

    metrics: CandidateMetrics
    work_item: CandidateWorkItem
    accepted: bool
    timings: Mapping[str, float]


def confirm_two_stage_candidate(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    provisional: CandidateMetrics,
    work_item: CandidateWorkItem,
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    seed: int,
) -> SelectedConfirmationResult:
    """Retrain one provisional winner from scratch and confirm on full eval."""

    confirmation_dir = round_dir / "selected_confirmation"
    ensure_dir(confirmation_dir)
    timings: Dict[str, float] = {}
    phase_start = time.monotonic()
    current_model, current_tokenizer = instantiate_model_and_tokenizer(
        current_checkpoint,
        bf16=args.bf16,
        fp16=args.fp16,
        init_from_scratch=False,
        tokenizer_mode=args.tokenizer_mode,
        recipe=args.recipe,
    )
    try:
        confirmed_items = attach_pseudo_labels(
            args=args,
            task=task,
            round_dir=confirmation_dir,
            work_items=[work_item],
            source_examples=source_examples,
            current_model=current_model,
            current_tokenizer=current_tokenizer,
            config=config,
            example_limit=int(args.candidate_train_per_size),
            fidelity="confirmed_full",
        )
    finally:
        del current_model
        del current_tokenizer
        clear_cuda_cache()
    timings["pseudolabel_seconds"] = time.monotonic() - phase_start
    confirmed_item = confirmed_items[0]

    if not confirmed_item.pseudo_examples:
        metrics = build_no_pseudo_candidate_metrics(
            args=args,
            item=confirmed_item,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
        )
        write_json(
            confirmation_dir / "selected_confirmation.json",
            {
                "accepted": False,
                "reason": "no_pseudo_labels_retained",
                "provisional_metrics": provisional.to_json_dict(),
                "confirmed_metrics": metrics.to_json_dict(),
                "timings": timings,
            },
        )
        return SelectedConfirmationResult(
            metrics=metrics,
            work_item=confirmed_item,
            accepted=False,
            timings=timings,
        )

    selected_max_steps = int(getattr(args, "selected_max_steps", 0))
    selected_config = replace(
        config,
        num_epochs=int(args.num_epochs),
        max_steps=selected_max_steps if selected_max_steps > 0 else None,
    )
    write_json(
        confirmation_dir / "training_plan.json",
        {
            "source_checkpoint": current_checkpoint,
            "fresh_from_pre_attempt_checkpoint": True,
            "pseudo_example_count": len(confirmed_item.pseudo_examples),
            "num_epochs": selected_config.num_epochs,
            "max_steps": selected_config.max_steps,
            "learning_rate": selected_config.learning_rate,
            "per_device_train_batch_size": selected_config.per_device_train_batch_size,
            "gradient_accumulation_steps": selected_config.gradient_accumulation_steps,
        },
    )
    phase_start = time.monotonic()
    model, tokenizer, model_dir = train_checkpoint(
        source_checkpoint=current_checkpoint,
        train_examples=confirmed_item.pseudo_examples,
        output_dir=confirmation_dir / "training",
        task=task,
        args=args,
        config=selected_config,
        seed=seed,
        recipe_phase_name="self_improve",
    )
    timings["training_seconds"] = time.monotonic() - phase_start
    phase_start = time.monotonic()
    final_accuracy, per_size_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=eval_examples,
        batch_size=selected_config.per_device_eval_batch_size,
        decode_max_new_tokens=selected_config.decode_max_new_tokens,
    )
    timings["evaluation_seconds"] = time.monotonic() - phase_start
    metrics = build_trained_candidate_metrics(
        args=args,
        item=confirmed_item,
        final_accuracy=final_accuracy,
        per_size_accuracy=per_size_accuracy,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        model_dir=model_dir,
        proposal_trace_replay_count=0,
        candidate_proposal_trace_count=0,
        outcome_trace_replay_count=0,
    )
    min_reward = float(getattr(args, "selection_min_reward", 0.0))
    accepted = bool(metrics.valid and math.isfinite(metrics.reward) and metrics.reward > min_reward)
    timings["total_seconds"] = sum(timings.values())
    write_json(
        confirmation_dir / "selected_confirmation.json",
        {
            "accepted": accepted,
            "acceptance_rule": "confirmed_reward_strictly_greater_than_selection_min_reward",
            "selection_min_reward": min_reward,
            "provisional_metrics": provisional.to_json_dict(),
            "confirmed_metrics": metrics.to_json_dict(),
            "screen_full_reward_disagreement": bool(
                provisional.reward > min_reward and metrics.reward <= min_reward
            ),
            "timings": timings,
        },
    )
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    clear_cuda_cache()
    if not accepted:
        shutil.rmtree(model_dir, ignore_errors=True)
    return SelectedConfirmationResult(
        metrics=metrics,
        work_item=confirmed_item,
        accepted=accepted,
        timings=timings,
    )


# --- from candidate_selection.py ---
from typing import Optional, Sequence

from self.core.models import CandidateMetrics


def select_candidate(metrics: Sequence[CandidateMetrics], min_reward: float) -> Optional[CandidateMetrics]:
    eligible = [metric for metric in metrics if metric.valid and metric.reward >= min_reward]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda metric: (
            metric.reward,
            metric.frontier_delta,
            metric.target_delta,
            metric.final_accuracy_delta_from_current,
        ),
        reverse=True,
    )[0]


# --- from candidate_workers.py ---
"""Candidate worker payloads, specs, and local/Slurm execution."""

# --- from candidate_worker_payloads.py ---
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


# --- from candidate_worker_inputs.py ---
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, MutableMapping, Optional, Sequence

import torch

from self.core.data_io import load_examples
from self.adaptive.traces import outcome_trace_from_json, proposal_trace_from_json
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class CandidateWorkerRuntimeDeps:
    load_json: Callable[[Path], Any]
    namespace_from_json_args: Callable[[Any], argparse.Namespace]
    normalize_args: Callable[[argparse.Namespace], argparse.Namespace]
    task_for_name: Callable[[str], Any]
    make_config: Callable[[argparse.Namespace], Any]
    load_trace_jsonl: Callable[[Path, Any], list[Any]]
    train_and_score_candidate: Callable[..., CandidateMetrics]
    write_json: Callable[[Path, Any], None]


@dataclass(frozen=True)
class CandidateWorkerSharedInputs:
    args: argparse.Namespace
    task: Any
    config: Any
    source_examples: Sequence[Any]
    eval_examples: Sequence[Any]
    proposal_trace_buffer: Sequence[Any]
    outcome_trace_buffer: Sequence[Any]
    proposal_prompt: PromptBundle
    model_bootstrap_cache: Optional[ModelBootstrapCache]


SharedInputCache = MutableMapping[str, CandidateWorkerSharedInputs]


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _shared_input_cache_key(payload: JsonDict) -> str:
    return _stable_json(
        {
            "args": payload.get("args"),
            "source_examples_path": payload.get("source_examples_path"),
            "eval_examples_path": payload.get("eval_examples_path"),
            "proposal_trace_buffer_path": payload.get("proposal_trace_buffer_path"),
            "outcome_trace_buffer_path": payload.get("outcome_trace_buffer_path"),
            "proposal_prompt_path": payload.get("proposal_prompt_path"),
        }
    )


def load_candidate_worker_shared_inputs(
    payload: JsonDict,
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    shared_cache: Optional[SharedInputCache],
) -> CandidateWorkerSharedInputs:
    cache_key = _shared_input_cache_key(payload)
    if shared_cache is not None and cache_key in shared_cache:
        return shared_cache[cache_key]

    args = deps.namespace_from_json_args(payload["args"])
    args.run_candidate_worker = True
    args.candidate_worker_spec = spec_path
    args = deps.normalize_args(args)
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        args.bf16 = True
        print("[INFO] Worker defaulting to bf16 on CUDA.", flush=True)
    task = deps.task_for_name(args.task)
    task.validate_args(args)
    config = deps.make_config(args)
    source_examples = load_examples(Path(payload["source_examples_path"]), task.deserialize_example)
    eval_examples = load_examples(Path(payload["eval_examples_path"]), task.deserialize_example)
    proposal_trace_buffer = deps.load_trace_jsonl(
        Path(payload["proposal_trace_buffer_path"]),
        proposal_trace_from_json,
    )
    outcome_trace_buffer = deps.load_trace_jsonl(
        Path(payload["outcome_trace_buffer_path"]),
        outcome_trace_from_json,
    )
    prompt_payload = deps.load_json(Path(payload["proposal_prompt_path"]))
    shared = CandidateWorkerSharedInputs(
        args=args,
        task=task,
        config=config,
        source_examples=source_examples,
        eval_examples=eval_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=PromptBundle(
            system=str(prompt_payload.get("system", "")),
            user=str(prompt_payload.get("user", "")),
        ),
        model_bootstrap_cache=_make_model_bootstrap_cache(args, shared_cache=shared_cache),
    )
    if shared_cache is not None:
        shared_cache[cache_key] = shared
    return shared


def _make_model_bootstrap_cache(
    args: argparse.Namespace,
    *,
    shared_cache: Optional[SharedInputCache],
) -> Optional[ModelBootstrapCache]:
    cache_base_state = bool(getattr(args, "candidate_local_cache_base_state", False))
    if shared_cache is not None or cache_base_state:
        return ModelBootstrapCache(cache_base_state=cache_base_state)
    return None


def candidate_item_from_payload(payload: JsonDict, pseudo_examples: Sequence[Any]) -> CandidateWorkItem:
    candidate_payload = dict(payload["candidate"])
    return candidate_payload_to_work_item(
        payload=candidate_payload,
        pseudo_examples=pseudo_examples,
    )


# --- from workers.py ---
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from self.core.models import CandidateMetrics


JsonDict = Dict[str, Any]


def run_candidate_worker_pack_from_spec(
    pack_spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    run_from_spec_fn: Callable[..., CandidateMetrics],
) -> JsonDict:
    payload = deps.load_json(pack_spec_path)
    spec_paths: Sequence[Path] = [Path(str(path)) for path in payload.get("spec_paths", [])]
    results: list[JsonDict] = []
    failed = 0
    shared_cache: SharedInputCache = {}
    runner_accepts_cache = _run_from_spec_accepts_shared_cache(run_from_spec_fn)
    for spec_path in spec_paths:
        try:
            if runner_accepts_cache:
                metrics = run_from_spec_fn(spec_path, shared_cache=shared_cache)
            else:
                metrics = run_from_spec_fn(spec_path)
            results.append(
                {
                    "spec_path": str(spec_path),
                    "status": "ok",
                    "candidate_index": metrics.index,
                }
            )
        except Exception as exc:
            failed += 1
            try:
                failure_payload = write_candidate_worker_failure_from_spec(
                    spec_path,
                    exc,
                    load_json_fn=deps.load_json,
                    write_json_fn=deps.write_json,
                )
            except Exception:
                failure_payload = {
                    "spec_path": str(spec_path),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            results.append(
                {
                    "spec_path": str(spec_path),
                    "status": "failed",
                    "failure": failure_payload,
                }
            )
            print(f"[ERROR] Packed candidate worker failed for {spec_path}: {exc}", flush=True)
    return {
        "pack_spec_path": str(pack_spec_path),
        "total": len(spec_paths),
        "succeeded": len(spec_paths) - failed,
        "failed": failed,
        "results": results,
        "shared_input_cache_entries": len(shared_cache),
        "model_bootstrap_cache": [
            shared.model_bootstrap_cache.stats()
            for shared in shared_cache.values()
            if shared.model_bootstrap_cache is not None
        ],
        "model_bootstrap_cache_details": [
            shared.model_bootstrap_cache.detailed_stats()
            for shared in shared_cache.values()
            if shared.model_bootstrap_cache is not None
        ],
    }


def _run_from_spec_accepts_shared_cache(run_from_spec_fn: Callable[..., CandidateMetrics]) -> bool:
    try:
        signature = inspect.signature(run_from_spec_fn)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "shared_cache":
            return True
    return False


# --- from workers.py ---
import copy
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import set_seed

from self.core.data_io import load_examples
from self.core.models import CandidateMetrics, float_or_nan


JsonDict = Dict[str, Any]


def run_candidate_worker_from_spec(
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    shared_cache: Optional[SharedInputCache] = None,
) -> CandidateMetrics:
    payload = deps.load_json(spec_path)
    shared = load_candidate_worker_shared_inputs(
        payload,
        spec_path,
        deps=deps,
        shared_cache=shared_cache,
    )
    args = copy.copy(shared.args)
    args.candidate_worker_spec = spec_path
    seed = int(payload["seed"])
    set_seed(seed)
    pseudo_examples = load_examples(Path(payload["pseudo_examples_path"]), shared.task.deserialize_example)
    item = candidate_item_from_payload(payload, pseudo_examples)
    current_per_size_accuracy = {
        int(size): float(score)
        for size, score in dict(payload.get("current_per_size_accuracy", {})).items()
        if score is not None
    }
    return deps.train_and_score_candidate(
        args=args,
        task=shared.task,
        current_checkpoint=str(payload["current_checkpoint"]),
        source_examples=shared.source_examples,
        proposal_trace_buffer=shared.proposal_trace_buffer,
        outcome_trace_buffer=shared.outcome_trace_buffer,
        proposal_prompt=shared.proposal_prompt,
        round_index=int(payload["round_index"]),
        item=item,
        round_dir=Path(payload["round_dir"]),
        eval_examples=shared.eval_examples,
        current_final_accuracy=float_or_nan(payload.get("current_final_accuracy")),
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=float_or_nan(payload.get("init_final_accuracy")),
        config=shared.config,
        seed=seed,
        model_bootstrap_cache=shared.model_bootstrap_cache,
    )


def run_candidate_worker(
    spec_path: Path,
    *,
    deps: CandidateWorkerRuntimeDeps,
    run_from_spec_fn: Callable[[Path], CandidateMetrics],
) -> JsonDict:
    try:
        metrics = run_from_spec_fn(spec_path)
        return metrics.to_json_dict()
    except Exception as exc:
        try:
            write_candidate_worker_failure_from_spec(
                spec_path,
                exc,
                load_json_fn=deps.load_json,
                write_json_fn=deps.write_json,
            )
        except Exception:
            print(f"[ERROR] Candidate worker failed before failure artifact could be written: {exc}", flush=True)
        raise


# --- from candidate_local_workers.py ---
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from self.core import worker_io
from self.core.data_io import ensure_dir, sanitize_json_value


JsonDict = Dict[str, Any]
CollectMetricsFn = Callable[..., List[Any]]
PreparePackSpecsFn = Callable[..., List[Tuple[int, List[Any], Path]]]


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


def train_candidates_local_parallel_from_specs(
    *,
    args: Any,
    round_dir: Path,
    work_items: Sequence[Any],
    spec_paths: Sequence[Path],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    collect_metrics_fn: CollectMetricsFn,
    prepare_pack_specs_fn: PreparePackSpecsFn,
    subprocess_module: Any = subprocess,
    executable: str | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[Any]:
    job_dir = round_dir / "candidate_jobs"
    logs_dir = job_dir / "logs"
    ensure_dir(logs_dir)
    max_parallel = max(1, int(args.candidate_local_parallelism))
    pack_size = max(1, int(getattr(args, "candidate_local_pack_size", 1)))
    if pack_size == 1:
        pending = [
            {
                "label": f"candidate-{item.index:02d}",
                "items": [item],
                "spec_path": spec_path,
                "is_pack": False,
            }
            for item, spec_path in zip(work_items, spec_paths)
        ]
    else:
        pending = [
            {
                "label": f"pack-{pack_index:02d}",
                "items": chunk_items,
                "spec_path": pack_path,
                "is_pack": True,
            }
            for pack_index, chunk_items, pack_path in prepare_pack_specs_fn(
                round_dir=round_dir,
                work_items=work_items,
                spec_paths=spec_paths,
                pack_size=pack_size,
            )
        ]
    dispatch_plan = local_candidate_dispatch_plan(
        args=args,
        candidate_count=len(spec_paths),
        process_units=pending,
        max_parallel=max_parallel,
        pack_size=pack_size,
    )
    active: List[Tuple[JsonDict, subprocess.Popen[Any], Any, Any]] = []
    launched: List[JsonDict] = []
    start = monotonic_fn()
    executable = executable or sys.executable
    print(
        f"[INFO] Running {len(spec_paths)} local candidate workers as {len(pending)} process(es) "
        f"(max_parallel={max_parallel}, pack_size={pack_size}).",
        flush=True,
    )
    try:
        while pending or active:
            while pending and len(active) < max_parallel:
                unit = pending.pop(0)
                stdout_path = logs_dir / f"candidate-local-{unit['label']}.out"
                stderr_path = logs_dir / f"candidate-local-{unit['label']}.err"
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                stderr_handle = stderr_path.open("w", encoding="utf-8")
                if unit["is_pack"]:
                    command = [
                        executable,
                        "-m",
                        "self.adaptive.driver",
                        "--run-candidate-pack-worker",
                        "--candidate-worker-pack-spec",
                        str(unit["spec_path"]),
                    ]
                else:
                    command = [
                        executable,
                        "-m",
                        "self.adaptive.driver",
                        "--run-candidate-worker",
                        "--candidate-worker-spec",
                        str(unit["spec_path"]),
                    ]
                process = subprocess_module.Popen(command, stdout=stdout_handle, stderr=stderr_handle)
                active.append((unit, process, stdout_handle, stderr_handle))
                launched.append(
                    {
                        "label": unit["label"],
                        "candidate_indices": [item.index for item in unit["items"]],
                        "spec_path": str(unit["spec_path"]),
                        "is_pack": bool(unit["is_pack"]),
                        "pid": process.pid,
                        "command": command,
                        "stdout_path": str(stdout_path),
                        "stderr_path": str(stderr_path),
                    }
                )
            write_json(
                job_dir / "local_dispatch.json",
                {
                    **dispatch_plan,
                    "max_parallel": max_parallel,
                    "pack_size": pack_size,
                    "launched": launched,
                    "active_pids": [process.pid for _, process, _, _ in active],
                    "pending": len(pending),
                },
            )
            next_active: List[Tuple[JsonDict, subprocess.Popen[Any], Any, Any]] = []
            for unit, process, stdout_handle, stderr_handle in active:
                returncode = process.poll()
                if returncode is None:
                    next_active.append((unit, process, stdout_handle, stderr_handle))
                    continue
                stdout_handle.close()
                stderr_handle.close()
                if returncode != 0:
                    for item in unit["items"]:
                        write_local_candidate_failure(
                            round_dir=round_dir,
                            item=item,
                            spec_path=Path(unit["spec_path"]),
                            returncode=returncode,
                            reason=f"candidate local worker {unit['label']} exited with code {returncode}",
                        )
            active = next_active
            done_count = sum(
                1
                for item in work_items
                if candidate_metric_path(round_dir, item).exists()
                or candidate_worker_failure_path(round_dir, item).exists()
            )
            print(
                f"[INFO] Local candidate workers: {done_count}/{len(work_items)} finished.",
                flush=True,
            )
            timeout_seconds = float(getattr(args, "candidate_worker_timeout_seconds", 0.0))
            if timeout_seconds > 0.0:
                elapsed = monotonic_fn() - start
                if elapsed >= timeout_seconds:
                    for unit, process, stdout_handle, stderr_handle in active:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        stdout_handle.close()
                        stderr_handle.close()
                        for item in unit["items"]:
                            write_local_candidate_failure(
                                round_dir=round_dir,
                                item=item,
                                spec_path=Path(unit["spec_path"]),
                                returncode=process.returncode,
                                reason=(
                                    f"candidate local worker {unit['label']} timed out after "
                                    f"{timeout_seconds} seconds"
                                ),
                            )
                    write_json(
                        job_dir / "local_timeout.json",
                        {
                            "elapsed_seconds": elapsed,
                            "timeout_seconds": timeout_seconds,
                            "max_parallel": max_parallel,
                            "pack_size": pack_size,
                        },
                    )
                    active = []
                    pending = []
                    break
            if pending or active:
                sleep_fn(float(getattr(args, "candidate_worker_poll_seconds", 5.0)))
    finally:
        for unit, process, stdout_handle, stderr_handle in active:
            if process.poll() is None:
                process.terminate()
            stdout_handle.close()
            stderr_handle.close()
    return collect_metrics_fn(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
    )


def local_candidate_dispatch_plan(
    *,
    args: Any,
    candidate_count: int,
    process_units: Sequence[JsonDict],
    max_parallel: int,
    pack_size: int,
) -> JsonDict:
    packed_workers = any(bool(unit.get("is_pack")) for unit in process_units)
    candidate_local_cache_base_state = bool(getattr(args, "candidate_local_cache_base_state", False))
    return {
        "candidate_count": int(candidate_count),
        "planned_processes": len(process_units),
        "max_parallel": int(max_parallel),
        "pack_size": int(pack_size),
        "packed_workers": packed_workers,
        "cache_plan": {
            "shared_input_cache": packed_workers,
            "tokenizer_bootstrap_cache": packed_workers or candidate_local_cache_base_state,
            "base_state_cache": candidate_local_cache_base_state,
        },
        "planned_units": [
            {
                "label": str(unit["label"]),
                "candidate_indices": [item.index for item in unit["items"]],
                "spec_path": str(unit["spec_path"]),
                "is_pack": bool(unit["is_pack"]),
            }
            for unit in process_units
        ],
    }


def write_local_candidate_failure(
    *,
    round_dir: Path,
    item: Any,
    spec_path: Path,
    returncode: Optional[int],
    reason: str,
) -> None:
    failure_path = candidate_worker_failure_path(round_dir, item)
    if failure_path.exists() or candidate_metric_path(round_dir, item).exists():
        return
    write_json(
        failure_path,
        {
            "spec_path": str(spec_path),
            "error_type": "LocalCandidateWorkerError",
            "error": reason,
            "returncode": returncode,
        },
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(sanitize_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


# --- from candidate_slurm_workers.py ---
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence

from self.core import worker_io
from self.core.data_io import ensure_dir


def candidate_metric_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_metric_path(round_dir, item.index)


def candidate_worker_failure_path(round_dir: Path, item: Any) -> Path:
    return worker_io.candidate_worker_failure_path(round_dir, item.index)


# --- from workers.py ---
import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from self.core import worker_io
from self.core.data_io import ensure_dir, save_examples
from self.adaptive.proposal import write_trace_jsonl
from self.adaptive.proposal import PromptBundle


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
            "candidate": candidate_payload_from_work_item(item),
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


# --- from workers.py ---
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.adaptive.proposal import PromptBundle


CollectMetricsFn = Callable[..., List[Any]]


def train_candidates_local_parallel(
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
    collect_metrics_fn: CollectMetricsFn,
) -> List[Any]:
    if not work_items:
        return []
    spec_paths = prepare_candidate_worker_specs(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
    )
    return train_candidates_local_parallel_from_specs(
        args=args,
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        collect_metrics_fn=collect_metrics_fn,
        prepare_pack_specs_fn=prepare_candidate_worker_pack_specs,
    )


_worker_train_candidates_local_parallel = train_candidates_local_parallel


# --- from candidate_dispatch.py ---
"""Candidate dispatch across serial, local-parallel, and Slurm modes."""

# --- from candidate_metric_collection.py ---
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from self.core import worker_io
import sys as _sys
workers = _sys.modules[__name__]
from self.core.models import (
    CandidateMetrics,
    CandidateWorkItem,
    candidate_metrics_from_json,
)


JsonDict = Dict[str, Any]


def _candidate_failure_metrics_impl(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    model_dir: Path | None = None,
) -> CandidateMetrics:
    return CandidateMetrics(
        index=item.index,
        row_id=item.row_id,
        proposal=item.proposal,
        valid=False,
        reward=float("-inf"),
        frontier_delta=float("-inf"),
        target_accuracy=math.nan,
        current_target_accuracy=float(
            current_per_size_accuracy.get(item.proposal.target, 0.0)
        ),
        final_accuracy=math.nan,
        init_final_accuracy=init_final_accuracy,
        final_accuracy_delta=math.nan,
        current_final_accuracy=current_final_accuracy,
        final_accuracy_delta_from_current=math.nan,
        per_size_accuracy={},
        pseudo_count=len(item.pseudo_examples),
        model_dir=model_dir,
        failure_reason=reason,
        proposal_prediction=dict(item.proposal_prediction),
    )


def _collect_candidate_worker_metrics_impl(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    failure_metrics_fn: Optional[Callable[..., CandidateMetrics]] = None,
) -> List[CandidateMetrics]:
    metrics: List[CandidateMetrics] = []
    failures: List[JsonDict] = []
    failure_metrics_fn = failure_metrics_fn or candidate_failure_metrics
    for item in work_items:
        metrics_path = workers.candidate_metric_path(round_dir, item)
        if metrics_path.exists():
            metrics.append(candidate_metrics_from_json(worker_io.load_json(metrics_path)))
            continue
        failure_path = workers.candidate_worker_failure_path(round_dir, item)
        if failure_path.exists():
            failure_payload = worker_io.load_json(failure_path)
            reason = str(failure_payload.get("error") or "candidate worker failed")
        else:
            reason = "candidate worker finished without candidate_metrics.json"
        candidate_dir = round_dir / "candidates" / f"candidate_{item.index:02d}"
        trained_model_dir = candidate_dir / "training" / "model"
        model_dir = trained_model_dir if trained_model_dir.exists() else None
        failure_metric = failure_metrics_fn(
            item=item,
            reason=reason,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            model_dir=model_dir,
        )
        worker_io.write_json(metrics_path, failure_metric.to_json_dict())
        metrics.append(failure_metric)
        failures.append(
            {
                "candidate_index": item.index,
                "failure_reason": reason,
                "metrics_path": str(metrics_path),
                "worker_failure_path": str(failure_path),
            }
        )
    if failures:
        worker_io.write_json(round_dir / "candidate_jobs" / "gather_failures.json", failures)
    return metrics


# --- from candidate_serial_runtime.py ---
import argparse
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.model_bootstrap_cache import ModelBootstrapCache
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle
from self.core.training import TrainingConfig


def train_candidates_serial(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    score_candidate_fn: Callable[..., CandidateMetrics],
) -> list[CandidateMetrics]:
    metrics: list[CandidateMetrics] = []
    model_bootstrap_cache = ModelBootstrapCache(
        cache_base_state=bool(getattr(args, "candidate_local_cache_base_state", False))
    )
    pass_model_bootstrap_cache = _score_candidate_accepts_model_bootstrap_cache(
        score_candidate_fn
    )
    for item in work_items:
        kwargs = dict(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            source_examples=source_examples,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_prompt=proposal_prompt,
            round_index=round_index,
            item=item,
            round_dir=round_dir,
            eval_examples=eval_examples,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            config=config,
            seed=args.seed + attempt_index * 1009 + item.index,
        )
        if pass_model_bootstrap_cache:
            kwargs["model_bootstrap_cache"] = model_bootstrap_cache
        metrics.append(score_candidate_fn(**kwargs))
    return metrics


def _score_candidate_accepts_model_bootstrap_cache(
    score_candidate_fn: Callable[..., CandidateMetrics],
) -> bool:
    try:
        signature = inspect.signature(score_candidate_fn)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "model_bootstrap_cache":
            return True
    return False


# --- from candidate_parallel_runtime.py ---
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import sys as _sys
workers = _sys.modules[__name__]
from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle


def train_candidates_local_parallel(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    collect_metrics_fn: Callable[..., list[CandidateMetrics]],
    subprocess_module: Any = None,
) -> list[CandidateMetrics]:
    if subprocess_module is not None:
        workers.subprocess = subprocess_module
    return _worker_train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
        collect_metrics_fn=collect_metrics_fn,
    )


# --- from py ---
import argparse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle
from self.core.training import TrainingConfig


def candidate_failure_metrics(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    model_dir: Path | None = None,
) -> CandidateMetrics:
    return _candidate_failure_metrics_impl(
        item=item,
        reason=reason,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        model_dir=model_dir,
    )


def collect_candidate_worker_metrics(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    failure_metrics_fn: Callable[..., CandidateMetrics] | None = None,
) -> list[CandidateMetrics]:
    return _collect_candidate_worker_metrics_impl(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=failure_metrics_fn,
    )


def train_candidate_metrics(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    serial_fn: Callable[..., list[CandidateMetrics]],
    local_parallel_fn: Callable[..., list[CandidateMetrics]],
) -> list[CandidateMetrics]:
    if args.candidate_execution_mode == "serial":
        return serial_fn(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            source_examples=source_examples,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_prompt=proposal_prompt,
            round_index=round_index,
            work_items=work_items,
            round_dir=round_dir,
            eval_examples=eval_examples,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            config=config,
            attempt_index=attempt_index,
        )
    if args.candidate_execution_mode == "local_parallel":
        return local_parallel_fn(
            args=args,
            task=task,
            current_checkpoint=current_checkpoint,
            source_examples=source_examples,
            proposal_trace_buffer=proposal_trace_buffer,
            outcome_trace_buffer=outcome_trace_buffer,
            proposal_prompt=proposal_prompt,
            round_index=round_index,
            work_items=work_items,
            round_dir=round_dir,
            eval_examples=eval_examples,
            current_final_accuracy=current_final_accuracy,
            current_per_size_accuracy=current_per_size_accuracy,
            init_final_accuracy=init_final_accuracy,
            attempt_index=attempt_index,
        )
    raise ValueError(f"Unsupported candidate_execution_mode={args.candidate_execution_mode!r}.")


# --- from candidate_dispatch_entrypoints.py ---
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Sequence

from self.adaptive.traces import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.adaptive.proposal import PromptBundle
from self.core.training import TrainingConfig


@dataclass(frozen=True)
class CandidateDispatchEntrypointDeps:
    train_and_score_candidate: Callable[..., CandidateMetrics]
    candidate_failure_metrics: Callable[..., CandidateMetrics]
    collect_candidate_worker_metrics: Callable[..., List[CandidateMetrics]]
    train_candidates_serial: Callable[..., List[CandidateMetrics]]
    train_candidates_local_parallel: Callable[..., List[CandidateMetrics]]
    subprocess_module: Any


def build_candidate_dispatch_deps(bindings: Any) -> CandidateDispatchEntrypointDeps:
    return CandidateDispatchEntrypointDeps(
        train_and_score_candidate=bindings.train_and_score_candidate,
        candidate_failure_metrics=bindings._candidate_failure_metrics,
        collect_candidate_worker_metrics=bindings._collect_candidate_worker_metrics,
        train_candidates_serial=bindings.train_candidates_serial,
        train_candidates_local_parallel=bindings.train_candidates_local_parallel,
        subprocess_module=bindings.subprocess,
    )


def candidate_failure_metrics_with_deps(
    *,
    item: CandidateWorkItem,
    reason: str,
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    model_dir: Path | None = None,
) -> CandidateMetrics:
    return candidate_failure_metrics(
        item=item,
        reason=reason,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        model_dir=model_dir,
    )


def train_candidates_serial_with_deps(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return train_candidates_serial(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        config=config,
        attempt_index=attempt_index,
        score_candidate_fn=deps.train_and_score_candidate,
    )


def collect_candidate_worker_metrics_with_deps(
    *,
    round_dir: Path,
    work_items: Sequence[CandidateWorkItem],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return collect_candidate_worker_metrics(
        round_dir=round_dir,
        work_items=work_items,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        failure_metrics_fn=deps.candidate_failure_metrics,
    )


def train_candidates_local_parallel_with_deps(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return train_candidates_local_parallel(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        attempt_index=attempt_index,
        collect_metrics_fn=deps.collect_candidate_worker_metrics,
        subprocess_module=deps.subprocess_module,
    )


def train_candidate_metrics_with_deps(
    *,
    args: argparse.Namespace,
    task: Any,
    current_checkpoint: str,
    source_examples: Sequence[Any],
    proposal_trace_buffer: Sequence[ProposalTraceExample],
    outcome_trace_buffer: Sequence[OutcomeTraceExample],
    proposal_prompt: PromptBundle,
    round_index: int,
    work_items: Sequence[CandidateWorkItem],
    round_dir: Path,
    eval_examples: Sequence[Any],
    current_final_accuracy: float,
    current_per_size_accuracy: Mapping[int, float],
    init_final_accuracy: float,
    config: TrainingConfig,
    attempt_index: int,
    deps: CandidateDispatchEntrypointDeps,
) -> List[CandidateMetrics]:
    return train_candidate_metrics(
        args=args,
        task=task,
        current_checkpoint=current_checkpoint,
        source_examples=source_examples,
        proposal_trace_buffer=proposal_trace_buffer,
        outcome_trace_buffer=outcome_trace_buffer,
        proposal_prompt=proposal_prompt,
        round_index=round_index,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=eval_examples,
        current_final_accuracy=current_final_accuracy,
        current_per_size_accuracy=current_per_size_accuracy,
        init_final_accuracy=init_final_accuracy,
        config=config,
        attempt_index=attempt_index,
        serial_fn=deps.train_candidates_serial,
        local_parallel_fn=deps.train_candidates_local_parallel,
    )

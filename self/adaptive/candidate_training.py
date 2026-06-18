#!/usr/bin/env python3
"""Candidate data construction, training, scoring, and selection."""

from __future__ import annotations

# --- from checkpoints.py ---
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from self.core.models import CandidateMetrics


@dataclass(frozen=True)
class CheckpointManager:
    output_dir: Path
    keep_candidate_models: bool = False
    keep_proposal_grpo_checkpoints: bool = False

    def cleanup_unselected_candidates(
        self,
        *,
        metrics: Sequence[CandidateMetrics],
        selected: Optional[CandidateMetrics],
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
        except ValueError:
            return []
        except OSError:
            return []
        if old_model_dir.parent.name == "proposal_grpo" and self.keep_proposal_grpo_checkpoints:
            return []
        if "candidates" in old_model_dir.parts and self.keep_candidate_models:
            return []
        shutil.rmtree(old_model_dir, ignore_errors=True)
        return [str(old_model_dir)]


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
) -> List[str]:
    return CheckpointManager(
        output_dir=output_dir,
        keep_candidate_models=keep_candidate_models,
        keep_proposal_grpo_checkpoints=keep_proposal_grpo_checkpoints,
    ).cleanup_replaced_checkpoint(
        old_checkpoint=old_checkpoint,
        new_checkpoint=new_checkpoint,
    )


# --- from candidate_data.py ---
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
    post_task_proposal_rehearsal_count: int,
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
    reward = frontier_delta + args.lambda_final * final_accuracy_delta
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
        pseudo_count=len(item.pseudo_examples),
        model_dir=model_dir,
        proposal_trace_replay_count=proposal_trace_replay_count,
        candidate_proposal_trace_count=candidate_proposal_trace_count,
        post_task_proposal_rehearsal_count=post_task_proposal_rehearsal_count,
        outcome_trace_replay_count=outcome_trace_replay_count,
        proposal_prediction=dict(item.proposal_prediction),
    )


# --- from training.py ---
import argparse
import shutil
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


def train_post_task_proposal_rehearsal(
    *,
    task_model_dir: Path,
    candidate_dir: Path,
    task: Any,
    args: argparse.Namespace,
    config: TrainingConfig,
    seed: int,
    proposal_trace_buffer: Sequence[Any],
    candidate_trace_examples: Sequence[Any],
    post_task_rehearsal_examples: Sequence[Any],
    train_checkpoint_fn: Callable[..., Tuple[Any, Any, Path]] = train_checkpoint,
    write_json_fn: Callable[[Path, Any], None] = worker_io.write_json,
) -> Tuple[Any, Any, Path]:
    model, tokenizer, model_dir = train_checkpoint_fn(
        source_checkpoint=str(task_model_dir),
        train_examples=post_task_rehearsal_examples,
        output_dir=candidate_dir / "proposal_rehearsal",
        task=task,
        args=args,
        config=config,
        seed=seed + 37,
        recipe_phase_name="proposal_rehearsal",
    )
    write_json_fn(
        candidate_dir / "proposal_rehearsal_summary.json",
        {
            "source_checkpoint": str(task_model_dir),
            "model_dir": str(model_dir),
            "examples": len(post_task_rehearsal_examples),
            "base_candidate_trace_examples": len(candidate_trace_examples),
            "base_selected_trace_buffer_size": len(proposal_trace_buffer),
            "repeat_count": args.post_task_proposal_rehearsal_repeat_count,
            "max_examples": args.post_task_proposal_rehearsal_max_examples,
        },
    )
    if not args.keep_all_candidate_models and task_model_dir.parent.exists():
        shutil.rmtree(task_model_dir.parent, ignore_errors=True)
    return model, tokenizer, model_dir


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
    build_post_task_proposal_rehearsal_examples,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.adaptive.traces import build_candidate_proposal_trace_example
from self.core.models import CandidateWorkItem
from self.adaptive.proposal_io import write_trace_jsonl
from self.adaptive.proposal_prompts import PromptBundle


@dataclass(frozen=True)
class CandidateTrainingMix:
    task_train_examples: List[Any]
    outcome_replay_examples: List[OutcomeTraceExample]
    candidate_trace_examples: List[ProposalTraceExample]
    mixed_proposal_replay_examples: List[ProposalTraceExample]
    mixed_candidate_trace_examples: List[ProposalTraceExample]
    post_task_rehearsal_examples: List[ProposalTraceExample]
    train_examples: List[Any]

    @property
    def summary_counts(self) -> dict[str, int]:
        return {
            "task_train_examples": len(self.task_train_examples),
            "outcome_trace_replay_examples": len(self.outcome_replay_examples),
            "proposal_trace_replay_examples": len(self.mixed_proposal_replay_examples),
            "candidate_proposal_trace_examples": len(self.candidate_trace_examples),
            "mixed_candidate_proposal_trace_examples": len(self.mixed_candidate_trace_examples),
            "post_task_proposal_rehearsal_examples": len(self.post_task_rehearsal_examples),
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
    task_train_examples = list(source_examples) + list(item.pseudo_examples)
    outcome_replay_examples = sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcome_trace_buffer,
        task_train_count=len(task_train_examples),
        rng=random_cls(seed + 6151),
    )
    candidate_trace_examples: List[ProposalTraceExample] = []
    if item.completion and (args.post_task_proposal_rehearsal or args.proposal_trace_replay_ratio > 0.0):
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
    if not args.post_task_proposal_rehearsal:
        mixed_proposal_replay_examples = sample_proposal_trace_replay(
            args=args,
            trace_buffer=proposal_trace_buffer,
            task_train_count=len(task_train_examples),
            rng=random_cls(seed + 7919),
        )
    mixed_candidate_trace_examples = [] if args.post_task_proposal_rehearsal else list(candidate_trace_examples)
    post_task_rehearsal_examples = build_post_task_proposal_rehearsal_examples(
        args=args,
        proposal_trace_buffer=proposal_trace_buffer,
        candidate_trace_examples=candidate_trace_examples,
        rng=random_cls(seed + 8863),
    )
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
        post_task_rehearsal_examples=list(post_task_rehearsal_examples),
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
    if mix.post_task_rehearsal_examples:
        write_trace_jsonl_fn(
            candidate_dir / "post_task_proposal_rehearsal_examples.jsonl",
            [example.to_json_dict() for example in mix.post_task_rehearsal_examples],
        )
    write_json_fn(
        candidate_dir / "train_mix_summary.json",
        {
            **mix.summary_counts,
            "source_examples": len(source_examples),
            "pseudo_examples": len(item.pseudo_examples),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "outcome_trace_target_mode": args.outcome_trace_target_mode,
            "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
            "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
            "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
            "post_task_proposal_rehearsal": bool(args.post_task_proposal_rehearsal),
            "post_task_proposal_rehearsal_repeat_count": args.post_task_proposal_rehearsal_repeat_count,
            "post_task_proposal_rehearsal_max_examples": args.post_task_proposal_rehearsal_max_examples,
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
from self.adaptive.proposal_prompts import PromptBundle
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
    if training_mix.post_task_rehearsal_examples:
        del model
        del tokenizer
        clear_cuda_cache()
        model, tokenizer, model_dir = train_post_task_proposal_rehearsal(
            task_model_dir=task_model_dir,
            candidate_dir=candidate_dir,
            task=task,
            args=args,
            config=config,
            seed=seed,
            proposal_trace_buffer=proposal_trace_buffer,
            candidate_trace_examples=training_mix.candidate_trace_examples,
            post_task_rehearsal_examples=training_mix.post_task_rehearsal_examples,
        )
    final_accuracy, per_size_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=eval_examples,
        batch_size=config.per_device_eval_batch_size,
        decode_max_new_tokens=config.decode_max_new_tokens,
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
        post_task_proposal_rehearsal_count=len(training_mix.post_task_rehearsal_examples),
        outcome_trace_replay_count=len(training_mix.outcome_replay_examples),
    )
    worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
    del model
    del tokenizer
    clear_cuda_cache()
    return metrics


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

#!/usr/bin/env python3
"""Candidate train/eval scoring helpers for adaptive self-improvement."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from self.core import worker_io
from self.core.candidate_training_mix import (
    build_candidate_training_mix,
    write_candidate_training_mix_artifacts,
)
from self.core.experience_trace_models import OutcomeTraceExample, ProposalTraceExample
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposals import PromptBundle
from self.core.evaluation import evaluate_accuracy_with_breakdown, resolve_max_new_tokens
from self.core.model_io import ModelBootstrapCache, instantiate_model_and_tokenizer
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


def static_frontier_sizes(args: argparse.Namespace) -> List[int]:
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
        current_target_accuracy = float(current_per_size_accuracy.get(item.proposal.target, 0.0))
        current_frontier_accuracy = mean_accuracy_for_sizes(current_per_size_accuracy, static_frontier_sizes(args))
        metrics = CandidateMetrics(
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer, model_dir = train_checkpoint(
            source_checkpoint=str(task_model_dir),
            train_examples=training_mix.post_task_rehearsal_examples,
            output_dir=candidate_dir / "proposal_rehearsal",
            task=task,
            args=args,
            config=config,
            seed=seed + 37,
            recipe_phase_name="proposal_rehearsal",
        )
        worker_io.write_json(
            candidate_dir / "proposal_rehearsal_summary.json",
            {
                "source_checkpoint": str(task_model_dir),
                "model_dir": str(model_dir),
                "examples": len(training_mix.post_task_rehearsal_examples),
                "base_candidate_trace_examples": len(training_mix.candidate_trace_examples),
                "base_selected_trace_buffer_size": len(proposal_trace_buffer),
                "repeat_count": args.post_task_proposal_rehearsal_repeat_count,
                "max_examples": args.post_task_proposal_rehearsal_max_examples,
            },
        )
        if not args.keep_all_candidate_models and task_model_dir.parent.exists():
            shutil.rmtree(task_model_dir.parent, ignore_errors=True)
    final_accuracy, per_size_accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        examples=eval_examples,
        batch_size=config.per_device_eval_batch_size,
        decode_max_new_tokens=config.decode_max_new_tokens,
    )
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
    metrics = CandidateMetrics(
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
        proposal_trace_replay_count=len(training_mix.mixed_proposal_replay_examples),
        candidate_proposal_trace_count=len(training_mix.candidate_trace_examples),
        post_task_proposal_rehearsal_count=len(training_mix.post_task_rehearsal_examples),
        outcome_trace_replay_count=len(training_mix.outcome_replay_examples),
        proposal_prediction=dict(item.proposal_prediction),
    )
    worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

#!/usr/bin/env python3
"""Candidate train/eval scoring helpers for adaptive self-improvement."""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from self.core import worker_io
from self.core.experience_trace_models import (
    OutcomeTraceExample,
    ProposalTraceExample,
    build_post_task_proposal_rehearsal_examples,
    sample_outcome_trace_replay,
    sample_proposal_trace_replay,
)
from self.core.experience_traces import build_candidate_proposal_trace_example
from self.core.models import CandidateMetrics, CandidateWorkItem
from self.core.proposals import PromptBundle, write_trace_jsonl
from self.core.data_io import save_examples
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

    task_train_examples = list(source_examples) + list(item.pseudo_examples)
    outcome_replay_examples = sample_outcome_trace_replay(
        args=args,
        trace_buffer=outcome_trace_buffer,
        task_train_count=len(task_train_examples),
        rng=random.Random(seed + 6151),
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
            rng=random.Random(seed + 7919),
        )
    mixed_candidate_trace_examples = [] if args.post_task_proposal_rehearsal else list(candidate_trace_examples)
    post_task_rehearsal_examples = build_post_task_proposal_rehearsal_examples(
        args=args,
        proposal_trace_buffer=proposal_trace_buffer,
        candidate_trace_examples=candidate_trace_examples,
        rng=random.Random(seed + 8863),
    )
    train_examples = (
        task_train_examples
        + list(outcome_replay_examples)
        + mixed_proposal_replay_examples
        + mixed_candidate_trace_examples
    )
    save_examples(candidate_dir / "train_examples.jsonl", task_train_examples, task.serialize_example)
    if outcome_replay_examples:
        write_trace_jsonl(
            candidate_dir / "outcome_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in outcome_replay_examples],
        )
    if mixed_proposal_replay_examples:
        write_trace_jsonl(
            candidate_dir / "proposal_trace_replay_examples.jsonl",
            [example.to_json_dict() for example in mixed_proposal_replay_examples],
        )
    if candidate_trace_examples:
        write_trace_jsonl(
            candidate_dir / "candidate_proposal_trace_example.jsonl",
            [example.to_json_dict() for example in candidate_trace_examples],
        )
    if post_task_rehearsal_examples:
        write_trace_jsonl(
            candidate_dir / "post_task_proposal_rehearsal_examples.jsonl",
            [example.to_json_dict() for example in post_task_rehearsal_examples],
        )
    worker_io.write_json(
        candidate_dir / "train_mix_summary.json",
        {
            "task_train_examples": len(task_train_examples),
            "source_examples": len(source_examples),
            "pseudo_examples": len(item.pseudo_examples),
            "outcome_trace_buffer_size": len(outcome_trace_buffer),
            "outcome_trace_replay_examples": len(outcome_replay_examples),
            "outcome_trace_target_mode": args.outcome_trace_target_mode,
            "outcome_trace_replay_ratio": args.outcome_trace_replay_ratio,
            "outcome_trace_replay_max_examples": args.outcome_trace_replay_max_examples,
            "proposal_trace_buffer_size": len(proposal_trace_buffer),
            "proposal_trace_replay_examples": len(mixed_proposal_replay_examples),
            "candidate_proposal_trace_examples": len(candidate_trace_examples),
            "mixed_candidate_proposal_trace_examples": len(mixed_candidate_trace_examples),
            "proposal_trace_replay_ratio": args.proposal_trace_replay_ratio,
            "proposal_trace_replay_max_examples": args.proposal_trace_replay_max_examples,
            "post_task_proposal_rehearsal": bool(args.post_task_proposal_rehearsal),
            "post_task_proposal_rehearsal_examples": len(post_task_rehearsal_examples),
            "post_task_proposal_rehearsal_repeat_count": args.post_task_proposal_rehearsal_repeat_count,
            "post_task_proposal_rehearsal_max_examples": args.post_task_proposal_rehearsal_max_examples,
            "total_train_examples": len(train_examples),
        },
    )
    model, tokenizer, task_model_dir = train_checkpoint(
        source_checkpoint=current_checkpoint,
        train_examples=train_examples,
        output_dir=candidate_dir / "training",
        task=task,
        args=args,
        config=config,
        seed=seed,
        recipe_phase_name="self_improve",
        model_bootstrap_cache=model_bootstrap_cache,
    )
    model_dir = task_model_dir
    if post_task_rehearsal_examples:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model, tokenizer, model_dir = train_checkpoint(
            source_checkpoint=str(task_model_dir),
            train_examples=post_task_rehearsal_examples,
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
                "examples": len(post_task_rehearsal_examples),
                "base_candidate_trace_examples": len(candidate_trace_examples),
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
        proposal_trace_replay_count=len(mixed_proposal_replay_examples),
        candidate_proposal_trace_count=len(candidate_trace_examples),
        post_task_proposal_rehearsal_count=len(post_task_rehearsal_examples),
        outcome_trace_replay_count=len(outcome_replay_examples),
        proposal_prediction=dict(item.proposal_prediction),
    )
    worker_io.write_json(candidate_dir / "candidate_metrics.json", metrics.to_json_dict())
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

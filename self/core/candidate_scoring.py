#!/usr/bin/env python3
"""Candidate train/eval scoring helpers for adaptive self-improvement."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

from self.core import worker_io
from self.core.candidate_training_mix import (
    build_candidate_training_mix,
    write_candidate_training_mix_artifacts,
)
from self.core.candidate_rewards import (
    build_no_pseudo_candidate_metrics,
    build_trained_candidate_metrics,
    mean_accuracy_for_sizes,
    static_frontier_sizes,
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics

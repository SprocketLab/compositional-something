"""Candidate checkpoint training, evaluation, and rehearsal runtime helpers."""

from __future__ import annotations

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
from self.core.recipe_presets import recipe_enabled


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

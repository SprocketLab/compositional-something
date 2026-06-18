"""Training dataset, collator, argument, and Trainer helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import Trainer, TrainingArguments

from self.core.recipe_training import (
    BatchSamplerWarmupStableDecayTrainer,
    WarmupStableDecayTrainer,
    make_recipe_training_args,
)
from self.core.recipe_presets import (
    fit_recipe_phase_to_max_steps,
    recipe_enabled,
    resolve_recipe_phase,
    resolve_self_improvement_recipe,
)
from self.core.training_data import (
    BatchSamplerTrainer,
    CausalLMDataCollator,
    PromptTargetExample,
    SizeBucketBatchSampler,
    SizeGetter,
    TokenizedPromptTargetDataset,
)


TRAINING_ARGUMENT_FIELDS = set(inspect.signature(TrainingArguments.__init__).parameters)
TRAINING_ARGUMENT_FIELDS.discard("self")


def training_arg_supported(name: str) -> bool:
    return name in TRAINING_ARGUMENT_FIELDS


@dataclass
class TrainingConfig:
    num_epochs: int
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    weight_decay: float
    logging_steps: int
    max_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    decode_max_new_tokens: int = 16


def make_training_args(
    output_dir: Path,
    config: TrainingConfig,
    *,
    bf16: bool,
    fp16: bool,
    skip_save: bool,
    keep_checkpoints: bool = False,
    seed: int,
    recipe: str = "none",
    recipe_phase_name: str = "self_improve",
    recipe_phase_overrides: Optional[Dict[str, object]] = None,
) -> TrainingArguments:
    if recipe_enabled(recipe):
        preset = resolve_self_improvement_recipe(recipe)
        phase = resolve_recipe_phase(preset, recipe_phase_name)
        return make_recipe_training_args(
            output_dir=output_dir / "trainer",
            preset=preset,
            phase=phase,
            phase_overrides=recipe_phase_overrides,
            seed=seed,
            bf16=bf16,
            fp16=fp16,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            max_steps=config.max_steps if config.max_steps is not None else phase.max_steps,
            auto_find_batch_size=preset.auto_find_batch_size,
        )

    raw_kwargs: Dict[str, object] = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "report_to": [],
        "bf16": bf16,
        "fp16": fp16 and not bf16,
        "seed": seed,
        "disable_tqdm": False,
    }
    if config.max_steps is not None:
        raw_kwargs["max_steps"] = config.max_steps
    if config.eval_steps is not None and config.eval_steps > 0:
        raw_kwargs["eval_steps"] = config.eval_steps

    evaluation_setting = "steps" if config.eval_steps is not None and config.eval_steps > 0 else "no"
    save_setting = "no" if skip_save else "epoch"

    training_kwargs: Dict[str, object] = {}
    for key, value in raw_kwargs.items():
        if training_arg_supported(key) and value is not None:
            training_kwargs[key] = value

    if training_arg_supported("evaluation_strategy"):
        training_kwargs["evaluation_strategy"] = evaluation_setting
    elif training_arg_supported("eval_strategy"):
        training_kwargs["eval_strategy"] = evaluation_setting
    elif evaluation_setting != "no" and training_arg_supported("evaluate_during_training"):
        training_kwargs["evaluate_during_training"] = True

    if training_arg_supported("save_strategy"):
        training_kwargs["save_strategy"] = save_setting
    elif training_arg_supported("save_steps") and save_setting == "no":
        training_kwargs["save_steps"] = 0

    if not skip_save and not keep_checkpoints and training_arg_supported("save_total_limit"):
        training_kwargs["save_total_limit"] = 1

    return TrainingArguments(**training_kwargs)


def build_trainer(
    *,
    model: Any,
    training_args: TrainingArguments,
    train_dataset: TokenizedPromptTargetDataset,
    data_collator: Any,
    seed: int,
    size_getter: SizeGetter,
    bucket_train_batches_by_size: bool,
    recipe: str = "none",
    recipe_phase_name: str = "self_improve",
    recipe_phase_overrides: Optional[Dict[str, object]] = None,
) -> Trainer:
    if recipe_enabled(recipe):
        preset = resolve_self_improvement_recipe(recipe)
        raw_phase = resolve_recipe_phase(preset, recipe_phase_name)
        if recipe_phase_overrides:
            filtered_overrides = {
                key: value
                for key, value in recipe_phase_overrides.items()
                if value is not None and hasattr(raw_phase, key)
            }
            if filtered_overrides:
                raw_phase = replace(raw_phase, **filtered_overrides)
        phase = fit_recipe_phase_to_max_steps(
            raw_phase,
            max_steps=int(training_args.max_steps),
        )
        if not bucket_train_batches_by_size:
            return WarmupStableDecayTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                num_stable_steps=phase.num_stable_steps,
                num_decay_steps=phase.num_decay_steps,
                min_lr_ratio=preset.min_lr_ratio,
            )

        train_batch_sampler = SizeBucketBatchSampler(
            train_dataset.examples,
            training_args.per_device_train_batch_size,
            size_getter=size_getter,
            seed=seed,
            drop_last=bool(getattr(training_args, "dataloader_drop_last", False)),
        )
        print("[INFO] Using exact-size train batch bucketing.", flush=True)
        return BatchSamplerWarmupStableDecayTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
            train_batch_sampler=train_batch_sampler,
            num_stable_steps=phase.num_stable_steps,
            num_decay_steps=phase.num_decay_steps,
            min_lr_ratio=preset.min_lr_ratio,
        )

    if not bucket_train_batches_by_size:
        return Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )

    train_batch_sampler = SizeBucketBatchSampler(
        train_dataset.examples,
        training_args.per_device_train_batch_size,
        size_getter=size_getter,
        seed=seed,
        drop_last=bool(getattr(training_args, "dataloader_drop_last", False)),
    )
    print("[INFO] Using exact-size train batch bucketing.", flush=True)
    return BatchSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        train_batch_sampler=train_batch_sampler,
    )

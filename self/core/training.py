"""Training dataset, collator, argument, and Trainer helpers."""

from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import Trainer, TrainingArguments

from self.core.recipes import (
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


SizeGetter = Callable[[Any], int]

TRAINING_ARGUMENT_FIELDS = set(inspect.signature(TrainingArguments.__init__).parameters)
TRAINING_ARGUMENT_FIELDS.discard("self")


class PromptTargetExample(Protocol):
    def prompt(self) -> str:
        ...

    def target(self) -> str:
        ...


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


class TokenizedPromptTargetDataset(Dataset):
    """Lazily tokenized prompt/target dataset for causal LM fine-tuning."""

    def __init__(self, examples: Sequence[PromptTargetExample], tokenizer: Any, add_eos: bool = True):
        self.tokenizer = tokenizer
        self.examples = list(examples)
        self.add_eos = add_eos

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.examples[idx]
        prompt_ids = self.tokenizer.encode(example.prompt(), add_special_tokens=False)
        target_prefix = " "
        target_prefix_fn = getattr(example, "target_prefix", None)
        if callable(target_prefix_fn):
            target_prefix = str(target_prefix_fn())
        target_ids = self.tokenizer.encode(f"{target_prefix}{example.target()}", add_special_tokens=False)

        input_ids: List[int] = []
        labels: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            labels.append(-100)

        input_ids.extend(prompt_ids)
        labels.extend([-100] * len(prompt_ids))

        input_ids.extend(target_ids)
        labels.extend(target_ids)

        if self.add_eos and self.tokenizer.eos_token_id is not None:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class SizeBucketBatchSampler(BatchSampler):
    """Yield batches that contain examples from exactly one size bucket."""

    def __init__(
        self,
        examples: Sequence[Any],
        batch_size: int,
        *,
        size_getter: SizeGetter,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.batch_size = int(batch_size)
        self.size_getter = size_getter
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._iteration = 0
        self._size_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, example in enumerate(examples):
            self._size_to_indices[int(size_getter(example))].append(idx)

    def __iter__(self):
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1

        batches: List[List[int]] = []
        for size_value in sorted(self._size_to_indices):
            indices = list(self._size_to_indices[size_value])
            rng.shuffle(indices)
            full_count = len(indices) // self.batch_size
            for batch_idx in range(full_count):
                start = batch_idx * self.batch_size
                batches.append(indices[start : start + self.batch_size])
            remainder = len(indices) % self.batch_size
            if remainder and not self.drop_last:
                batches.append(indices[-remainder:])

        rng.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self._size_to_indices.values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += math.ceil(len(indices) / self.batch_size)
        return total


class BatchSamplerTrainer(Trainer):
    """Trainer variant that accepts an explicit train batch sampler."""

    def __init__(self, *args, train_batch_sampler: Optional[BatchSampler] = None, **kwargs) -> None:
        self._train_batch_sampler = train_batch_sampler
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if self._train_batch_sampler is None:
            return super().get_train_dataloader()

        dataloader_kwargs: Dict[str, Any] = {
            "batch_sampler": self._train_batch_sampler,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if getattr(self.args, "dataloader_prefetch_factor", None) is not None:
                dataloader_kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(self.train_dataset, **dataloader_kwargs)
        return self.accelerator.prepare(dataloader)


@dataclass
class CausalLMDataCollator:
    tokenizer: Any

    def __call__(self, features: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer needs pad_token_id or eos_token_id for padding.")

        batch_input_ids: List[List[int]] = []
        batch_attention: List[List[int]] = []
        batch_labels: List[List[int]] = []
        for feature in features:
            pad_count = max_length - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_count)
            batch_attention.append(feature["attention_mask"] + [0] * pad_count)
            batch_labels.append(feature["labels"] + [-100] * pad_count)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


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

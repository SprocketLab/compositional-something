"""Training datasets, collators, and batch-sampler Trainer variants."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset
from transformers import Trainer


SizeGetter = Callable[[Any], int]


class PromptTargetExample(Protocol):
    def prompt(self) -> str:
        ...

    def target(self) -> str:
        ...


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

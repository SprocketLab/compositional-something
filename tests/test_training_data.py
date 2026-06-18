from __future__ import annotations

from dataclasses import dataclass

from self.core import training, training_data
from self import self_improvement_core as core_facade


class _Tokenizer:
    bos_token_id = 101
    eos_token_id = 102
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]


@dataclass
class _Example:
    prompt_text: str
    target_text: str
    prefix: str = " "

    def prompt(self) -> str:
        return self.prompt_text

    def target(self) -> str:
        return self.target_text

    def target_prefix(self) -> str:
        return self.prefix


def test_training_data_old_reexports_keep_identity() -> None:
    assert training.TokenizedPromptTargetDataset is training_data.TokenizedPromptTargetDataset
    assert training.CausalLMDataCollator is training_data.CausalLMDataCollator
    assert training.SizeBucketBatchSampler is training_data.SizeBucketBatchSampler
    assert training.BatchSamplerTrainer is training_data.BatchSamplerTrainer
    assert core_facade.TokenizedPromptTargetDataset is training_data.TokenizedPromptTargetDataset
    assert core_facade.CausalLMDataCollator is training_data.CausalLMDataCollator


def test_tokenized_prompt_target_dataset_masks_prompt_and_honors_target_prefix() -> None:
    tokenizer = _Tokenizer()
    dataset = training_data.TokenizedPromptTargetDataset(
        [_Example(prompt_text="Q:", target_text="7", prefix="")],
        tokenizer,
    )

    item = dataset[0]

    assert item["input_ids"] == [101, ord("Q"), ord(":"), ord("7"), 102]
    assert item["labels"] == [-100, -100, -100, ord("7"), 102]
    assert item["attention_mask"] == [1, 1, 1, 1, 1]


def test_causal_lm_data_collator_pads_inputs_attention_and_labels() -> None:
    collator = training_data.CausalLMDataCollator(_Tokenizer())
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [3], "attention_mask": [1], "labels": [3]},
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [3, -100]]

from __future__ import annotations

from self.coding.atomic_data import AtomicExample
from self.coding.training import ChatTargetDataset


class FakeTokenizer:
    unk_token_id = -1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        return {"input_ids": [10, 11, 12]}

    def encode(self, text, add_special_tokens=False):
        return [20, 21]

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 99


def test_chat_dataset_masks_entire_prompt_and_trains_target_plus_end_token():
    example = AtomicExample(
        task="bfcl",
        source_id="x",
        source_group_id="x",
        split="train",
        messages=({"role": "user", "content": "hello"},),
        target="[]",
        evaluator={},
    )
    item = ChatTargetDataset([example], FakeTokenizer(), max_length=16)[0]
    assert item["input_ids"] == [10, 11, 12, 20, 21, 99]
    assert item["labels"] == [-100, -100, -100, 20, 21, 99]

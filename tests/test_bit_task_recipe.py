from __future__ import annotations

from self.self_improvement_core import SizeBucketBatchSampler, instantiate_model_and_tokenizer
from self.self_improvement_recipe import RECIPE_ALGORITHMIC_SELF_IMPROVE_V1, build_recipe_tokenizer, resolve_self_improvement_recipe
from self.self_improvement_tasks import RunLengthExample


def test_algorithmic_recipe_instantiates_scratch_model_for_bit_tasks(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    model, tokenizer = instantiate_model_and_tokenizer(
        "ignored",
        bf16=False,
        fp16=False,
        init_from_scratch=True,
        tokenizer_mode="auto",
        recipe=RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
    )

    assert getattr(model.config, "use_no_position_embeddings", False) is True
    assert model.config.hidden_size == 384
    assert model.config.num_hidden_layers == 6
    assert tokenizer.padding_side == "left"


def test_algorithmic_recipe_tokenizer_round_trips_run_length_prompts():
    preset = resolve_self_improvement_recipe(RECIPE_ALGORITHMIC_SELF_IMPROVE_V1)
    tokenizer = build_recipe_tokenizer(preset)

    texts = [
        "Q: runlen(001110) = ?\nA:",
        "3|0|2",
    ]

    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert token_ids
        assert tokenizer.unk_token_id not in token_ids
        assert tokenizer.decode(token_ids, skip_special_tokens=True) == text


def test_size_bucket_batch_sampler_keeps_each_batch_single_size():
    examples = [
        RunLengthExample(bitstring="0101", bits=4, max_run=1, prefix_run=1, suffix_run=1),
        RunLengthExample(bitstring="1111", bits=4, max_run=4, prefix_run=4, suffix_run=4),
        RunLengthExample(bitstring="010101", bits=6, max_run=1, prefix_run=1, suffix_run=1),
        RunLengthExample(bitstring="000111", bits=6, max_run=3, prefix_run=3, suffix_run=3),
        RunLengthExample(bitstring="00111111", bits=8, max_run=6, prefix_run=0, suffix_run=6),
    ]
    sampler = SizeBucketBatchSampler(
        examples,
        batch_size=2,
        size_getter=lambda example: example.bits,
        seed=0,
    )

    batches = list(sampler)

    assert sorted(len(batch) for batch in batches) == [1, 2, 2]
    for batch in batches:
        sizes = {examples[idx].bits for idx in batch}
        assert len(sizes) == 1


def test_run_length_legacy_prompt_remains_unchanged():
    run_length = RunLengthExample(bitstring="001110", bits=6, max_run=3, prefix_run=0, suffix_run=0)

    assert run_length.prompt() == "Q: runlen(001110) = ?\nA:"

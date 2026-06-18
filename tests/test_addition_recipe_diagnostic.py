from __future__ import annotations

import math

from core.addition_pipeline import AdditionExample
from self.addition_recipe import (
    NoPELlamaForCausalLM,
    PaddingAwareCausalLMDataCollator,
    build_recipe_tokenizer,
    instantiate_recipe_model,
    make_warmup_stable_decay_lambda,
    resolve_recipe_phase,
    resolve_addition_recipe,
)


def test_recipe_preset_instantiates_scratch_nope_model_and_character_tokenizer(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    preset = resolve_addition_recipe("arithmetic_self_improve_v1")
    tokenizer = build_recipe_tokenizer(preset)
    model = instantiate_recipe_model(tokenizer, preset, bf16=False, fp16=False)

    assert isinstance(model, NoPELlamaForCausalLM)
    assert model.config.hidden_size == 384
    assert model.config.intermediate_size == 1536
    assert model.config.num_attention_heads == 6
    assert model.config.num_hidden_layers == 6
    assert getattr(model.config, "use_no_position_embeddings", False) is True
    assert tokenizer.padding_side == "left"
    assert preset.seed_phase.weight_decay == 0.1
    assert preset.seed_phase.adam_beta2 == 0.99
    assert preset.seed_phase.max_steps == 10_000
    assert preset.seed_phase.warmup_steps == 1_000
    assert preset.seed_phase.num_decay_steps == 1_000
    assert preset.self_improve_phase.max_steps == 3_000
    assert preset.self_improve_phase.warmup_steps == 0
    assert preset.self_improve_phase.num_decay_steps == 1_000
    assert tokenizer.encode("Q: 12 + 34 = ?\nA:", add_special_tokens=False)


def test_recipe_tokenizer_supports_current_prompt_newline_without_unk():
    preset = resolve_addition_recipe("arithmetic_self_improve_v1")
    tokenizer = build_recipe_tokenizer(preset)

    token_ids = tokenizer.encode("Q: 12 + 34 = ?\nA:", add_special_tokens=False)

    assert token_ids
    assert tokenizer.unk_token_id not in token_ids


def test_padding_aware_collator_supports_right_and_left_padding():
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 3})()
    features = [
        {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [-100, 11]},
        {"input_ids": [20, 21, 22], "attention_mask": [1, 1, 1], "labels": [-100, 21, 22]},
    ]

    right = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="right")(features)
    left = PaddingAwareCausalLMDataCollator(tokenizer=tokenizer, padding_side="left")(features)

    assert right["input_ids"].tolist() == [[10, 11, 0], [20, 21, 22]]
    assert right["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert left["input_ids"].tolist() == [[0, 10, 11], [20, 21, 22]]
    assert left["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]


def test_warmup_stable_decay_lambda_matches_recipe_shape():
    preset = resolve_addition_recipe("arithmetic_self_improve_v1")
    phase = resolve_recipe_phase(preset, "seed")
    lr_lambda = make_warmup_stable_decay_lambda(
        num_warmup_steps=phase.warmup_steps,
        num_stable_steps=phase.num_stable_steps,
        num_decay_steps=phase.num_decay_steps,
        min_lr_ratio=preset.min_lr_ratio,
    )

    assert lr_lambda(0) == 0.0
    assert lr_lambda(500) == 0.5
    assert lr_lambda(1000) == 1.0
    assert lr_lambda(8999) == 1.0
    assert lr_lambda(9500) == 0.505
    assert math.isclose(lr_lambda(10_000), 0.01)
    assert math.isclose(lr_lambda(12_000), 0.01)


def test_self_improve_phase_has_zero_warmup_and_shorter_schedule():
    preset = resolve_addition_recipe("arithmetic_self_improve_v1")
    phase = resolve_recipe_phase(preset, "self_improve")

    assert phase.max_steps == 3_000
    assert phase.warmup_steps == 0
    assert phase.num_stable_steps == 2_000
    assert phase.num_decay_steps == 1_000


def test_addition_recipe_path_keeps_current_prompt_format():
    example = AdditionExample(a=12, b=34, result=46, digits=2, has_carry=False)
    assert example.prompt() == "Q: 12 + 34 = ?\nA:"

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pytest
import torch

from core.addition_pipeline import (
    AdditionExample,
    CausalLMDataCollator,
    DigitBucketBatchSampler,
    compose_examples,
    generate_prediction_map,
)
from core.addition_pipeline import VariantTrainingConfig
from core.addition_pipeline import evaluate_accuracy_with_breakdown
from self import self_improvement as legacy


class _DummyTokenizer:
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"
    padding_side = "left"


def test_addition_collator_uses_pad_token_id_and_right_padding():
    tokenizer = type("Tokenizer", (), {"pad_token_id": 0, "eos_token_id": 3, "padding_side": "left"})()
    collator = CausalLMDataCollator(tokenizer)

    batch = collator(
        [
            {"input_ids": [11, 12], "attention_mask": [1, 1], "labels": [-100, 12]},
            {"input_ids": [21, 22, 23, 24], "attention_mask": [1, 1, 1, 1], "labels": [-100, -100, 23, 24]},
        ]
    )

    assert batch["input_ids"].tolist() == [
        [11, 12, 0, 0],
        [21, 22, 23, 24],
    ]
    assert batch["attention_mask"].tolist() == [
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]
    assert batch["labels"].tolist() == [
        [-100, 12, -100, -100],
        [-100, -100, 23, 24],
    ]


def test_instantiate_model_and_tokenizer_uses_fixed_char_builder(monkeypatch):
    dummy_tokenizer = _DummyTokenizer()
    captured: dict[str, object] = {}

    def fake_build_fixed_char_tokenizer():
        return dummy_tokenizer

    def fake_load_model_for_tokenizer(model_path, tokenizer, *, bf16, fp16, recipe="none"):
        captured["model_path"] = model_path
        captured["tokenizer"] = tokenizer
        captured["bf16"] = bf16
        captured["fp16"] = fp16
        captured["recipe"] = recipe
        return "dummy-model"

    monkeypatch.setattr(legacy, "build_fixed_char_tokenizer", fake_build_fixed_char_tokenizer)
    monkeypatch.setattr(legacy, "load_model_for_tokenizer", fake_load_model_for_tokenizer)

    model, tokenizer = legacy.instantiate_model_and_tokenizer(
        "artifacts/models/addition_tiny_seed_best",
        bf16=True,
        fp16=False,
        tokenizer_mode="fixed_char",
    )

    assert model == "dummy-model"
    assert tokenizer is dummy_tokenizer
    assert captured == {
        "model_path": "artifacts/models/addition_tiny_seed_best",
        "tokenizer": dummy_tokenizer,
        "bf16": True,
        "fp16": False,
        "recipe": "none",
    }


def test_instantiate_model_and_tokenizer_uses_recipe_tokenizer_without_auto_tokenizer(monkeypatch):
    dummy_tokenizer = _DummyTokenizer()
    captured: dict[str, object] = {}

    def fake_resolve_addition_recipe(name):
        captured["recipe_name"] = name
        return type(
            "Preset",
            (),
            {
                "bf16": True,
                "per_device_train_batch_size": 1024,
                "per_device_eval_batch_size": 1024,
            },
        )()

    def fake_build_recipe_tokenizer(preset):
        captured["preset"] = preset
        return dummy_tokenizer

    def fake_apply_recipe_runtime_settings(preset):
        captured["runtime_preset"] = preset

    def fake_load_model_for_tokenizer(model_path, tokenizer, *, bf16, fp16, recipe="none"):
        captured["model_path"] = model_path
        captured["tokenizer"] = tokenizer
        captured["bf16"] = bf16
        captured["fp16"] = fp16
        captured["recipe"] = recipe
        return "recipe-model"

    def fail_auto_tokenizer(*args, **kwargs):
        raise AssertionError("AutoTokenizer should not be used for recipe-backed addition")

    monkeypatch.setattr(legacy, "resolve_addition_recipe", fake_resolve_addition_recipe)
    monkeypatch.setattr(legacy, "build_recipe_tokenizer", fake_build_recipe_tokenizer)
    monkeypatch.setattr(legacy, "apply_recipe_runtime_settings", fake_apply_recipe_runtime_settings)
    monkeypatch.setattr(legacy, "load_model_for_tokenizer", fake_load_model_for_tokenizer)
    monkeypatch.setattr(legacy.AutoTokenizer, "from_pretrained", fail_auto_tokenizer)

    model, tokenizer = legacy.instantiate_model_and_tokenizer(
        "/tmp/fake-recipe-seed",
        bf16=True,
        fp16=False,
        tokenizer_mode="auto",
        recipe="arithmetic_self_improve_v1",
    )

    assert model == "recipe-model"
    assert tokenizer is dummy_tokenizer
    assert captured["recipe_name"] == "arithmetic_self_improve_v1"
    assert captured["model_path"] == "/tmp/fake-recipe-seed"
    assert captured["tokenizer"] is dummy_tokenizer
    assert captured["recipe"] == "arithmetic_self_improve_v1"


def test_make_training_args_uses_recipe_phase_schedule(tmp_path: Path):
    config = VariantTrainingConfig(
        num_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=25,
    )

    seed_args = legacy.make_training_args(
        tmp_path / "seed",
        config,
        bf16=False,
        fp16=False,
        skip_save=False,
        keep_checkpoints=False,
        seed=0,
        recipe="arithmetic_self_improve_v1",
        recipe_phase_name="seed",
    )
    self_improve_args = legacy.make_training_args(
        tmp_path / "self_improve",
        config,
        bf16=False,
        fp16=False,
        skip_save=False,
        keep_checkpoints=False,
        seed=0,
        recipe="arithmetic_self_improve_v1",
        recipe_phase_name="self_improve",
    )

    assert seed_args.max_steps == 10_000
    assert seed_args.warmup_steps == 1_000
    assert seed_args.learning_rate == 5e-4
    assert seed_args.weight_decay == 0.1
    assert seed_args.adam_beta2 == 0.99
    assert self_improve_args.max_steps == 3_000
    assert self_improve_args.warmup_steps == 0
    assert self_improve_args.learning_rate == 5e-4


def test_make_training_args_applies_self_improve_recipe_phase_overrides(tmp_path: Path):
    config = VariantTrainingConfig(
        num_epochs=3,
        learning_rate=2e-4,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=256,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=25,
        max_steps=4500,
    )

    overrides = {"warmup_steps": 0, "num_stable_steps": 3500, "num_decay_steps": 1000}
    self_improve_args = legacy.make_training_args(
        tmp_path / "self_improve",
        config,
        bf16=False,
        fp16=False,
        skip_save=False,
        keep_checkpoints=False,
        seed=0,
        recipe="arithmetic_self_improve_v1",
        recipe_phase_name="self_improve",
        recipe_phase_overrides=overrides,
    )
    preset = legacy.resolve_addition_recipe("arithmetic_self_improve_v1")
    phase = legacy.resolve_recipe_phase(preset, "self_improve")
    overridden_phase = legacy.apply_recipe_phase_overrides(phase, overrides)

    assert self_improve_args.max_steps == 4500
    assert self_improve_args.warmup_steps == 0
    assert overridden_phase.num_stable_steps == 3500
    assert overridden_phase.num_decay_steps == 1000


def test_recipe_phase_overrides_for_args_only_applies_to_self_improve():
    args = argparse.Namespace(
        self_improve_warmup_steps=12,
        self_improve_stable_steps=34,
        self_improve_decay_steps=56,
    )

    assert legacy.recipe_phase_overrides_for_args(args, recipe_phase_name="seed") is None
    assert legacy.recipe_phase_overrides_for_args(args, recipe_phase_name="self_improve") == {
        "warmup_steps": 12,
        "num_stable_steps": 34,
        "num_decay_steps": 56,
    }


def test_cleanup_round_checkpoints_keeps_round_final_model(tmp_path: Path):
    round_dir = tmp_path / "round_00"
    round_dir.mkdir()
    checkpoint_dir = round_dir / "checkpoint-1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").write_text("checkpoint", encoding="utf-8")
    model_file = round_dir / "model.safetensors"
    model_file.write_text("final-model", encoding="utf-8")
    training_args = round_dir / "training_args.bin"
    training_args.write_text("args", encoding="utf-8")

    legacy.cleanup_round_checkpoints([round_dir])

    assert model_file.exists()
    assert training_args.exists()
    assert not checkpoint_dir.exists()


def test_make_training_args_does_not_cap_snapshots_when_keep_checkpoints_enabled(tmp_path: Path):
    config = VariantTrainingConfig(
        num_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        weight_decay=0.0,
        logging_steps=100,
    )

    capped_args = legacy.make_training_args(
        tmp_path / "capped",
        config,
        bf16=False,
        fp16=False,
        skip_save=False,
        keep_checkpoints=False,
        seed=0,
    )
    keep_all_args = legacy.make_training_args(
        tmp_path / "keep_all",
        config,
        bf16=False,
        fp16=False,
        skip_save=False,
        keep_checkpoints=True,
        seed=0,
    )

    assert capped_args.save_total_limit == 1
    assert keep_all_args.save_total_limit is None


def test_prepare_composed_train_can_target_no_boundary_bucket():
    base_examples = [
        AdditionExample(a=1, b=2, result=3, digits=1, has_carry=False),
        AdditionExample(a=3, b=4, result=7, digits=1, has_carry=False),
        AdditionExample(a=1, b=1, result=2, digits=1, has_carry=False),
        AdditionExample(a=8, b=7, result=15, digits=1, has_carry=True),
    ]
    base_splits = {"train": list(base_examples), "validation": [], "test": []}
    base_records = {
        "train": {legacy.example_key(example) for example in base_examples},
        "validation": set(),
        "test": set(),
    }

    generated, component_map, _ = legacy.prepare_composed_train(
        rng=random.Random(0),
        base_splits=base_splits,
        base_records=base_records,
        min_digits=2,
        max_digits=2,
        per_digit_count=5,
        allow_carry=True,
        boundary_carry_policy="no_boundary_carry",
    )

    assert len(generated) == 5
    for example in generated:
        assert legacy.get_boundary_carry_status(example, component_map) is False


def test_prepare_composed_eval_can_target_no_boundary_bucket():
    base_examples = [
        AdditionExample(a=1, b=2, result=3, digits=1, has_carry=False),
        AdditionExample(a=3, b=4, result=7, digits=1, has_carry=False),
        AdditionExample(a=1, b=1, result=2, digits=1, has_carry=False),
        AdditionExample(a=8, b=7, result=15, digits=1, has_carry=True),
    ]
    base_splits = {"train": list(base_examples), "validation": [], "test": []}
    base_records = {
        "train": {legacy.example_key(example) for example in base_examples},
        "validation": set(),
        "test": set(),
    }

    generated, component_map, _ = legacy.prepare_composed_eval(
        rng=random.Random(0),
        base_splits=base_splits,
        base_records=base_records,
        min_digits=2,
        max_digits=2,
        per_digit_count=5,
        boundary_carry_policy="no_boundary_carry",
    )

    assert len(generated) == 5
    for example in generated:
        assert legacy.get_boundary_carry_status(example, component_map) is False


class _LeftPadNumericTokenizer:
    pad_token_id = 0
    bos_token_id = 2
    eos_token_id = 3
    padding_side = "left"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1] * len(text)

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for token_id in token_ids:
            value = int(token_id)
            if value in {self.pad_token_id, self.bos_token_id, self.eos_token_id} and skip_special_tokens:
                continue
            if value == 1:
                pieces.append("9")
            elif value >= 100:
                pieces.append(str(value - 100))
            else:
                pieces.append("?")
        return "".join(pieces)


class _DummyGenerateModel(torch.nn.Module):
    def __init__(self, completions: list[list[int]]):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.completions = completions

    def generate(self, input_ids, attention_mask, max_new_tokens, do_sample=False, temperature=0.0, top_p=1.0):
        del attention_mask, max_new_tokens, do_sample, temperature, top_p
        batch_outputs = []
        max_completion = max(len(tokens) for tokens in self.completions[: input_ids.shape[0]])
        for row, completion in zip(input_ids, self.completions):
            completion_tensor = torch.tensor(completion, dtype=row.dtype, device=row.device)
            padded_completion = torch.full(
                (max_completion,),
                0,
                dtype=row.dtype,
                device=row.device,
            )
            padded_completion[: len(completion)] = completion_tensor
            batch_outputs.append(torch.cat([row, padded_completion], dim=0))
        return torch.stack(batch_outputs, dim=0)


def test_left_padded_generation_slices_after_full_prompt_width():
    tokenizer = _LeftPadNumericTokenizer()
    examples = [
        AdditionExample(a=1, b=2, result=3, digits=1, has_carry=False),
        AdditionExample(a=44, b=55, result=99, digits=2, has_carry=True),
    ]
    # Completion ids decode to "3" and "99". If decoding starts before the
    # shared prompt width for the shorter row, leaked prompt tokens decode to
    # leading 9s and corrupt the numeric answer.
    model = _DummyGenerateModel(completions=[[103], [109, 109]])

    accuracy, per_digit = evaluate_accuracy_with_breakdown(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=2,
        max_new_tokens=2,
    )
    prediction_map = generate_prediction_map(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=2,
        max_new_tokens=2,
    )

    assert accuracy == 1.0
    assert per_digit == {1: 1.0, 2: 1.0}
    assert prediction_map == {
        (1, 1, 2): "3",
        (2, 44, 55): "99",
    }


def test_build_direct_pseudo_examples_filters_missing_predictions(monkeypatch):
    examples = [
        AdditionExample(a=12, b=34, result=46, digits=2, has_carry=False),
        AdditionExample(a=56, b=78, result=134, digits=2, has_carry=True),
    ]

    def fake_generate_prediction_map(model, tokenizer, examples, batch_size, max_new_tokens):
        del model, tokenizer, batch_size, max_new_tokens
        return {legacy.example_key(examples[0]): "46"}

    monkeypatch.setattr(legacy, "generate_prediction_map", fake_generate_prediction_map)

    pseudo_examples, missing_total, diagnostics = legacy.build_direct_pseudo_examples(
        examples,
        model=object(),
        tokenizer=object(),
        batch_size=2,
        decode_max_new_tokens=8,
        mode="seed_replay_direct",
    )

    assert missing_total == 1
    assert len(pseudo_examples) == 1
    assert pseudo_examples[0].target_override == "46"
    assert diagnostics["candidate_total"] == 2
    assert diagnostics["retained_total"] == 1
    assert diagnostics["missing_total"] == 1


def test_collect_seed_replay_pseudo_examples_refills_missing_predictions(monkeypatch):
    state = {"calls": 0}

    def fake_generate_prediction_map(model, tokenizer, examples, batch_size, max_new_tokens):
        del model, tokenizer, batch_size, max_new_tokens
        state["calls"] += 1
        predictions = {legacy.example_key(example): str(example.result) for example in examples}
        if state["calls"] == 1 and examples:
            predictions.pop(legacy.example_key(examples[0]), None)
        return predictions

    monkeypatch.setattr(legacy, "generate_prediction_map", fake_generate_prediction_map)

    raw_examples, pseudo_examples, diagnostics = legacy.collect_seed_replay_pseudo_examples(
        rng=random.Random(0),
        min_digits=3,
        max_digits=4,
        per_digit_target=2,
        additional_exclude=None,
        model=object(),
        tokenizer=object(),
        batch_size=2,
        decode_max_new_tokens=8,
    )

    assert legacy.count_examples_by_digit(pseudo_examples) == {3: 2, 4: 2}
    assert len(raw_examples) == 5
    assert len(pseudo_examples) == 4
    assert diagnostics["requested_total"] == 4
    assert diagnostics["candidate_total"] == 5
    assert diagnostics["retained_total"] == 4
    assert diagnostics["missing_total"] == 1
    assert diagnostics["refill_rounds"] == 2


def test_collect_expansion_pseudo_examples_refills_missing_predictions(monkeypatch):
    counters = {8: 0, 9: 0}
    derive_state = {"calls": 0}

    def fake_prepare_composed_train(
        rng,
        base_splits,
        base_records,
        min_digits,
        max_digits,
        per_digit_count,
        allow_carry,
        boundary_carry_policy="any",
        additional_exclude=None,
        **kwargs,
    ):
        del rng, base_splits, base_records, allow_carry, boundary_carry_policy, additional_exclude, kwargs
        assert min_digits == max_digits
        examples = []
        component_map = {}
        keys = set()
        for _ in range(per_digit_count):
            serial = counters[min_digits]
            counters[min_digits] += 1
            example = AdditionExample(
                a=min_digits * 100 + serial,
                b=serial,
                result=min_digits * 100 + serial * 2,
                digits=min_digits,
                has_carry=False,
            )
            examples.append(example)
            key = legacy.example_key(example)
            component_map[key] = []
            keys.add(key)
        return examples, component_map, keys

    def fake_build_base_predictions(model, tokenizer, base_examples, *, batch_size, decode_max_new_tokens):
        del model, tokenizer, batch_size, decode_max_new_tokens
        return {legacy.example_key(example): str(example.result) for example in base_examples}

    def fake_derive_round_targets(
        composed_examples,
        component_map,
        target_max_digits,
        base_examples,
        *,
        model,
        tokenizer,
        batch_size,
        decode_max_new_tokens,
        pseudo_label_mode,
        corruption_rate=0.0,
        filter_component_carries=False,
        carry_error_fraction=0.0,
        rng=None,
        base_prediction_map=None,
    ):
        del (
            component_map,
            target_max_digits,
            base_examples,
            model,
            tokenizer,
            batch_size,
            decode_max_new_tokens,
            pseudo_label_mode,
            corruption_rate,
            filter_component_carries,
            carry_error_fraction,
            rng,
            base_prediction_map,
        )
        derive_state["calls"] += 1
        retained = list(composed_examples)
        if derive_state["calls"] == 1 and retained:
            retained = retained[1:]
        pseudo_examples = [legacy.clone_with_override(example, str(example.result)) for example in retained]
        return pseudo_examples, len(composed_examples) - len(pseudo_examples), {"corrupted_total": 0}

    monkeypatch.setattr(legacy, "prepare_composed_train", fake_prepare_composed_train)
    monkeypatch.setattr(legacy, "build_base_predictions", fake_build_base_predictions)
    monkeypatch.setattr(legacy, "derive_round_targets", fake_derive_round_targets)

    raw_examples, component_map, pseudo_examples, diagnostics = legacy.collect_expansion_pseudo_examples(
        rng=random.Random(0),
        base_splits={"train": [], "validation": [], "test": []},
        base_records={"train": set(), "validation": set(), "test": set()},
        min_digits=8,
        max_digits=9,
        per_digit_target=2,
        allow_carry=True,
        boundary_carry_policy="no_boundary_carry",
        additional_exclude=None,
        base_examples=[
            AdditionExample(a=111, b=222, result=333, digits=3, has_carry=False),
            AdditionExample(a=444, b=555, result=999, digits=3, has_carry=True),
        ],
        model=object(),
        tokenizer=object(),
        batch_size=2,
        decode_max_new_tokens=8,
        pseudo_label_mode="compose",
        corruption_rate=0.0,
        filter_component_carries=True,
        carry_error_fraction=0.0,
        pseudo_rng=random.Random(1),
    )

    assert legacy.count_examples_by_digit(pseudo_examples) == {8: 2, 9: 2}
    assert len(component_map) == len(raw_examples) == 5
    assert diagnostics["requested_total"] == 4
    assert diagnostics["candidate_total"] == 5
    assert diagnostics["retained_total"] == 4
    assert diagnostics["missing_total"] == 1
    assert diagnostics["refill_rounds"] == 2


def test_digit_bucket_batch_sampler_keeps_each_batch_single_digit():
    tokenizer = _LeftPadNumericTokenizer()
    examples = [
        AdditionExample(a=1, b=2, result=3, digits=1, has_carry=False),
        AdditionExample(a=4, b=5, result=9, digits=1, has_carry=False),
        AdditionExample(a=12, b=34, result=46, digits=2, has_carry=False),
        AdditionExample(a=56, b=78, result=134, digits=2, has_carry=True),
        AdditionExample(a=90, b=10, result=100, digits=2, has_carry=True),
    ]
    dataset = legacy.TokenizedAdditionDataset(examples, tokenizer)
    sampler = DigitBucketBatchSampler(dataset, batch_size=2, seed=0)

    batches = list(sampler)

    assert sorted(len(batch) for batch in batches) == [1, 2, 2]
    for batch in batches:
        digits = {dataset.digits_for_index(idx) for idx in batch}
        assert len(digits) == 1


class _FakeTokenizerForLegacy:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class _FakeModelForLegacy:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_text("fake", encoding="utf-8")


class _FakeTrainerForLegacy:
    def __init__(self, model, args, train_dataset, eval_dataset, data_collator):
        del eval_dataset, data_collator
        self.model = model
        self.args = args
        self.train_dataset = train_dataset

    def train(self):
        return None

    def save_model(self, output_dir=None):
        self.model.save_pretrained(output_dir or self.args.output_dir)


class _FakeRecipeTrainerForLegacy(_FakeTrainerForLegacy):
    def __init__(self, *args, **kwargs):
        kwargs.pop("num_stable_steps", None)
        kwargs.pop("num_decay_steps", None)
        kwargs.pop("min_lr_ratio", None)
        kwargs.pop("train_batch_sampler", None)
        super().__init__(*args, **kwargs)


def _install_fake_legacy_runtime(monkeypatch):
    def fake_instantiate_model_and_tokenizer(model_path, *, bf16, fp16, tokenizer_mode="auto", recipe="none"):
        del model_path, bf16, fp16, tokenizer_mode, recipe
        return _FakeModelForLegacy(), _FakeTokenizerForLegacy()

    def fake_generate_prediction_map(model, tokenizer, examples, batch_size, max_new_tokens):
        del model, tokenizer, batch_size, max_new_tokens
        return {legacy.example_key(example): str(example.result) for example in examples}

    def fake_evaluate_accuracy_with_breakdown(model, tokenizer, examples, batch_size, max_new_tokens):
        del model, tokenizer, batch_size, max_new_tokens
        return 1.0, {example.digits: 1.0 for example in examples}

    monkeypatch.setattr(legacy, "instantiate_model_and_tokenizer", fake_instantiate_model_and_tokenizer)
    monkeypatch.setattr(legacy, "generate_prediction_map", fake_generate_prediction_map)
    monkeypatch.setattr(legacy, "evaluate_accuracy_with_breakdown", fake_evaluate_accuracy_with_breakdown)
    monkeypatch.setattr(legacy, "Trainer", _FakeTrainerForLegacy)


def _install_fake_recipe_runtime(monkeypatch):
    _install_fake_legacy_runtime(monkeypatch)
    monkeypatch.setattr(legacy, "WarmupStableDecayTrainer", _FakeRecipeTrainerForLegacy)
    monkeypatch.setattr(legacy, "BatchSamplerWarmupStableDecayTrainer", _FakeRecipeTrainerForLegacy)
    monkeypatch.setattr(
        legacy,
        "collect_prediction_debug_rows",
        lambda **kwargs: [
            {
                "prompt": kwargs["examples"][0].prompt() if kwargs["examples"] else "",
                "gold_target": str(kwargs["examples"][0].result) if kwargs["examples"] else None,
                "pseudo_target": kwargs["examples"][0].target_override if kwargs["examples"] else None,
                "decoded_output": kwargs["examples"][0].target() if kwargs["examples"] else "",
                "parsed_prediction": kwargs["examples"][0].target() if kwargs["examples"] else None,
                "correct_vs_gold": True,
                "correct_vs_pseudo": True,
                "pseudo_matches_gold": True,
            }
        ]
        if kwargs["examples"]
        else [],
    )


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _run_direct_pseudo_experiment(
    tmp_path: Path,
    monkeypatch,
    *,
    pseudo_label_mode: str,
    composed_strategy: str = "with_carry_filtered",
    corruption_rate: float = 0.0,
    seed_replay_train_per_digit: int = 2,
    expand_train_per_digit: int = 2,
    output_dir: Path | None = None,
    extra_args: list[str] | None = None,
    use_recipe: bool = False,
) -> Path:
    if use_recipe:
        _install_fake_recipe_runtime(monkeypatch)
    else:
        _install_fake_legacy_runtime(monkeypatch)
    output_dir = output_dir or (tmp_path / f"run_{pseudo_label_mode}")
    args = [
        "--model-name",
        "fake-model",
        "--output-dir",
        str(output_dir),
        "--treat-seed-as-round-zero",
        "--seed-range-train-mode",
        "direct_pseudo",
        "--initial-min-digits",
        "3",
        "--initial-max-digits",
        "7",
        "--initial-train-per-digit",
        "0",
        "--initial-eval-per-digit",
        "1",
        "--num-expand-rounds",
        "1",
        "--expand-num-digits",
        "5",
        "--seed-replay-train-per-digit",
        str(seed_replay_train_per_digit),
        "--expand-train-per-digit",
        str(expand_train_per_digit),
        "--eval-per-digit",
        "1",
        "--composed-eval-per-digit",
        "1",
        "--per-device-train-batch-size",
        "2",
        "--per-device-eval-batch-size",
        "2",
        "--pseudo-label-mode",
        pseudo_label_mode,
        "--composed-strategy",
        composed_strategy,
        "--composed-refresh-mode",
        "dynamic",
    ]
    if use_recipe:
        args.extend(["--recipe", "arithmetic_self_improve_v1"])
    if corruption_rate:
        args.extend(["--corruption-rate", str(corruption_rate)])
    if extra_args:
        args.extend(extra_args)
    legacy.main(args)
    return output_dir


def test_direct_pseudo_requires_treat_seed_as_round_zero():
    with pytest.raises(ValueError, match="treat-seed-as-round-zero"):
        legacy.main(
            [
                "--seed-range-train-mode",
                "direct_pseudo",
            ]
        )


def test_direct_pseudo_rejects_static_composed_refresh():
    with pytest.raises(ValueError, match="composed-refresh-mode dynamic"):
        legacy.main(
            [
                "--treat-seed-as-round-zero",
                "--seed-range-train-mode",
                "direct_pseudo",
                "--composed-refresh-mode",
                "static",
            ]
        )


@pytest.mark.parametrize(
    ("pseudo_label_mode", "expected_seed", "expected_expansion"),
    [
        ("none", 10, 0),
        ("direct", 10, 10),
    ],
)
def test_direct_pseudo_round_one_uses_expected_seed_and_expansion_counts(
    tmp_path: Path,
    monkeypatch,
    pseudo_label_mode: str,
    expected_seed: int,
    expected_expansion: int,
):
    output_dir = _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode=pseudo_label_mode,
        composed_strategy="with_carry",
    )

    round_00 = output_dir / "round_00"
    round_01 = output_dir / "round_01"
    metrics = json.loads((round_01 / "metrics.json").read_text(encoding="utf-8"))

    assert _count_jsonl_rows(round_00 / "seed_replay_pseudo_for_next_round.jsonl") == expected_seed
    assert _count_jsonl_rows(round_00 / "expansion_pseudo_for_next_round.jsonl") == expected_expansion
    assert _count_jsonl_rows(round_01 / "train_examples.jsonl") == expected_seed + expected_expansion
    assert metrics["supervised_examples"] == 0
    assert metrics["seed_replay_pseudo_examples"] == expected_seed
    assert metrics["expansion_pseudo_examples"] == expected_expansion


def test_direct_pseudo_compose_filtered_round_one_excludes_supervised_seed_data(tmp_path: Path, monkeypatch):
    output_dir = _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode="compose",
        composed_strategy="with_carry_filtered",
    )

    round_00 = output_dir / "round_00"
    round_01 = output_dir / "round_01"
    round_00_metrics = json.loads((round_00 / "metrics.json").read_text(encoding="utf-8"))
    metrics = json.loads((round_01 / "metrics.json").read_text(encoding="utf-8"))
    train_examples = [json.loads(line) for line in (round_01 / "train_examples.jsonl").read_text(encoding="utf-8").splitlines()]

    assert _count_jsonl_rows(output_dir / "data" / "initial_train.jsonl") == 0
    assert _count_jsonl_rows(round_00 / "pseudo_for_next_round.jsonl") == 20
    assert _count_jsonl_rows(round_00 / "seed_replay_pseudo_for_next_round.jsonl") == 10
    assert _count_jsonl_rows(round_00 / "expansion_pseudo_for_next_round.jsonl") == 10
    assert all(example["target_override"] is not None for example in train_examples)
    assert metrics["supervised_examples"] == 0
    assert metrics["seed_replay_pseudo_examples"] == 10
    assert metrics["expansion_pseudo_examples"] == 10
    assert round_00_metrics["pseudo_generation_stats"]["seed_replay"]["retained_total"] == 10
    assert round_00_metrics["pseudo_generation_stats"]["expansion"]["retained_total"] == 10
    assert json.loads((output_dir / "data" / "metadata.json").read_text(encoding="utf-8"))["composed_eval_support_split"] == "validation"


def test_direct_pseudo_can_use_larger_expansion_pool_than_seed_replay(tmp_path: Path, monkeypatch):
    output_dir = _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode="compose",
        composed_strategy="with_carry_filtered",
        seed_replay_train_per_digit=2,
        expand_train_per_digit=4,
    )

    round_00 = output_dir / "round_00"
    round_01 = output_dir / "round_01"
    round_00_metrics = json.loads((round_00 / "metrics.json").read_text(encoding="utf-8"))
    round_01_metrics = json.loads((round_01 / "metrics.json").read_text(encoding="utf-8"))

    assert _count_jsonl_rows(round_00 / "seed_replay_pseudo_for_next_round.jsonl") == 10
    assert _count_jsonl_rows(round_00 / "expansion_pseudo_for_next_round.jsonl") == 20
    assert _count_jsonl_rows(round_01 / "train_examples.jsonl") == 30
    assert round_01_metrics["supervised_examples"] == 0
    assert round_01_metrics["seed_replay_pseudo_examples"] == 10
    assert round_01_metrics["expansion_pseudo_examples"] == 20
    assert round_00_metrics["pseudo_generation_stats"]["seed_replay"]["candidate_total"] == 10
    assert round_00_metrics["pseudo_generation_stats"]["expansion"]["candidate_total"] == 20


def test_direct_pseudo_resume_uses_combined_pseudo_dataset_only(tmp_path: Path, monkeypatch):
    output_dir = _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode="compose",
        composed_strategy="with_carry_filtered",
    )

    round_00 = output_dir / "round_00"
    (round_00 / "seed_replay_pseudo_for_next_round.jsonl").unlink()
    (round_00 / "expansion_pseudo_for_next_round.jsonl").unlink()

    _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode="compose",
        composed_strategy="with_carry_filtered",
        output_dir=output_dir,
        extra_args=["--resume-from-round", "1"],
    )

    metrics = json.loads((output_dir / "round_01" / "metrics.json").read_text(encoding="utf-8"))
    assert _count_jsonl_rows(output_dir / "round_01" / "train_examples.jsonl") == 20
    assert metrics["supervised_examples"] == 0
    assert metrics["seed_replay_pseudo_examples"] == 10
    assert metrics["expansion_pseudo_examples"] == 10


def test_direct_and_composed_share_prompt_and_gold_target_for_same_arithmetic_example():
    left = AdditionExample(a=12, b=34, result=46, digits=2, has_carry=False)
    right = AdditionExample(a=56, b=78, result=134, digits=2, has_carry=True)
    composed = compose_examples(left, right)
    direct = AdditionExample(
        a=composed.a,
        b=composed.b,
        result=composed.result,
        digits=composed.digits,
        has_carry=composed.has_carry,
    )

    assert composed.prompt() == direct.prompt()
    assert str(composed.result) == str(direct.result)
    assert composed.target() == direct.target()


def test_recipe_backed_direct_pseudo_smoke_writes_debug_artifacts(tmp_path: Path, monkeypatch):
    output_dir = _run_direct_pseudo_experiment(
        tmp_path,
        monkeypatch,
        pseudo_label_mode="compose",
        composed_strategy="with_carry_filtered",
        use_recipe=True,
    )

    round_00 = output_dir / "round_00"
    round_01 = output_dir / "round_01"
    metrics = json.loads((round_01 / "metrics.json").read_text(encoding="utf-8"))

    assert (round_00 / "seed_replay_debug_predictions.jsonl").exists()
    assert (round_00 / "expansion_debug_predictions.jsonl").exists()
    assert (round_01 / "frontier_train_debug_predictions.jsonl").exists()
    assert _count_jsonl_rows(round_00 / "seed_replay_debug_predictions.jsonl") > 0
    assert _count_jsonl_rows(round_00 / "expansion_debug_predictions.jsonl") > 0
    assert metrics["supervised_examples"] == 0
    assert metrics["seed_replay_pseudo_examples"] == 10
    assert metrics["expansion_pseudo_examples"] == 10

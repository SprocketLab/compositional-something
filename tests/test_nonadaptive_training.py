from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from self.nonadaptive.nonadaptive_training import train_nonadaptive_round_model


class _Model:
    def __init__(self) -> None:
        self.saved_paths = []

    def save_pretrained(self, path: Path) -> None:
        self.saved_paths.append(Path(path))


class _Tokenizer:
    def __init__(self) -> None:
        self.saved_paths = []

    def save_pretrained(self, path: Path) -> None:
        self.saved_paths.append(Path(path))


class _Task:
    @staticmethod
    def size_of(example) -> int:
        return int(example)


class _Trainer:
    def __init__(self, model="trained-model") -> None:
        self.model = model
        self.train_called = False
        self.save_paths = []

    def train(self) -> None:
        self.train_called = True

    def save_model(self, output_dir=None) -> None:
        self.save_paths.append(output_dir)


def _args(**overrides):
    args = dict(
        bf16=False,
        fp16=False,
        keep_checkpoints=False,
        seed=11,
        treat_seed_as_round_zero=False,
        bucket_train_batches_by_bits=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_train_nonadaptive_round_model_skips_seed_round_without_building_trainer(tmp_path: Path):
    model = _Model()
    tokenizer = _Tokenizer()

    result = train_nonadaptive_round_model(
        args=_args(treat_seed_as_round_zero=True),
        task=_Task(),
        model=model,
        tokenizer=tokenizer,
        train_examples=[1],
        round_dir=tmp_path,
        config=SimpleNamespace(),
        data_collator="collator",
        round_idx=0,
        new_run=True,
        save_model_this_round=True,
        use_recipe=True,
        recipe_name="recipe",
        dataset_cls=lambda *args, **kwargs: pytest.fail("dataset should not be built"),
        make_training_args_fn=lambda *args, **kwargs: pytest.fail("training args should not be built"),
        build_trainer_fn=lambda *args, **kwargs: pytest.fail("trainer should not be built"),
    )

    assert result.skipped is True
    assert result.trainer is None
    assert result.model is model
    assert result.recipe_phase_name == "self_improve"
    assert model.saved_paths == [tmp_path]
    assert tokenizer.saved_paths == [tmp_path]


def test_train_nonadaptive_round_model_builds_recipe_trainer_with_overrides(tmp_path: Path):
    records = {}
    trainer = _Trainer()
    tokenizer = _Tokenizer()

    def dataset_cls(examples, tokenizer_arg):
        records["dataset"] = (list(examples), tokenizer_arg)
        return "dataset"

    def make_training_args_fn(*args, **kwargs):
        records["make_training_args"] = (args, kwargs)
        return "training-args"

    def build_trainer_fn(**kwargs):
        records["build_trainer"] = kwargs
        return trainer

    args = _args(
        bf16=True,
        keep_checkpoints=True,
        self_improve_learning_rate=0.1,
        self_improve_lr_switch_round=3,
        self_improve_learning_rate_after_switch=0.2,
        self_improve_warmup_steps=5,
        bucket_train_batches_by_bits=True,
    )

    result = train_nonadaptive_round_model(
        args=args,
        task=_Task(),
        model="model",
        tokenizer=tokenizer,
        train_examples=[2, 3],
        round_dir=tmp_path,
        config="config",
        data_collator="collator",
        round_idx=3,
        new_run=False,
        save_model_this_round=True,
        use_recipe=True,
        recipe_name="recipe",
        dataset_cls=dataset_cls,
        make_training_args_fn=make_training_args_fn,
        build_trainer_fn=build_trainer_fn,
    )

    expected_overrides = {"learning_rate": 0.2, "warmup_steps": 5}
    assert records["dataset"] == ([2, 3], tokenizer)
    assert records["make_training_args"] == (
        (tmp_path, "config"),
        {
            "bf16": True,
            "fp16": False,
            "skip_save": False,
            "keep_checkpoints": True,
            "seed": 11,
            "recipe": "recipe",
            "recipe_phase_name": "self_improve",
            "recipe_phase_overrides": expected_overrides,
        },
    )
    assert records["build_trainer"]["model"] == "model"
    assert records["build_trainer"]["training_args"] == "training-args"
    assert records["build_trainer"]["train_dataset"] == "dataset"
    assert records["build_trainer"]["data_collator"] == "collator"
    assert records["build_trainer"]["seed"] == 11 + 3 * 9973
    assert records["build_trainer"]["size_getter"] is _Task.size_of
    assert records["build_trainer"]["bucket_train_batches_by_size"] is True
    assert records["build_trainer"]["recipe_phase_overrides"] == expected_overrides
    assert trainer.train_called is True
    assert trainer.save_paths == [str(tmp_path)]
    assert tokenizer.saved_paths == [tmp_path]
    assert result.model == "trained-model"
    assert result.trainer is trainer
    assert result.skipped is False

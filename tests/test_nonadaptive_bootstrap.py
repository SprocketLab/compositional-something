from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from self.core.nonadaptive_bootstrap import prepare_nonadaptive_bootstrap


@dataclass(frozen=True)
class _Example:
    bits: int


class _Task:
    def token_initializers(self, args):
        del args
        return {"<x>": "seed"}

    def deserialize_example(self, payload):
        return _Example(bits=int(payload["bits"]))


class _TrainingConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DefaultCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _RecipeCollator:
    def __init__(self, *, tokenizer, padding_side):
        self.tokenizer = tokenizer
        self.padding_side = padding_side


def _args(**overrides):
    args = dict(
        model_name="base-model",
        bf16=False,
        fp16=False,
        init_from_scratch=False,
        tokenizer_mode="auto",
        num_epochs=2,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        weight_decay=0.01,
        logging_steps=10,
        max_steps=-1,
        eval_steps=0,
        decode_max_new_tokens=6,
        resume_from_round=None,
        num_expand_rounds=3,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_prepare_nonadaptive_bootstrap_initializes_base_model_and_config(tmp_path: Path):
    instantiate_calls = []

    def instantiate(path, **kwargs):
        instantiate_calls.append((path, kwargs))
        return "model", "tokenizer"

    bootstrap = prepare_nonadaptive_bootstrap(
        _args(),
        _Task(),
        base_output_dir=tmp_path,
        base_train_examples=[_Example(1), _Example(2)],
        eval_examples=[_Example(3)],
        composed_eval_examples=[],
        existing_summaries={0: {"round": 0}},
        resume_requested=False,
        reset_each_round=False,
        use_recipe=False,
        recipe_name="none",
        instantiate_model_and_tokenizer_fn=instantiate,
        training_config_cls=_TrainingConfig,
        resolve_max_new_tokens_fn=lambda examples, default: default + len(examples),
        default_collator_cls=_DefaultCollator,
        recipe_collator_cls=_RecipeCollator,
    )

    assert bootstrap.resume_round == 0
    assert instantiate_calls == [
        (
            "base-model",
            {
                "bf16": False,
                "fp16": False,
                "token_initializers": {"<x>": "seed"},
                "init_from_scratch": False,
                "tokenizer_mode": "auto",
                "recipe": "none",
            },
        )
    ]
    assert bootstrap.model == "model"
    assert bootstrap.tokenizer == "tokenizer"
    assert bootstrap.config.max_steps is None
    assert bootstrap.config.eval_steps is None
    assert bootstrap.train_base_decode_tokens == 8
    assert bootstrap.eval_decode_tokens == 7
    assert bootstrap.composed_eval_decode_tokens == 6
    assert isinstance(bootstrap.data_collator, _DefaultCollator)
    assert bootstrap.summary_records == {0: {"round": 0}}
    assert bootstrap.pseudo_examples == []


def test_prepare_nonadaptive_bootstrap_resumes_from_checkpoint_and_loads_pseudo_seed(tmp_path: Path):
    checkpoint_dir = tmp_path / "round_01"
    checkpoint_dir.mkdir()
    pseudo_path = checkpoint_dir / "pseudo_for_next_round.jsonl"
    pseudo_path.write_text('{"bits": 5}\n', encoding="utf-8")
    existing_summaries = {0: {"round": 0}, 1: {"round": 1}, 2: {"round": 2}}
    instantiate_paths = []

    def instantiate(path, **kwargs):
        del kwargs
        instantiate_paths.append(path)
        return "model", "tokenizer"

    def load_examples(path, deserializer):
        assert path == pseudo_path
        return [deserializer({"bits": 5})]

    bootstrap = prepare_nonadaptive_bootstrap(
        _args(resume_from_round=2, max_steps=12, eval_steps=3),
        _Task(),
        base_output_dir=tmp_path,
        base_train_examples=[],
        eval_examples=[],
        composed_eval_examples=[],
        existing_summaries=existing_summaries,
        resume_requested=True,
        reset_each_round=False,
        use_recipe=True,
        recipe_name="recipe-v1",
        load_examples_fn=load_examples,
        instantiate_model_and_tokenizer_fn=instantiate,
        training_config_cls=_TrainingConfig,
        resolve_max_new_tokens_fn=lambda examples, default: default,
        default_collator_cls=_DefaultCollator,
        recipe_collator_cls=_RecipeCollator,
    )

    assert bootstrap.resume_round == 2
    assert instantiate_paths == [str(checkpoint_dir)]
    assert existing_summaries == {0: {"round": 0}, 1: {"round": 1}}
    assert bootstrap.summary_records == {0: {"round": 0}, 1: {"round": 1}}
    assert bootstrap.config.max_steps == 12
    assert bootstrap.config.eval_steps == 3
    assert isinstance(bootstrap.data_collator, _RecipeCollator)
    assert bootstrap.data_collator.padding_side == "right"
    assert bootstrap.pseudo_examples == [_Example(bits=5)]

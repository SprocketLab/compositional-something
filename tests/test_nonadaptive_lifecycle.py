from __future__ import annotations

from types import SimpleNamespace

import pytest

from self.core.nonadaptive_lifecycle import NonAdaptiveRoundResources, finish_nonadaptive_round


def _args(**overrides):
    args = dict(
        num_expand_rounds=2,
        init_from_scratch=False,
        model_name="model-name",
        bf16=False,
        fp16=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _fail_loader(*args, **kwargs):
    del args, kwargs
    pytest.fail("loader should not be called")


def test_finish_nonadaptive_round_stop_after_releases_trainer_and_breaks(capsys):
    clears = []
    resources = NonAdaptiveRoundResources(model="model", trainer="trainer")

    result = finish_nonadaptive_round(
        args=_args(),
        tokenizer="tokenizer",
        resources=resources,
        round_idx=1,
        stop_after_round=1,
        reset_each_round=True,
        use_recipe=True,
        recipe_preset="preset",
        cuda_is_available_fn=lambda: True,
        empty_cache_fn=lambda: clears.append("cleared"),
        instantiate_recipe_model_fn=_fail_loader,
        load_recipe_model_fn=_fail_loader,
        load_model_for_tokenizer_fn=_fail_loader,
    )

    assert result.should_break is True
    assert result.should_continue is False
    assert resources.model == "model"
    assert resources.trainer is None
    assert clears == ["cleared"]
    assert "Stop-after-round reached at round 1" in capsys.readouterr().out


def test_finish_nonadaptive_round_final_round_releases_trainer_and_continues():
    clears = []
    resources = NonAdaptiveRoundResources(model="model", trainer="trainer")

    result = finish_nonadaptive_round(
        args=_args(num_expand_rounds=2),
        tokenizer="tokenizer",
        resources=resources,
        round_idx=2,
        stop_after_round=None,
        reset_each_round=True,
        use_recipe=True,
        recipe_preset="preset",
        cuda_is_available_fn=lambda: True,
        empty_cache_fn=lambda: clears.append("cleared"),
        instantiate_recipe_model_fn=_fail_loader,
        load_recipe_model_fn=_fail_loader,
        load_model_for_tokenizer_fn=_fail_loader,
    )

    assert result.should_break is False
    assert result.should_continue is True
    assert resources.model == "model"
    assert resources.trainer is None
    assert clears == ["cleared"]


def test_finish_nonadaptive_round_reset_recipe_scratch_releases_before_loading():
    events = []
    resources = NonAdaptiveRoundResources(model="old-model", trainer="trainer")

    def empty_cache():
        events.append(("clear", resources.model, resources.trainer))

    def instantiate(tokenizer, preset, *, bf16, fp16):
        events.append(("instantiate", resources.model, resources.trainer, tokenizer, preset, bf16, fp16))
        return "new-recipe-model"

    result = finish_nonadaptive_round(
        args=_args(init_from_scratch=True, bf16=True),
        tokenizer="tokenizer",
        resources=resources,
        round_idx=0,
        stop_after_round=None,
        reset_each_round=True,
        use_recipe=True,
        recipe_preset="preset",
        cuda_is_available_fn=lambda: True,
        empty_cache_fn=empty_cache,
        instantiate_recipe_model_fn=instantiate,
        load_recipe_model_fn=_fail_loader,
        load_model_for_tokenizer_fn=_fail_loader,
    )

    assert result.should_break is False
    assert result.should_continue is False
    assert events == [
        ("clear", None, None),
        ("instantiate", None, None, "tokenizer", "preset", True, False),
    ]
    assert resources.model == "new-recipe-model"
    assert resources.trainer is None


def test_finish_nonadaptive_round_reset_nonrecipe_reloads_base_model():
    calls = []
    resources = NonAdaptiveRoundResources(model="old-model", trainer="trainer")

    def load_model_for_tokenizer(model_name, tokenizer, *, bf16, fp16):
        calls.append((resources.model, resources.trainer, model_name, tokenizer, bf16, fp16))
        return "new-base-model"

    result = finish_nonadaptive_round(
        args=_args(model_name="base-model", fp16=True),
        tokenizer="tokenizer",
        resources=resources,
        round_idx=0,
        stop_after_round=None,
        reset_each_round=True,
        use_recipe=False,
        recipe_preset=None,
        cuda_is_available_fn=lambda: False,
        empty_cache_fn=lambda: pytest.fail("cache should not be cleared when CUDA is unavailable"),
        instantiate_recipe_model_fn=_fail_loader,
        load_recipe_model_fn=_fail_loader,
        load_model_for_tokenizer_fn=load_model_for_tokenizer,
    )

    assert result.should_break is False
    assert result.should_continue is False
    assert calls == [(None, None, "base-model", "tokenizer", False, True)]
    assert resources.model == "new-base-model"
    assert resources.trainer is None


def test_finish_nonadaptive_round_missing_recipe_checkpoint_raises_after_release(tmp_path):
    clears = []
    resources = NonAdaptiveRoundResources(model="old-model", trainer="trainer")
    missing_model = tmp_path / "missing-model"

    with pytest.raises(FileNotFoundError, match="Recipe-backed reset-in-each-round"):
        finish_nonadaptive_round(
            args=_args(model_name=str(missing_model)),
            tokenizer="tokenizer",
            resources=resources,
            round_idx=0,
            stop_after_round=None,
            reset_each_round=True,
            use_recipe=True,
            recipe_preset="preset",
            cuda_is_available_fn=lambda: True,
            empty_cache_fn=lambda: clears.append("cleared"),
            instantiate_recipe_model_fn=_fail_loader,
            load_recipe_model_fn=_fail_loader,
            load_model_for_tokenizer_fn=_fail_loader,
        )

    assert resources.model is None
    assert resources.trainer is None
    assert clears == ["cleared"]

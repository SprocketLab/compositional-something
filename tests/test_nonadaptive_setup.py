from __future__ import annotations

from types import SimpleNamespace

import pytest

from self.core.nonadaptive_setup import prepare_nonadaptive_run_setup
from self.core.recipes import RECIPE_ALGORITHMIC_SELF_IMPROVE_V1


class _SetupTask:
    def __init__(self) -> None:
        self.validated = False

    def validate_args(self, args) -> None:
        del args
        self.validated = True


def _base_args(**overrides):
    args = dict(
        bf16=False,
        fp16=False,
        initial_min_size=4,
        initial_max_size=8,
        eval_per_size=2,
        composed_eval_per_size=0,
        expand_num_size=4,
        num_expand_rounds=1,
        resume_from_round=None,
        stop_after_round=None,
        skip_save_model=False,
        save_model_policy="all_rounds",
        frontier_min_size=None,
        recipe="none",
        tokenizer_mode="auto",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        composed_refresh_mode="dynamic",
        reset_in_each_round=False,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_prepare_nonadaptive_setup_validates_stop_after_resume_order():
    args = _base_args(resume_from_round=2, stop_after_round=1)

    with pytest.raises(ValueError, match="stop_after_round"):
        prepare_nonadaptive_run_setup(args, _SetupTask(), cuda_available_fn=lambda: False)


def test_prepare_nonadaptive_setup_sets_skip_save_from_policy_none():
    args = _base_args(save_model_policy="none")
    task = _SetupTask()

    setup = prepare_nonadaptive_run_setup(args, task, cuda_available_fn=lambda: False)

    assert task.validated is True
    assert setup.save_model_policy == "none"
    assert args.skip_save_model is True
    assert setup.final_max_size == 12
    assert setup.composed_min_size == 9


def test_prepare_nonadaptive_setup_defaults_to_bf16_when_cuda_available(capsys):
    args = _base_args()

    setup = prepare_nonadaptive_run_setup(args, _SetupTask(), cuda_available_fn=lambda: True)

    assert args.bf16 is True
    assert setup.use_recipe is False
    assert "defaulting to bf16 on CUDA" in capsys.readouterr().out


def test_prepare_nonadaptive_setup_applies_recipe_defaults(capsys):
    args = _base_args(
        recipe=RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
        tokenizer_mode="character",
    )

    setup = prepare_nonadaptive_run_setup(args, _SetupTask(), cuda_available_fn=lambda: False)

    assert setup.use_recipe is True
    assert setup.recipe_preset is not None
    assert args.bf16 is True
    assert args.per_device_train_batch_size == 1024
    assert args.per_device_eval_batch_size == 1024
    assert "ignores --tokenizer-mode" in capsys.readouterr().out


def test_prepare_nonadaptive_setup_keeps_explicit_recipe_batch_sizes():
    args = _base_args(
        recipe=RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
    )

    prepare_nonadaptive_run_setup(args, _SetupTask(), cuda_available_fn=lambda: False)

    assert args.per_device_train_batch_size == 16
    assert args.per_device_eval_batch_size == 32

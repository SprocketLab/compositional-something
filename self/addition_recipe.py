#!/usr/bin/env python3
"""Compatibility wrapper for addition recipe helpers in :mod:`self.core.recipes`."""

from __future__ import annotations

import sys as _sys
import types as _types

from self.core import recipes as _impl


_EXPORT_NAMES = [
    "AdditionRecipePreset",
    "BatchSamplerWarmupStableDecayTrainer",
    "NoPELlamaForCausalLM",
    "PaddingAwareCausalLMDataCollator",
    "RECIPE_ARITHMETIC_SELF_IMPROVE_V1",
    "RecipeTrainingPhaseConfig",
    "WarmupStableDecayTrainer",
    "apply_recipe_runtime_settings",
    "build_recipe_model_config",
    "build_recipe_tokenizer",
    "instantiate_recipe_model",
    "load_recipe_model",
    "make_recipe_training_args",
    "make_warmup_stable_decay_lambda",
    "resolve_addition_recipe",
    "resolve_recipe_phase",
    "tokenizer_padding_side",
]


def __getattr__(name: str):
    return getattr(_impl, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))


class _ModuleProxy(_types.ModuleType):
    def __getattr__(self, name: str):
        return getattr(_impl, name)

    def __setattr__(self, name: str, value):
        if not name.startswith("__"):
            setattr(_impl, name, value)
        super().__setattr__(name, value)


__all__ = list(_EXPORT_NAMES)
_sys.modules[__name__].__class__ = _ModuleProxy

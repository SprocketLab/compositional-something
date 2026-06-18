#!/usr/bin/env python3
"""Compatibility wrapper for addition recipe helpers in :mod:`self.core.recipes`."""

from __future__ import annotations

from self.core.module_proxy import install_module_proxy
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


install_module_proxy(__name__, _impl, export_names=_EXPORT_NAMES)

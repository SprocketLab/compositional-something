#!/usr/bin/env python3
"""Compatibility facade for recipe presets, models, and training helpers."""

from __future__ import annotations

from self.core.recipe_models import (
    NoPELlamaForCausalLM,
    NoPositionRotaryEmbedding,
    apply_recipe_runtime_settings,
    build_recipe_model_config,
    build_recipe_tokenizer,
    instantiate_recipe_model,
    load_recipe_model,
    sync_model_special_token_ids,
    tokenizer_padding_side,
)
from self.core.recipe_presets import (
    AdditionRecipePreset,
    RECIPE_ALGORITHMIC_SELF_IMPROVE_V1,
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    RECIPE_MULTIPLICATION_SELF_IMPROVE_V1,
    SUPPORTED_SELF_IMPROVEMENT_RECIPES,
    RecipeTrainingPhaseConfig,
    SelfImprovementRecipePreset,
    fit_recipe_phase_to_max_steps,
    recipe_enabled,
    resolve_addition_recipe,
    resolve_recipe_phase,
    resolve_self_improvement_recipe,
)
from self.core.recipe_training import (
    TRAINING_ARGUMENT_FIELDS,
    BatchSamplerWarmupStableDecayTrainer,
    PaddingAwareCausalLMDataCollator,
    WarmupStableDecayTrainer,
    make_recipe_training_args,
    make_warmup_stable_decay_lambda,
    training_arg_supported,
)


__all__ = [
    "AdditionRecipePreset",
    "BatchSamplerWarmupStableDecayTrainer",
    "NoPELlamaForCausalLM",
    "NoPositionRotaryEmbedding",
    "PaddingAwareCausalLMDataCollator",
    "RECIPE_ALGORITHMIC_SELF_IMPROVE_V1",
    "RECIPE_ARITHMETIC_SELF_IMPROVE_V1",
    "RECIPE_MULTIPLICATION_SELF_IMPROVE_V1",
    "SUPPORTED_SELF_IMPROVEMENT_RECIPES",
    "RecipeTrainingPhaseConfig",
    "SelfImprovementRecipePreset",
    "TRAINING_ARGUMENT_FIELDS",
    "WarmupStableDecayTrainer",
    "apply_recipe_runtime_settings",
    "build_recipe_model_config",
    "build_recipe_tokenizer",
    "fit_recipe_phase_to_max_steps",
    "instantiate_recipe_model",
    "load_recipe_model",
    "make_recipe_training_args",
    "make_warmup_stable_decay_lambda",
    "recipe_enabled",
    "resolve_addition_recipe",
    "resolve_recipe_phase",
    "resolve_self_improvement_recipe",
    "sync_model_special_token_ids",
    "tokenizer_padding_side",
    "training_arg_supported",
]

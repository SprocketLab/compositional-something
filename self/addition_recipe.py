#!/usr/bin/env python3
"""Backward-compatible addition recipe wrapper around shared recipe helpers."""

from __future__ import annotations

from self.core.recipes import (
    RECIPE_ARITHMETIC_SELF_IMPROVE_V1,
    BatchSamplerWarmupStableDecayTrainer,
    NoPELlamaForCausalLM,
    PaddingAwareCausalLMDataCollator,
    RecipeTrainingPhaseConfig,
    SelfImprovementRecipePreset,
    WarmupStableDecayTrainer,
    apply_recipe_runtime_settings,
    build_recipe_model_config,
    build_recipe_tokenizer,
    instantiate_recipe_model,
    load_recipe_model,
    make_recipe_training_args,
    make_warmup_stable_decay_lambda,
    resolve_recipe_phase,
    resolve_self_improvement_recipe,
    tokenizer_padding_side,
)


AdditionRecipePreset = SelfImprovementRecipePreset


def resolve_addition_recipe(name: str) -> AdditionRecipePreset:
    if name != RECIPE_ARITHMETIC_SELF_IMPROVE_V1:
        raise ValueError(f"Unsupported addition recipe: {name!r}")
    return resolve_self_improvement_recipe(name)


__all__ = [
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

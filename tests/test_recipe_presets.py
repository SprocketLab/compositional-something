from __future__ import annotations

from self import self_improvement_recipe
from self.core import recipe_models, recipe_presets, recipe_training, recipes


def test_recipe_preset_owner_reexports() -> None:
    assert recipes.RecipeTrainingPhaseConfig is recipe_presets.RecipeTrainingPhaseConfig
    assert recipes.SelfImprovementRecipePreset is recipe_presets.SelfImprovementRecipePreset
    assert recipes.resolve_self_improvement_recipe is recipe_presets.resolve_self_improvement_recipe
    assert recipes.resolve_addition_recipe is recipe_presets.resolve_addition_recipe
    assert recipes.resolve_recipe_phase is recipe_presets.resolve_recipe_phase
    assert recipes.fit_recipe_phase_to_max_steps is recipe_presets.fit_recipe_phase_to_max_steps
    assert recipes.recipe_enabled is recipe_presets.recipe_enabled
    assert (
        self_improvement_recipe.resolve_self_improvement_recipe
        is recipe_presets.resolve_self_improvement_recipe
    )


def test_recipe_preset_schedule_compression_stays_canonical() -> None:
    preset = recipe_presets.resolve_self_improvement_recipe(
        recipe_presets.RECIPE_MULTIPLICATION_SELF_IMPROVE_V1
    )
    compressed = recipe_presets.fit_recipe_phase_to_max_steps(preset.seed_phase, max_steps=1000)

    assert compressed.warmup_steps == 100
    assert compressed.num_stable_steps == 700
    assert compressed.num_decay_steps == 200


def test_recipe_runtime_owner_reexports() -> None:
    assert recipes.NoPELlamaForCausalLM is recipe_models.NoPELlamaForCausalLM
    assert recipes.NoPositionRotaryEmbedding is recipe_models.NoPositionRotaryEmbedding
    assert recipes.build_recipe_tokenizer is recipe_models.build_recipe_tokenizer
    assert recipes.instantiate_recipe_model is recipe_models.instantiate_recipe_model
    assert recipes.load_recipe_model is recipe_models.load_recipe_model
    assert recipes.tokenizer_padding_side is recipe_models.tokenizer_padding_side


def test_recipe_training_owner_reexports() -> None:
    assert recipes.PaddingAwareCausalLMDataCollator is recipe_training.PaddingAwareCausalLMDataCollator
    assert recipes.WarmupStableDecayTrainer is recipe_training.WarmupStableDecayTrainer
    assert recipes.BatchSamplerWarmupStableDecayTrainer is recipe_training.BatchSamplerWarmupStableDecayTrainer
    assert recipes.make_recipe_training_args is recipe_training.make_recipe_training_args
    assert recipes.make_warmup_stable_decay_lambda is recipe_training.make_warmup_stable_decay_lambda
    assert recipes.training_arg_supported is recipe_training.training_arg_supported
    assert self_improvement_recipe.build_recipe_tokenizer is recipe_models.build_recipe_tokenizer

from __future__ import annotations

from self import self_improvement_recipe
from self.core import recipe_presets, recipes


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

from __future__ import annotations

from self.core import recipes


def test_recipe_exports_are_available_and_unique() -> None:
    assert len(recipes.__all__) == len(set(recipes.__all__))
    assert all(hasattr(recipes, name) for name in recipes.__all__)


def test_recipe_preset_schedule_compression_stays_canonical() -> None:
    preset = recipes.resolve_self_improvement_recipe(
        recipes.RECIPE_MULTIPLICATION_SELF_IMPROVE_V1
    )
    compressed = recipes.fit_recipe_phase_to_max_steps(preset.seed_phase, max_steps=1000)

    assert compressed.warmup_steps == 100
    assert compressed.num_stable_steps == 700
    assert compressed.num_decay_steps == 200

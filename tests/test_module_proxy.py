from __future__ import annotations

from self import addition_recipe
from self import multiplication_rectangular as legacy_rectangular
from self import self_improvement_experiment
from self.core import recipes
from self.legacy import addition_self_improvement
from self.tasks import rectangular_composition
from self.tasks import rectangular_multiplication


def test_module_proxy_exposes_canonical_rectangular_exports():
    assert (
        legacy_rectangular.RectangularCompositionLeaf
        is rectangular_composition.RectangularCompositionLeaf
    )
    assert "RectangularCompositionLeaf" in legacy_rectangular.__all__
    assert "RectangularCompositionLeaf" in dir(legacy_rectangular)


def test_module_proxy_forwards_monkeypatch_writes():
    original = rectangular_multiplication.sample_int_with_exact_digits

    def always_two(*args, **kwargs):
        return 2

    try:
        legacy_rectangular.sample_int_with_exact_digits = always_two
        assert rectangular_multiplication.sample_int_with_exact_digits is always_two
    finally:
        legacy_rectangular.sample_int_with_exact_digits = original


def test_module_proxy_preserves_explicit_export_lists():
    assert "resolve_addition_recipe" in addition_recipe.__all__
    assert "_impl" not in addition_recipe.__all__
    assert addition_recipe.resolve_addition_recipe is recipes.resolve_addition_recipe


def test_old_self_improvement_experiment_proxies_legacy_addition_module():
    original = addition_self_improvement.parse_args

    def fake_parse_args(*args, **kwargs):
        return ("parsed", args, kwargs)

    try:
        assert self_improvement_experiment.main is addition_self_improvement.main
        self_improvement_experiment.parse_args = fake_parse_args
        assert addition_self_improvement.parse_args is fake_parse_args
    finally:
        self_improvement_experiment.parse_args = original

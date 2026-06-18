from __future__ import annotations

import importlib

from self import addition_recipe
from self import multiplication_rectangular as legacy_rectangular
from self import self_improvement_experiment
from self.core import recipes
from self.core.module_proxy import module_star_export_names
from self.legacy import addition_self_improvement
from self.tasks import rectangular_composition
from self.tasks import rectangular_multiplication


STAR_PROXY_PAIRS = (
    ("self.addition_recipe_diagnostic", "self.diagnostics.addition_recipe_diagnostic"),
    ("self.analyze_symbolic_training_dynamics", "self.diagnostics.analyze_symbolic_training_dynamics"),
    ("self.check_self_improvement_overfit", "self.diagnostics.check_self_improvement_overfit"),
    ("self.evaluate_fixed_composition_slices", "self.diagnostics.evaluate_fixed_composition_slices"),
    ("self.figure2_condition_sweep", "self.experiments.figure2_condition_sweep"),
    ("self.figure2_paper_retune", "self.experiments.figure2_paper_retune"),
    ("self.figure3_real_seed_data_ablation", "self.experiments.figure3_real_seed_data_ablation"),
    ("self.figure3_seed_quality_sweep", "self.experiments.figure3_seed_quality_sweep"),
    ("self.multiplication_rectangular_tune", "self.experiments.multiplication_rectangular_tune"),
    ("self.multiplication_self_improvement", "self.legacy.multiplication_self_improvement"),
    ("self.paper_schedule_selection", "self.experiments.paper_schedule_selection"),
    ("self.plot_appendix_baseline_heatmaps", "self.analysis.plot_appendix_baseline_heatmaps"),
    ("self.plot_self_improvement_figure", "self.analysis.plot_self_improvement_figure"),
    (
        "self.rectangular_multiplication_compose_diagnostic",
        "self.diagnostics.rectangular_multiplication_compose_diagnostic",
    ),
    ("self.rectangular_multiplication_recipe_seed_fit", "self.experiments.rectangular_multiplication_recipe_seed_fit"),
    ("self.rectangular_multiplication_seed_fit", "self.experiments.rectangular_multiplication_seed_fit"),
    ("self.rectangular_multiplication_self_improvement", "self.experiments.rectangular_multiplication_self_improvement"),
    ("self.run_length_self_improvement", "self.legacy.run_length_self_improvement"),
    ("self.seed_fit_experiment", "self.experiments.seed_fit_experiment"),
    ("self.seed_fit_curve_notebook_utils", "self.analysis.seed_fit_curve_notebook_utils"),
    (
        "self.self_improvement_multiplication_cot_pseudo_addition",
        "self.legacy.multiplication_cot_pseudo_addition",
    ),
    ("self.summarize_seed_fit_grid", "self.analysis.summarize_seed_fit_grid"),
    ("self.training_curve_notebook_utils", "self.analysis.training_curve_notebook_utils"),
    ("self.run_length_balanced_eval", "self.diagnostics.run_length_balanced_eval"),
)


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


def test_module_star_export_names_matches_star_import_policy():
    class NoAll:
        public = object()
        _private = object()

    class WithAll:
        __all__ = ["public", "_explicit_private"]
        public = object()
        _explicit_private = object()

    assert "public" in module_star_export_names(NoAll)
    assert "_private" not in module_star_export_names(NoAll)
    assert module_star_export_names(WithAll) == ["public", "_explicit_private"]


def test_star_wrappers_proxy_canonical_modules():
    for wrapper_name, impl_name in STAR_PROXY_PAIRS:
        wrapper = importlib.import_module(wrapper_name)
        impl = importlib.import_module(impl_name)

        assert wrapper.__all__ == module_star_export_names(impl)
        if hasattr(impl, "main"):
            assert wrapper.main is impl.main

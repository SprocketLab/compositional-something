from self.core import composition, composition_program_pseudolabels, composition_pseudolabels
from self.adaptive.run import driver_compat_exports


def test_composition_pseudolabel_compat_reexports() -> None:
    assert composition.compose_pseudo_examples is composition_pseudolabels.compose_pseudo_examples
    assert composition.compose_addition_pseudo_examples is composition_pseudolabels.compose_addition_pseudo_examples
    assert composition.compose_run_length_pseudo_examples is composition_pseudolabels.compose_run_length_pseudo_examples
    assert composition.compose_program_pseudo_examples is composition_pseudolabels.compose_program_pseudo_examples
    assert composition.target_pattern_for_task is composition_pseudolabels.target_pattern_for_task
    assert (
        composition_pseudolabels.compose_program_pseudo_examples
        is composition_program_pseudolabels.compose_program_pseudo_examples
    )
    assert composition_pseudolabels.target_pattern_for_task is composition_program_pseudolabels.target_pattern_for_task


def test_driver_compat_uses_pseudolabel_module() -> None:
    assert driver_compat_exports.compose_pseudo_examples is composition_pseudolabels.compose_pseudo_examples
    assert driver_compat_exports.target_pattern_for_task is composition_pseudolabels.target_pattern_for_task

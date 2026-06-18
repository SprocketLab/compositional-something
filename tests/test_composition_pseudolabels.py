from self.core import composition


def test_composition_exports_pseudolabel_helpers() -> None:
    assert callable(composition.compose_pseudo_examples)
    assert callable(composition.compose_addition_pseudo_examples)
    assert callable(composition.compose_run_length_pseudo_examples)
    assert callable(composition.compose_program_pseudo_examples)
    assert callable(composition.target_pattern_for_task)

from __future__ import annotations

from self import multiplication_rectangular as legacy_rectangular
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

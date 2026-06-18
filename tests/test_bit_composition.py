from __future__ import annotations

import random

from self.tasks import bit_common, bit_composition


def test_fixed_binary_component_sizes_and_old_reexport_match():
    assert bit_composition.choose_component_sizes(
        9,
        [4, 5, 6],
        random.Random(0),
        bit_composition_path_mode=bit_composition.BIT_COMPOSITION_PATH_FIXED_BINARY,
    ) == [4, 5]

    assert bit_common.choose_component_sizes is bit_composition.choose_component_sizes
    assert bit_common.BIT_COMPOSITION_PATH_FIXED_BINARY == bit_composition.BIT_COMPOSITION_PATH_FIXED_BINARY


def test_reachable_sizes_respect_composition_path_mode():
    examples = [type("Example", (), {"bits": bits})() for bits in (4, 5, 6)]

    assert bit_composition.bit_composed_target_sizes_from_examples(
        examples,
        size_getter=lambda example: example.bits,
        min_size=8,
        max_size=11,
        compose_arity="exact2",
        bit_composition_path_mode=bit_composition.BIT_COMPOSITION_PATH_RANDOM,
    ) == [8, 9, 10, 11]
    assert bit_composition.bit_composed_target_sizes_from_examples(
        examples,
        size_getter=lambda example: example.bits,
        min_size=8,
        max_size=11,
        compose_arity="at_least2",
        bit_composition_path_mode=bit_composition.BIT_COMPOSITION_PATH_FIXED_BINARY,
    ) == [8, 9, 10, 11]

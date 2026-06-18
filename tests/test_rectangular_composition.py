from __future__ import annotations

from self import multiplication_rectangular as legacy_rectangular
from self.tasks import rectangular_composition
from self.tasks import rectangular_multiplication
from self.tasks.rectangular_data import RectangularMultiplicationExample
from self.tasks.rectangular_partitions import iter_partition_grid


def test_rectangular_lsd_block_helpers_split_from_least_significant_side():
    assert rectangular_composition._lsd_block_digit_sizes(8, 3) == [3, 3, 2]
    assert rectangular_composition._split_value_into_lsd_blocks(
        12345678,
        total_digits=8,
        max_chunk_digits=3,
    ) == [(0, 678, 3), (3, 345, 3), (6, 12, 2)]


def test_rectangular_multiplier_digit_components_and_trace_composition():
    example = RectangularMultiplicationExample(
        a=12,
        b=345,
        a_digits=2,
        b_digits=3,
        format_version="cot_reverse_v1",
    )

    components = rectangular_composition.build_multiplier_digit_components(example)
    component_values = [component.a * component.b for _, component in components]

    assert [digit_index for digit_index, _ in components] == [0, 1, 2]
    assert rectangular_composition.compose_target_from_multiplier_digit_values(
        example,
        component_values,
    ) == example.target()


def test_rectangular_partition_components_and_weighted_composition():
    example = RectangularMultiplicationExample(
        a=12345678,
        b=87654321,
        a_digits=8,
        b_digits=8,
        format_version="symbolic_v1",
    )
    leaves = rectangular_composition.build_partition_supported_components(
        example,
        supported_partitions=iter_partition_grid(1, 3, 1, 3),
    )
    weighted_values = [
        (leaf.shift_digits, leaf.example.a * leaf.example.b)
        for leaf in leaves
    ]

    assert len(leaves) == 9
    assert rectangular_composition.compose_target_from_weighted_component_values(
        example,
        weighted_values,
    ) == example.target()


def test_rectangular_composition_helpers_remain_available_through_old_modules():
    assert (
        legacy_rectangular.RectangularCompositionLeaf
        is rectangular_composition.RectangularCompositionLeaf
    )
    assert (
        legacy_rectangular.build_partition_supported_components
        is rectangular_composition.build_partition_supported_components
    )
    assert (
        legacy_rectangular.compose_target_from_weighted_component_values
        is rectangular_composition.compose_target_from_weighted_component_values
    )

    assert (
        rectangular_multiplication.build_multiplier_digit_components
        is rectangular_composition.build_multiplier_digit_components
    )
    assert (
        rectangular_multiplication._lsd_block_digit_sizes
        is rectangular_composition._lsd_block_digit_sizes
    )

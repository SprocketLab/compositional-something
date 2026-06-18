from __future__ import annotations

from self.tasks import rectangular
from self.tasks.rectangular import RectangularMultiplicationExample
from self.tasks.rectangular import iter_partition_grid


def test_rectangular_lsd_block_helpers_split_from_least_significant_side():
    assert rectangular._lsd_block_digit_sizes(8, 3) == [3, 3, 2]
    assert rectangular._split_value_into_lsd_blocks(
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

    components = rectangular.build_multiplier_digit_components(example)
    component_values = [component.a * component.b for _, component in components]

    assert [digit_index for digit_index, _ in components] == [0, 1, 2]
    assert rectangular.compose_target_from_multiplier_digit_values(
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
    leaves = rectangular.build_partition_supported_components(
        example,
        supported_partitions=iter_partition_grid(1, 3, 1, 3),
    )
    weighted_values = [
        (leaf.shift_digits, leaf.example.a * leaf.example.b)
        for leaf in leaves
    ]

    assert len(leaves) == 9
    assert rectangular.compose_target_from_weighted_component_values(
        example,
        weighted_values,
    ) == example.target()


def test_rectangular_composition_helpers_live_in_merged_module():
    assert rectangular.RectangularCompositionLeaf.__module__ == "self.tasks.rectangular"
    assert rectangular.build_partition_supported_components.__module__ == "self.tasks.rectangular"
    assert rectangular.compose_target_from_weighted_component_values.__module__ == "self.tasks.rectangular"

from __future__ import annotations

from self import multiplication_rectangular as legacy_rectangular
from self.tasks import rectangular_partitions
from self.tasks.rectangular_multiplication import RectangularMultiplicationExample


def test_rectangular_partition_helpers_match_legacy_reexports():
    assert rectangular_partitions.parse_partition_spec("1x1,1x2,1x2,2x1") == [(1, 1), (1, 2), (2, 1)]
    assert rectangular_partitions.partition_label((2, 3)) == "2x3"
    assert rectangular_partitions.iter_partition_grid(1, 2, 3, 4) == [(1, 3), (1, 4), (2, 3), (2, 4)]

    assert legacy_rectangular.parse_partition_spec is rectangular_partitions.parse_partition_spec
    assert legacy_rectangular.partition_label is rectangular_partitions.partition_label
    assert legacy_rectangular.iter_partition_grid is rectangular_partitions.iter_partition_grid


def test_rectangular_partition_bucket_id_uses_digit_axes():
    example = RectangularMultiplicationExample(a=12, b=345, a_digits=2, b_digits=3)

    assert rectangular_partitions.partition_bucket_id(example) == 2003
    assert legacy_rectangular.partition_bucket_id(example) == 2003

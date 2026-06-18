from __future__ import annotations

from self.tasks import rectangular
from self.tasks.rectangular import RectangularMultiplicationExample


def test_rectangular_partition_helpers_match_legacy_reexports():
    assert rectangular.parse_partition_spec("1x1,1x2,1x2,2x1") == [(1, 1), (1, 2), (2, 1)]
    assert rectangular.partition_label((2, 3)) == "2x3"
    assert rectangular.iter_partition_grid(1, 2, 3, 4) == [(1, 3), (1, 4), (2, 3), (2, 4)]


def test_rectangular_partition_bucket_id_uses_digit_axes():
    example = RectangularMultiplicationExample(a=12, b=345, a_digits=2, b_digits=3)

    assert rectangular.partition_bucket_id(example) == 2003

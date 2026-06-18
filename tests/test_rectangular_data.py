from __future__ import annotations

import random

from self import multiplication_rectangular as legacy_rectangular
from self.tasks import rectangular_data
from self.tasks import rectangular_multiplication


def test_rectangular_example_prompt_target_and_key():
    example = rectangular_data.RectangularMultiplicationExample(
        a=12,
        b=345,
        a_digits=2,
        b_digits=3,
        format_version="symbolic_v1",
    )

    assert example.prompt() == "12×345="
    assert example.target() == "04140"
    assert example.target_prefix() == ""
    assert example.total_digits == 5
    assert rectangular_data.rectangular_multiplication_key(example) == (2, 3, 12, 345)


def test_rectangular_prediction_helpers_parse_by_format():
    cot_example = rectangular_data.RectangularMultiplicationExample(
        a=12,
        b=34,
        a_digits=2,
        b_digits=2,
        format_version="cot_reverse_v1",
    )
    symbolic_example = rectangular_data.RectangularMultiplicationExample(
        a=12,
        b=34,
        a_digits=2,
        b_digits=2,
        format_version="symbolic_v1",
    )

    parsed = rectangular_data.parse_rectangular_multiplication_final_value(
        "840+0630=804",
        cot_example,
    )
    assert parsed == 408
    normalized = rectangular_data.normalize_rectangular_prediction_for_training("408", symbolic_example)
    assert normalized == "0408"
    assert rectangular_data.prediction_matches_example("408", symbolic_example) is True


def test_rectangular_data_sampler_can_be_injected_without_old_module():
    calls = {"value": 0}

    def always_one_digit(*args, **kwargs):
        calls["value"] += 1
        return 1

    generated = rectangular_data.build_sampled_rectangular_dataset(
        partitions=[(1, 1)],
        per_partition_counts={"train": 2, "validation": 0, "test": 0},
        rng=random.Random(0),
        format_version="legacy",
        max_attempts=1,
        sample_int_fn=always_one_digit,
    )

    assert [example.a * example.b for example in generated["train"]] == [1, 1]
    assert calls["value"] == 4


def test_rectangular_data_helpers_remain_available_through_old_modules():
    assert legacy_rectangular.RectangularMultiplicationExample is rectangular_data.RectangularMultiplicationExample
    assert legacy_rectangular.rectangular_multiplication_key is rectangular_data.rectangular_multiplication_key
    assert legacy_rectangular.prediction_matches_example is rectangular_data.prediction_matches_example

    assert rectangular_multiplication.extract_numeric_answer is rectangular_data.extract_numeric_answer
    assert rectangular_multiplication.values_for_digits is rectangular_data.values_for_digits

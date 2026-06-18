from __future__ import annotations

from self import multiplication_rectangular as legacy_rectangular
from self.tasks import rectangular_digits
from self.tasks import rectangular_multiplication


def test_rectangular_digit_helpers_match_expected_reverse_order():
    assert rectangular_digits.split_digits_lsd_first(120) == [0, 2, 1]
    assert rectangular_digits.reverse_digit_text(120) == "021"


def test_rectangular_cot_reverse_helpers_match_expected_trace():
    assert rectangular_digits.format_cot_reverse_prompt(12, 345) == "21*543="
    assert rectangular_digits.format_cot_reverse_target(12, 345) == "060+0840(0450)+00630=04140"
    normalized = rectangular_digits.normalize_cot_reverse_prediction_for_training(
        " 060 +0840(0450)\n+00630=04140 "
    )
    assert normalized == "060+0840(0450)+00630=04140"
    assert rectangular_digits.extract_cot_reverse_final_digits("840+0630=804", 4) == "8040"
    assert rectangular_digits.parse_cot_reverse_final_value("840+0630=804", 4) == 408


def test_rectangular_digit_helpers_remain_available_through_old_modules():
    assert legacy_rectangular.format_cot_reverse_prompt is rectangular_digits.format_cot_reverse_prompt
    assert legacy_rectangular.format_cot_reverse_target is rectangular_digits.format_cot_reverse_target
    assert legacy_rectangular.parse_cot_reverse_final_value is rectangular_digits.parse_cot_reverse_final_value
    assert legacy_rectangular.split_digits_lsd_first is rectangular_digits.split_digits_lsd_first

    assert rectangular_multiplication.reverse_digit_text is rectangular_digits.reverse_digit_text
    assert (
        rectangular_multiplication.normalize_cot_reverse_prediction_for_training
        is rectangular_digits.normalize_cot_reverse_prediction_for_training
    )

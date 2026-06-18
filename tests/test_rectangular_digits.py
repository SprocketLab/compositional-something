from __future__ import annotations

from self.tasks import rectangular


def test_rectangular_digit_helpers_match_expected_reverse_order():
    assert rectangular.split_digits_lsd_first(120) == [0, 2, 1]
    assert rectangular.reverse_digit_text(120) == "021"


def test_rectangular_cot_reverse_helpers_match_expected_trace():
    assert rectangular.format_cot_reverse_prompt(12, 345) == "21*543="
    assert rectangular.format_cot_reverse_target(12, 345) == "060+0840(0450)+00630=04140"
    normalized = rectangular.normalize_cot_reverse_prediction_for_training(
        " 060 +0840(0450)\n+00630=04140 "
    )
    assert normalized == "060+0840(0450)+00630=04140"
    assert rectangular.extract_cot_reverse_final_digits("840+0630=804", 4) == "8040"
    assert rectangular.parse_cot_reverse_final_value("840+0630=804", 4) == 408


def test_rectangular_digit_helpers_live_in_merged_module():
    assert rectangular.reverse_digit_text.__module__ == "self.tasks.rectangular"
    assert rectangular.normalize_cot_reverse_prediction_for_training.__module__ == "self.tasks.rectangular"

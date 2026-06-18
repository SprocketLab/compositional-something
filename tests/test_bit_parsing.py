from __future__ import annotations

from types import SimpleNamespace

from self import self_improvement_tasks as tasks
from self.tasks import bit_common, bit_parsing


def test_bit_parsing_old_reexports_keep_identity() -> None:
    assert bit_common.parse_run_length_prediction is bit_parsing.parse_run_length_prediction
    assert bit_common.parse_run_length_run_state_prediction is bit_parsing.parse_run_length_run_state_prediction
    assert bit_common.parse_multiplication_prediction is bit_parsing.parse_multiplication_prediction
    assert tasks.parse_run_length_prediction is bit_parsing.parse_run_length_prediction
    assert tasks.format_multiplication_target is bit_parsing.format_multiplication_target
    assert tasks.RUN_LENGTH_TARGET_RUN_STATE == bit_parsing.RUN_LENGTH_TARGET_RUN_STATE


def test_bit_parsing_handles_task_specific_targets() -> None:
    symbolic_multiplication = SimpleNamespace(format_version="symbolic_v1", digits=2)
    symbol_pair_run_length = SimpleNamespace(bitstring="00111222", target_mode="symbol_run_pair")
    run_state = SimpleNamespace(
        bitstring="00111222",
        bits=8,
        target_mode=bit_parsing.RUN_LENGTH_TARGET_RUN_STATE,
    )

    assert bit_parsing.parse_multiplication_prediction("answer 141", symbolic_multiplication) == "0141"
    assert bit_parsing.parse_run_length_prediction("A: 9|3 then 1|3", symbol_pair_run_length) == "1|3"
    assert bit_parsing.parse_run_length_prediction("answer 3|0|2|2|3", run_state) == "3|0|2|2|3"
    assert bit_parsing.parse_run_length_prediction("answer 9|0|2|2|3", run_state) is None

from __future__ import annotations

import random

from self.diagnostics.run_length_balanced_eval import (
    construct_run_length_string,
    generate_balanced_examples,
    summarize_prediction_rows,
)
from self.tasks import compute_run_stats


def test_construct_run_length_string_hits_requested_max_run():
    rng = random.Random(0)
    alphabet = "01234"
    for bits in range(2, 12):
        for target in range(1, bits + 1):
            bitstring = construct_run_length_string(bits, target, alphabet, rng)
            max_run, _, _ = compute_run_stats(bitstring)
            assert len(bitstring) == bits
            assert max_run == target


def test_generate_balanced_examples_records_underfilled_extreme_cells():
    rng = random.Random(1)
    examples, counts, underfilled = generate_balanced_examples(
        min_bits=4,
        max_bits=4,
        alphabet="012",
        per_answer=5,
        rng=rng,
        format_version="legacy",
        target_mode="plain_output",
        max_attempts_per_cell=200,
    )

    assert counts["4"]["4"] == 3
    assert any(cell["bits"] == 4 and cell["answer"] == 4 and cell["retained"] == 3 for cell in underfilled)
    for example in examples:
        assert example.target() == str(example.max_run)


def test_summarize_prediction_rows_reports_answer_cell_macro_accuracy():
    rows = [
        {"bits": 4, "answer": 1, "correct": True},
        {"bits": 4, "answer": 1, "correct": True},
        {"bits": 4, "answer": 2, "correct": False},
    ]

    summary = summarize_prediction_rows(rows, min_supported_count=2, frontier_min_bits=4)

    assert summary["micro_accuracy"] == 2 / 3
    assert summary["macro_answer_accuracy"] == 0.5
    assert summary["macro_supported_accuracy"] == 1.0
    assert summary["per_bit_macro_accuracy"]["4"] == 0.5
    assert summary["per_answer_accuracy"]["1"] == 1.0
    assert summary["per_answer_accuracy"]["2"] == 0.0
    assert summary["frontier"]["macro_answer_accuracy"] == 0.5

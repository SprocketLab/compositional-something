from __future__ import annotations

import math
from pathlib import Path

from self.core.nonadaptive_evaluation import evaluate_nonadaptive_round


class _Task:
    prediction_parser = staticmethod(lambda text, example=None: text)

    @staticmethod
    def size_of(example) -> int:
        return len(str(example))

    @staticmethod
    def key_for_example(example) -> str:
        return str(example)


def test_evaluate_nonadaptive_round_aggregates_composed_slices_and_debugs(tmp_path: Path):
    eval_calls = []
    debug_calls = []

    def evaluate_accuracy_fn(**kwargs):
        examples = list(kwargs["examples"])
        eval_calls.append((examples, kwargs["max_new_tokens"]))
        if examples == ["base"]:
            return 0.25, {4: 0.25}
        if examples == ["acc1", "acc2"]:
            return 0.5, {4: 0.5}
        if examples == ["rej"]:
            return 1.0, {3: 1.0}
        if examples == ["nan"]:
            return math.nan, {3: math.nan}
        raise AssertionError(f"unexpected examples: {examples!r}")

    def write_debug_samples_fn(path, **kwargs):
        debug_calls.append((path, list(kwargs["examples"]), kwargs["max_new_tokens"], kwargs["component_map"]))

    result = evaluate_nonadaptive_round(
        model="model",
        tokenizer="tokenizer",
        task=_Task(),
        eval_examples=["base"],
        composed_eval_slices={
            "accepted_by_guard": ["acc1", "acc2"],
            "rejected_by_guard": ["rej"],
            "empty": [],
            "other": ["nan"],
        },
        composed_eval_component_map={"acc1": ["a"], "rej": ["r"]},
        round_dir=tmp_path,
        batch_size=4,
        eval_decode_tokens=7,
        composed_eval_decode_tokens=9,
        evaluate_accuracy_fn=evaluate_accuracy_fn,
        write_debug_samples_fn=write_debug_samples_fn,
    )

    assert result.eval_accuracy == 0.25
    assert result.per_size_accuracy == {4: 0.25}
    assert result.composed_eval_accuracy == (0.5 * 2 + 1.0 * 1) / 3
    assert result.composed_slice_metrics["accepted_by_guard"].count == 2
    assert result.composed_slice_metrics["accepted_by_guard"].per_size_accuracy == {4: 0.5}
    assert result.composed_slice_metrics["rejected_by_guard"].accuracy == 1.0
    assert math.isnan(result.composed_slice_metrics["empty"].accuracy)
    assert result.composed_slice_metrics["empty"].count == 0
    assert math.isnan(result.composed_slice_metrics["other"].accuracy)
    assert eval_calls == [
        (["base"], 7),
        (["acc1", "acc2"], 9),
        (["rej"], 9),
        (["nan"], 9),
    ]
    assert debug_calls == [
        (
            tmp_path / "composed_eval_accepted_by_guard_debug.jsonl",
            ["acc1", "acc2"],
            9,
            {"acc1": ["a"], "rej": ["r"]},
        ),
        (
            tmp_path / "composed_eval_rejected_by_guard_debug.jsonl",
            ["rej"],
            9,
            {"acc1": ["a"], "rej": ["r"]},
        ),
    ]


def test_evaluate_nonadaptive_round_reports_nan_without_composed_counts(tmp_path: Path):
    result = evaluate_nonadaptive_round(
        model="model",
        tokenizer="tokenizer",
        task=_Task(),
        eval_examples=["base"],
        composed_eval_slices={"all": []},
        composed_eval_component_map={},
        round_dir=tmp_path,
        batch_size=4,
        eval_decode_tokens=7,
        composed_eval_decode_tokens=9,
        evaluate_accuracy_fn=lambda **kwargs: (1.0, {1: 1.0}),
        write_debug_samples_fn=lambda *args, **kwargs: None,
    )

    assert result.eval_accuracy == 1.0
    assert math.isnan(result.composed_eval_accuracy)
    assert math.isnan(result.composed_slice_metrics["all"].accuracy)

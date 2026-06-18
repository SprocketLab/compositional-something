from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import self.tasks as tasks
from self.tasks import bit


@dataclass(frozen=True)
class TinyExample:
    key: int
    size: int
    target: Optional[str] = None


def _clone(example: TinyExample, override: Optional[str]) -> TinyExample:
    return TinyExample(key=example.key, size=example.size, target=override)


def test_bit_pseudolabel_helpers_keep_compat_aliases() -> None:
    assert bit.build_direct_pseudo_examples is bit.build_direct_pseudo_examples
    assert bit.build_guarded_bit_pseudo_examples is bit.build_guarded_bit_pseudo_examples
    assert bit.guard_slice_partition is bit.guard_slice_partition
    assert bit.run_length_guard_accepts_true_components is bit.run_length_guard_accepts_true_components

    assert tasks.build_direct_pseudo_examples is bit.build_direct_pseudo_examples
    assert tasks.build_guarded_bit_pseudo_examples is bit.build_guarded_bit_pseudo_examples


def test_direct_pseudo_examples_use_facade_prediction_map(monkeypatch) -> None:
    examples = [TinyExample(1, 1), TinyExample(2, 1)]

    def fake_prediction_map(**kwargs):
        assert kwargs["examples"] == examples
        return {1: "kept"}

    monkeypatch.setattr(tasks, "generate_prediction_map", fake_prediction_map)

    pseudo_examples, missing_total, diagnostics = bit.build_direct_pseudo_examples(
        examples,
        model=None,
        tokenizer=None,
        batch_size=4,
        decode_max_new_tokens=8,
        key_getter=lambda example: example.key,
        prediction_parser=lambda text: text,
        clone_builder=_clone,
        mode="direct",
    )

    assert pseudo_examples == [TinyExample(1, 1, "kept")]
    assert missing_total == 1
    assert diagnostics["candidate_total"] == 2
    assert diagnostics["retained_total"] == 1
    assert diagnostics["missing_total"] == 1


def test_run_length_boundary_guard_and_partition() -> None:
    assert bit.run_length_guard_accepts_true_components([(2, "00"), (2, "11")]) is True
    assert bit.run_length_guard_accepts_true_components([(2, "01"), (2, "10")]) is False
    assert bit.run_length_guard_accepts_true_components([(2, "01")]) is None

    examples = [TinyExample(1, 1), TinyExample(2, 1), TinyExample(3, 1)]
    partitions = bit.guard_slice_partition(
        examples,
        {
            1: [(2, "00"), (2, "11")],
            2: [(2, "01"), (2, "10")],
        },
        key_getter=lambda example: example.key,
        guard_fn=bit.run_length_guard_accepts_true_components,
    )

    assert partitions["accepted_by_guard"] == [TinyExample(1, 1)]
    assert partitions["rejected_by_guard"] == [TinyExample(2, 1)]
    assert partitions["all"] == examples


def test_guarded_bit_pseudo_examples_tracks_rejections_without_refill() -> None:
    examples = [TinyExample(1, 1), TinyExample(2, 1)]
    component_map = {1: ["ok"], 2: ["bad"]}

    def evaluate_candidate(example: TinyExample, component_keys):
        del example
        if component_keys == ["ok"]:
            return "accepted", "label"
        return "rejected", None

    pseudo_examples, missing_total, diagnostics = bit.build_guarded_bit_pseudo_examples(
        examples,
        component_map,
        target_max_size=1,
        requested_per_size=1,
        size_getter=lambda example: example.size,
        key_getter=lambda example: example.key,
        clone_builder=_clone,
        evaluate_candidate=evaluate_candidate,
        refill_builder=lambda _size, _need, _occupied: ([], {}),
        mode="compose_guarded",
    )

    assert pseudo_examples == [TinyExample(1, 1, "label")]
    assert missing_total == 0
    assert diagnostics["candidate_total"] == 2
    assert diagnostics["retained_total"] == 1
    assert diagnostics["rejected_total"] == 1
    assert diagnostics["refill_rounds"] == 0

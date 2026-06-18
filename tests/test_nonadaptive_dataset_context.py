from __future__ import annotations

import pytest

from self.nonadaptive.nonadaptive_dataset_context import prepare_nonadaptive_dataset_context


class _Task:
    def split_composed_eval_slices(self, examples, component_map):
        return {
            "safe": [example for example in examples if component_map[example] == "safe"],
            "carry": [example for example in examples if component_map[example] == "carry"],
        }

    def keys_for_examples(self, examples):
        return {f"key:{example}" for example in examples}


def test_prepare_nonadaptive_dataset_context_reports_counts_and_returns_context():
    messages = []

    context = prepare_nonadaptive_dataset_context(
        task=_Task(),
        base_splits={"train": ["base"], "validation": [], "test": []},
        composed_examples=["comp-1", "comp-2"],
        eval_examples=["eval-1", "eval-2"],
        composed_eval_examples=["ce-1", "ce-2", "ce-3"],
        composed_eval_component_map={"ce-1": "safe", "ce-2": "carry", "ce-3": "safe"},
        print_fn=lambda message, **kwargs: messages.append((message, kwargs)),
    )

    assert context.composed_eval_slices == {"safe": ["ce-1", "ce-3"], "carry": ["ce-2"]}
    assert context.eval_keys == {"key:eval-1", "key:eval-2"}
    assert messages == [
        (
            "[INFO] Dataset sizes -- base train: 1 | composed pool: 2 | eval: 2 | composed eval: 3",
            {"flush": True},
        ),
        ("[INFO] Composed eval slices -- safe: 2 | carry: 1", {"flush": True}),
    ]


def test_prepare_nonadaptive_dataset_context_skips_slice_report_without_composed_eval():
    messages = []

    context = prepare_nonadaptive_dataset_context(
        task=_Task(),
        base_splits={"train": ["base"]},
        composed_examples=[],
        eval_examples=[],
        composed_eval_examples=[],
        composed_eval_component_map={},
        print_fn=lambda message, **kwargs: messages.append((message, kwargs)),
    )

    assert context.composed_eval_slices == {"safe": [], "carry": []}
    assert context.eval_keys == set()
    assert messages == [
        (
            "[INFO] Dataset sizes -- base train: 1 | composed pool: 0 | eval: 0 | composed eval: 0",
            {"flush": True},
        )
    ]


def test_prepare_nonadaptive_dataset_context_rejects_empty_train_split():
    with pytest.raises(ValueError, match="Base training split is empty"):
        prepare_nonadaptive_dataset_context(
            task=_Task(),
            base_splits={"train": []},
            composed_examples=[],
            eval_examples=[],
            composed_eval_examples=[],
            composed_eval_component_map={},
        )

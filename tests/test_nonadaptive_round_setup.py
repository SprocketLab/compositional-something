from __future__ import annotations

from pathlib import Path

from self.nonadaptive.nonadaptive_round_setup import (
    prepare_nonadaptive_round_plan,
    prepare_nonadaptive_round_training_data,
)


class _SizeSchedule:
    def round_max_size_for_index(self, round_idx: int) -> int:
        return 10 + round_idx


class _Task:
    @staticmethod
    def serialize_example(example: dict) -> dict:
        return {"serialized": example["id"]}


def test_prepare_nonadaptive_round_plan_creates_directory_and_marks_resume_skip(tmp_path: Path):
    ensured = []

    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        ensured.append(path)
        return path

    plan = prepare_nonadaptive_round_plan(
        base_output_dir=tmp_path,
        round_idx=2,
        size_schedule=_SizeSchedule(),
        save_model_policy="final_only",
        num_expand_rounds=4,
        resume_requested=True,
        resume_round=3,
        ensure_dir_fn=ensure_dir,
    )

    assert plan.round_idx == 2
    assert plan.max_size == 12
    assert plan.round_dir == tmp_path / "round_02"
    assert plan.round_dir.exists()
    assert ensured == [tmp_path / "round_02"]
    assert plan.save_model_this_round is False
    assert plan.should_skip_completed_round is True


def test_prepare_nonadaptive_round_plan_marks_final_round_save(tmp_path: Path):
    plan = prepare_nonadaptive_round_plan(
        base_output_dir=tmp_path,
        round_idx=4,
        size_schedule=_SizeSchedule(),
        save_model_policy="final_only",
        num_expand_rounds=4,
        resume_requested=False,
        resume_round=0,
        ensure_dir_fn=lambda path: path.mkdir(parents=True, exist_ok=True),
    )

    assert plan.save_model_this_round is True
    assert plan.should_skip_completed_round is False


def test_prepare_nonadaptive_round_training_data_persists_base_then_pseudo(tmp_path: Path):
    saved = []

    def save_examples(path, examples, serialize_example):
        saved.append((path.name, [serialize_example(example) for example in examples]))

    result = prepare_nonadaptive_round_training_data(
        round_dir=tmp_path,
        base_train_examples=[{"id": "base-1"}, {"id": "base-2"}],
        pseudo_examples=[{"id": "pseudo-1"}],
        task=_Task(),
        save_examples_fn=save_examples,
    )

    assert result.train_examples == [{"id": "base-1"}, {"id": "base-2"}, {"id": "pseudo-1"}]
    assert result.pseudo_used_count == 1
    assert saved == [
        ("train_examples.jsonl", [{"serialized": "base-1"}, {"serialized": "base-2"}, {"serialized": "pseudo-1"}]),
        ("pseudo_examples_used.jsonl", [{"serialized": "pseudo-1"}]),
    ]

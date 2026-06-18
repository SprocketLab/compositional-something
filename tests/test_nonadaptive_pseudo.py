from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from self.core.nonadaptive_pseudo import prepare_nonadaptive_next_pseudo_round
from self.core.nonadaptive_schedule import NonAdaptiveSizeSchedule


@dataclass(frozen=True)
class _Example:
    name: str
    size: int
    value: str = "x"

    def prompt(self) -> str:
        return f"{self.name}?"

    def target(self) -> str:
        return self.value


class _Task:
    def __init__(self) -> None:
        self.component_saves: list[tuple[Path, object]] = []
        self.prepare_calls: list[dict[str, object]] = []
        self.derive_calls: list[dict[str, object]] = []

    @staticmethod
    def serialize_example(example: _Example) -> dict[str, object]:
        return {"name": example.name, "size": example.size, "value": example.value}

    @staticmethod
    def keys_for_examples(examples) -> set[str]:
        return {example.name for example in examples}

    def save_component_map(self, path, component_map) -> None:
        self.component_saves.append((Path(path), component_map))

    def prepare_composed_train(self, rng, args, base_splits, base_records, min_size, max_size, additional_exclude=None):
        self.prepare_calls.append(
            {
                "rng": rng,
                "args": args,
                "base_splits": base_splits,
                "base_records": base_records,
                "min_size": min_size,
                "max_size": max_size,
                "additional_exclude": set(additional_exclude or set()),
            }
        )
        return [_Example("fresh", 12, "fresh-target")], {"fresh": ["train"]}, {"fresh"}

    def derive_round_targets(
        self,
        model,
        tokenizer,
        composed_examples,
        component_map,
        *,
        target_max_size,
        base_examples,
        batch_size,
        decode_max_new_tokens,
        args,
        rng,
    ):
        self.derive_calls.append(
            {
                "model": model,
                "tokenizer": tokenizer,
                "composed_examples": list(composed_examples),
                "component_map": component_map,
                "target_max_size": target_max_size,
                "base_examples": list(base_examples),
                "batch_size": batch_size,
                "decode_max_new_tokens": decode_max_new_tokens,
                "args": args,
                "rng": rng,
            }
        )
        return [_Example("pseudo", 12, "pseudo-target")], 2, {"candidate_total": 3, "retained_total": 1}


def _args(**overrides):
    args = dict(
        num_expand_rounds=1,
        expand_train_per_size=2,
        pseudo_label_mode="direct",
        bit_composition_path_mode="guarded",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _schedule() -> NonAdaptiveSizeSchedule:
    return NonAdaptiveSizeSchedule(
        initial_max_size=8,
        expand_num_size=4,
        num_expand_rounds=1,
        frontier_min_size=None,
    )


def _save_recorder(records):
    def save_examples(path, examples, serializer):
        records.append((Path(path), [serializer(example) for example in examples]))

    return save_examples


def test_prepare_nonadaptive_next_pseudo_round_final_round_skips_side_effects(tmp_path):
    records = []
    persist_called = {"value": False}
    task = _Task()
    composed = [_Example("old", 9)]
    component_map = {"old": ["train"]}

    result = prepare_nonadaptive_next_pseudo_round(
        args=_args(num_expand_rounds=0),
        task=task,
        model="model",
        tokenizer="tokenizer",
        rng=SimpleNamespace(random=lambda: pytest.fail("rng should not be advanced")),
        round_idx=0,
        round_dir=tmp_path / "round_00",
        train_examples=[_Example("train", 8)],
        base_splits={"train": [_Example("train", 8)]},
        base_records={"train": {"train"}},
        composed_examples=composed,
        component_map=component_map,
        composed_pool_path=tmp_path / "data" / "composed_pool.jsonl",
        component_map_path=tmp_path / "data" / "component_map.json",
        metadata={},
        eval_keys={"eval"},
        size_schedule=_schedule(),
        composed_min_size=9,
        final_max_size=12,
        train_base_decode_tokens=4,
        config_decode_max_new_tokens=3,
        eval_batch_size=5,
        dynamic_composed=True,
        persist_metadata_fn=lambda: persist_called.__setitem__("value", True),
        save_examples_fn=_save_recorder(records),
        resolve_max_new_tokens_fn=lambda *args, **kwargs: pytest.fail("decode length should not be resolved"),
        random_cls=lambda seed: pytest.fail("pseudo rng should not be created"),
    )

    assert result.composed_examples is composed
    assert result.component_map is component_map
    assert result.pseudo_examples == []
    assert result.pseudo_generation_stats == {}
    assert result.missing_labels == 0
    assert persist_called["value"] is False
    assert records == []
    assert task.prepare_calls == []
    assert task.derive_calls == []
    assert task.component_saves == []


def test_prepare_nonadaptive_next_pseudo_round_refreshes_dynamic_pool_and_derives_targets(tmp_path, capsys):
    records = []
    persist_calls = []
    metadata = {}
    task = _Task()

    result = prepare_nonadaptive_next_pseudo_round(
        args=_args(),
        task=task,
        model="model",
        tokenizer="tokenizer",
        rng=SimpleNamespace(random=lambda: 0.25),
        round_idx=0,
        round_dir=tmp_path / "round_00",
        train_examples=[_Example("train", 8), _Example("pseudo-used", 9)],
        base_splits={"train": [_Example("train", 8)], "validation": [], "test": []},
        base_records={"train": {"train"}},
        composed_examples=[_Example("old", 9)],
        component_map={"old": ["train"]},
        composed_pool_path=tmp_path / "data" / "composed_pool.jsonl",
        component_map_path=tmp_path / "data" / "component_map.json",
        metadata=metadata,
        eval_keys={"eval"},
        size_schedule=_schedule(),
        composed_min_size=9,
        final_max_size=12,
        train_base_decode_tokens=4,
        config_decode_max_new_tokens=3,
        eval_batch_size=5,
        dynamic_composed=True,
        persist_metadata_fn=lambda: persist_calls.append(dict(metadata)),
        save_examples_fn=_save_recorder(records),
        resolve_max_new_tokens_fn=lambda examples, base_value: max(base_value, 11),
        random_cls=lambda seed: ("pseudo-rng", seed),
    )

    assert metadata["last_composed_refresh"] == "round_00_next"
    assert persist_calls == [{"last_composed_refresh": "round_00_next"}]
    assert task.prepare_calls[0]["additional_exclude"] == {"eval", "train", "pseudo-used"}
    assert task.prepare_calls[0]["max_size"] == 12
    assert task.derive_calls[0]["composed_examples"] == [_Example("fresh", 12, "fresh-target")]
    assert task.derive_calls[0]["component_map"] == {"fresh": ["train"]}
    assert task.derive_calls[0]["target_max_size"] == 12
    assert task.derive_calls[0]["base_examples"] == [_Example("train", 8), _Example("pseudo-used", 9)]
    assert task.derive_calls[0]["batch_size"] == 5
    assert task.derive_calls[0]["decode_max_new_tokens"] == 11
    assert task.derive_calls[0]["rng"] == ("pseudo-rng", 0.25)
    assert result.composed_examples == [_Example("fresh", 12, "fresh-target")]
    assert result.component_map == {"fresh": ["train"]}
    assert result.pseudo_examples == [_Example("pseudo", 12, "pseudo-target")]
    assert result.pseudo_generation_stats == {
        "candidate_total": 3,
        "retained_total": 1,
        "bit_composition_path_mode": "guarded",
    }
    assert result.missing_labels == 2
    assert records == [
        (tmp_path / "data" / "composed_pool.jsonl", [{"name": "fresh", "size": 12, "value": "fresh-target"}]),
        (
            tmp_path / "round_00" / "composed_pool_for_next_round.jsonl",
            [{"name": "fresh", "size": 12, "value": "fresh-target"}],
        ),
        (tmp_path / "round_00" / "pseudo_for_next_round.jsonl", [{"name": "pseudo", "size": 12, "value": "pseudo-target"}]),
    ]
    assert task.component_saves == [
        (tmp_path / "data" / "component_map.json", {"fresh": ["train"]}),
        (tmp_path / "round_00" / "composed_component_map_next_round.json", {"fresh": ["train"]}),
    ]
    assert "skipped 2 composed examples" in capsys.readouterr().out


def test_prepare_nonadaptive_next_pseudo_round_none_mode_saves_empty_stats(tmp_path, capsys):
    records = []
    metadata = {"last_composed_refresh": "static_initial"}
    task = _Task()
    composed = [_Example("old", 9, "target")]
    component_map = {"old": ["train"]}

    result = prepare_nonadaptive_next_pseudo_round(
        args=_args(pseudo_label_mode="none"),
        task=task,
        model="model",
        tokenizer="tokenizer",
        rng=SimpleNamespace(random=lambda: 0.5),
        round_idx=0,
        round_dir=tmp_path / "round_00",
        train_examples=[_Example("train", 8)],
        base_splits={"train": [_Example("train", 8)]},
        base_records={"train": {"train"}},
        composed_examples=composed,
        component_map=component_map,
        composed_pool_path=tmp_path / "data" / "composed_pool.jsonl",
        component_map_path=tmp_path / "data" / "component_map.json",
        metadata=metadata,
        eval_keys={"eval"},
        size_schedule=_schedule(),
        composed_min_size=9,
        final_max_size=12,
        train_base_decode_tokens=4,
        config_decode_max_new_tokens=3,
        eval_batch_size=5,
        dynamic_composed=False,
        persist_metadata_fn=lambda: None,
        save_examples_fn=_save_recorder(records),
        resolve_max_new_tokens_fn=lambda examples, base_value: max(base_value, 9),
        random_cls=lambda seed: ("unused-rng", seed),
    )

    assert result.composed_examples is composed
    assert result.component_map is component_map
    assert result.pseudo_examples == []
    assert result.pseudo_generation_stats == {
        "mode": "none",
        "target_max_size": 12,
        "candidate_total": 0,
        "retained_total": 0,
        "missing_total": 0,
        "bit_composition_path_mode": "guarded",
    }
    assert result.missing_labels == 0
    assert task.prepare_calls == []
    assert task.derive_calls == []
    assert task.component_saves == []
    assert records == [(tmp_path / "round_00" / "pseudo_for_next_round.jsonl", [])]
    assert "No pseudo-labeled examples generated" in capsys.readouterr().out

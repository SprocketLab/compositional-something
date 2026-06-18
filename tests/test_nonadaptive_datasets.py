from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from self.core.data_io import save_examples
from self.core.nonadaptive_datasets import prepare_nonadaptive_datasets
from self.core.nonadaptive_schedule import NonAdaptiveSizeSchedule
from self.core.nonadaptive_state import prepare_nonadaptive_run_state


@dataclass(frozen=True)
class _Example:
    name: str
    size: int

    def prompt(self) -> str:
        return f"{self.name}?"

    def target(self) -> str:
        return str(self.size)


class _DatasetTask:
    name = "dummy"
    size_label = "bits"
    size_alias_singular = "bit"
    size_alias_plural = "bits"

    def __init__(self) -> None:
        self.composed_train_exclude = None
        self.composed_eval_exclude = None

    def serialize_example(self, example: _Example) -> dict[str, object]:
        return {"name": example.name, "size": example.size}

    def deserialize_example(self, payload: dict[str, object]) -> _Example:
        return _Example(name=str(payload["name"]), size=int(payload["size"]))

    def prepare_eval_examples(self, rng, args, min_size, max_size, exclude):
        del rng, args, min_size, exclude
        return [_Example("reserved", max_size)]

    def keys_for_examples(self, examples):
        return {example.name for example in examples}

    def prepare_initial_splits(self, rng, args):
        del rng, args
        train = [_Example("train", 4)]
        validation = [_Example("validation", 4)]
        test = [_Example("test", 4)]
        records = {
            "train": {"train"},
            "validation": {"validation"},
            "test": {"test"},
        }
        return {"train": train, "validation": validation, "test": test}, records

    def prepare_composed_train(self, rng, args, base_splits, base_records, min_size, max_size, additional_exclude=None):
        del rng, args, base_splits, base_records, min_size, max_size
        self.composed_train_exclude = set(additional_exclude or set())
        return [_Example("comp", 9)], {"comp": ["train"]}, {"comp"}

    def prepare_composed_eval(self, rng, args, base_splits, base_records, min_size, max_size, additional_exclude=None):
        del rng, args, base_splits, base_records, min_size, max_size
        self.composed_eval_exclude = set(additional_exclude or set())
        return [_Example("comp_eval", 9)], {"comp_eval": ["comp"]}, {"comp_eval"}

    def save_component_map(self, path, component_map) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(component_map), encoding="utf-8")

    def load_component_map(self, path):
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def build_task_metadata(self, args, final_max_size):
        del args
        return {"final_max_size": final_max_size}

    def metadata_aliases(self, args, final_max_size):
        del args, final_max_size
        return {"alias": "ok"}

    def rebuild_records(self, base_splits):
        return {split: {example.name for example in examples} for split, examples in base_splits.items()}


def _args(tmp_path, **overrides):
    args = dict(
        output_dir=str(tmp_path),
        resume=False,
        resume_from_round=None,
        reserve_shared_eval_first=True,
        eval_per_size=1,
        initial_min_size=4,
        initial_max_size=8,
        expand_num_size=4,
        expand_train_per_size=2,
        composed_eval_per_size=1,
        composed_refresh_mode="dynamic",
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


def test_prepare_nonadaptive_datasets_generates_and_persists_new_run(tmp_path):
    args = _args(tmp_path)
    task = _DatasetTask()
    state = prepare_nonadaptive_run_state(args, reset_each_round=False, json_module=json)
    persisted = {}
    wrote_config = {"value": False}

    datasets = prepare_nonadaptive_datasets(
        args,
        task,
        rng=None,
        run_state=state,
        size_schedule=_schedule(),
        final_max_size=12,
        composed_min_size=9,
        frontier_min_size=None,
        reset_each_round=False,
        dynamic_composed=True,
        persist_metadata_fn=lambda metadata: persisted.update(metadata),
        write_config_args_fn=lambda: wrote_config.__setitem__("value", True),
    )

    assert args._initial_exclude_keys == {"reserved"}
    assert task.composed_train_exclude == {"reserved", "train"}
    assert task.composed_eval_exclude == {"reserved", "comp"}
    assert [example.name for example in datasets.eval_examples] == ["reserved"]
    assert datasets.metadata["last_composed_refresh"] == "initial_dynamic"
    assert datasets.metadata["alias"] == "ok"
    assert persisted["task"] == "dummy"
    assert wrote_config["value"] is True
    assert state.paths.base_train_path.exists()
    assert state.paths.composed_pool_path.exists()
    assert state.paths.composed_eval_path.exists()


def test_prepare_nonadaptive_datasets_loads_existing_run_and_validates(tmp_path):
    args = _args(tmp_path, resume=True, reserve_shared_eval_first=False)
    task = _DatasetTask()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    metadata = {
        "task": "dummy",
        "initial_min_size": 4,
        "initial_max_size": 8,
        "composed_max_size": 12,
        "composed_eval_per_size": 1,
        "composed_refresh_mode": "dynamic",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    save_examples(data_dir / "initial_train.jsonl", [_Example("train", 4)], task.serialize_example)
    save_examples(data_dir / "initial_validation.jsonl", [_Example("validation", 4)], task.serialize_example)
    save_examples(data_dir / "initial_test.jsonl", [_Example("test", 4)], task.serialize_example)
    save_examples(data_dir / "composed_pool.jsonl", [_Example("comp", 9)], task.serialize_example)
    save_examples(data_dir / "evaluation.jsonl", [_Example("eval", 12)], task.serialize_example)
    save_examples(data_dir / "composed_evaluation.jsonl", [_Example("comp_eval", 9)], task.serialize_example)
    task.save_component_map(data_dir / "composed_component_map.json", {"comp": ["train"]})
    task.save_component_map(data_dir / "composed_evaluation_component_map.json", {"comp_eval": ["comp"]})
    state = prepare_nonadaptive_run_state(args, reset_each_round=False, json_module=json)
    validation_called = {"value": False}

    datasets = prepare_nonadaptive_datasets(
        args,
        task,
        rng=None,
        run_state=state,
        size_schedule=_schedule(),
        final_max_size=12,
        composed_min_size=9,
        frontier_min_size=None,
        reset_each_round=False,
        dynamic_composed=True,
        persist_metadata_fn=lambda metadata: None,
        write_config_args_fn=lambda: None,
        validate_loaded_metadata_fn=lambda *args, **kwargs: validation_called.__setitem__("value", True),
    )

    assert validation_called["value"] is True
    assert [example.name for example in datasets.base_splits["train"]] == ["train"]
    assert datasets.base_records["train"] == {"train"}
    assert datasets.component_map == {"comp": ["train"]}
    assert [example.name for example in datasets.eval_examples] == ["eval"]
    assert datasets.metadata["task"] == "dummy"

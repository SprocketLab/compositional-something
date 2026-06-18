from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from self.core.data_io import encode_rng_state
from self.core.nonadaptive_state import (
    NonAdaptiveArtifactPaths,
    NonAdaptiveRunState,
    persist_nonadaptive_metadata,
    prepare_nonadaptive_run_state,
    validate_loaded_nonadaptive_metadata,
    write_nonadaptive_config_args,
)


class _MetadataTask:
    name = "dummy"
    size_alias_singular = "bit"
    size_alias_plural = "bits"

    def __init__(self) -> None:
        self.validated = False

    def validate_loaded_metadata(self, args, metadata, final_max_size, dynamic_composed) -> None:
        del args, metadata, final_max_size, dynamic_composed
        self.validated = True


def _metadata_args(**overrides):
    args = dict(
        initial_min_size=4,
        initial_max_size=8,
        composed_eval_per_size=2,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _run_state(tmp_path, metadata):
    data_dir = tmp_path / "data"
    paths = NonAdaptiveArtifactPaths(
        original_output_dir=tmp_path,
        base_output_dir=tmp_path,
        data_dir=data_dir,
        metadata_path=data_dir / "metadata.json",
        results_path=tmp_path / "self_improvement_results.json",
        base_train_path=data_dir / "initial_train.jsonl",
        base_val_path=data_dir / "initial_validation.jsonl",
        base_test_path=data_dir / "initial_test.jsonl",
        composed_pool_path=data_dir / "composed_pool.jsonl",
        component_map_path=data_dir / "composed_component_map.json",
        eval_path=data_dir / "evaluation.jsonl",
        composed_eval_path=data_dir / "composed_evaluation.jsonl",
        composed_eval_component_map_path=data_dir / "composed_evaluation_component_map.json",
    )
    return NonAdaptiveRunState(
        paths=paths,
        metadata=dict(metadata),
        existing_summaries={},
        resume_requested=True,
    )


def test_prepare_nonadaptive_run_state_creates_reset_paths(tmp_path, capsys):
    args = SimpleNamespace(output_dir=str(tmp_path / "run"), resume=False, resume_from_round=None)

    state = prepare_nonadaptive_run_state(
        args,
        reset_each_round=True,
        json_module=json,
    )

    assert state.paths.original_output_dir == tmp_path / "run"
    assert state.paths.base_output_dir == tmp_path / "run" / "reset_each_round"
    assert state.paths.data_dir == tmp_path / "run" / "reset_each_round" / "data"
    assert state.paths.original_output_dir.is_dir()
    assert state.paths.base_output_dir.is_dir()
    assert state.paths.data_dir.is_dir()
    assert state.resume_requested is False
    assert state.existing_summaries == {}
    assert state.new_run is True
    assert "reset_in_each_round enabled" in capsys.readouterr().out


def test_prepare_nonadaptive_run_state_loads_resume_metadata_and_summaries(tmp_path):
    output_dir = tmp_path / "resume"
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "metadata.json").write_text('{"task": "dummy", "legacy_size": 4}', encoding="utf-8")
    (data_dir / "initial_train.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    (output_dir / "self_improvement_results.json").write_text(
        '[{"round": 0, "eval_accuracy": 0.25}]',
        encoding="utf-8",
    )
    args = SimpleNamespace(output_dir=str(output_dir), resume=True, resume_from_round=None)

    state = prepare_nonadaptive_run_state(
        args,
        reset_each_round=False,
        json_module=json,
    )

    assert state.resume_requested is True
    assert state.metadata["task"] == "dummy"
    assert state.existing_summaries[0]["eval_accuracy"] == 0.25
    assert state.stored_value("missing", "legacy_size") == 4
    assert state.new_run is False


def test_persist_nonadaptive_metadata_writes_rng_state(tmp_path):
    metadata = {"task": "dummy"}
    rng = random.Random(7)
    metadata_path = tmp_path / "metadata.json"

    persist_nonadaptive_metadata(
        metadata,
        metadata_path,
        rng.getstate(),
        json_module=json,
        encode_rng_state_fn=encode_rng_state,
    )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["task"] == "dummy"
    assert payload["rng_state"]["version"] == 3


def test_write_nonadaptive_config_args_sanitizes_namespace(tmp_path):
    args = SimpleNamespace(output_dir=str(tmp_path), sizes={3, 1, 2})

    write_nonadaptive_config_args(args, tmp_path, json_module=json)

    payload = json.loads((tmp_path / "config_args.json").read_text(encoding="utf-8"))
    assert payload["output_dir"] == str(tmp_path)
    assert payload["sizes"] == [1, 2, 3]


def test_validate_loaded_nonadaptive_metadata_accepts_matching_metadata(tmp_path):
    task = _MetadataTask()
    state = _run_state(
        tmp_path,
        {
            "task": "dummy",
            "initial_min_size": 4,
            "initial_max_size": 8,
            "frontier_min_size": 10,
            "composed_max_size": 12,
            "reset_each_round": False,
            "composed_refresh_mode": "dynamic",
            "composed_eval_per_size": 2,
        },
    )

    validate_loaded_nonadaptive_metadata(
        _metadata_args(),
        task,
        state,
        final_max_size=12,
        frontier_min_size=10,
        reset_each_round=False,
        dynamic_composed=True,
    )

    assert task.validated is True


def test_validate_loaded_nonadaptive_metadata_rejects_task_mismatch(tmp_path):
    state = _run_state(
        tmp_path,
        {
            "task": "other",
            "initial_min_size": 4,
            "initial_max_size": 8,
            "composed_max_size": 12,
        },
    )

    with pytest.raises(ValueError, match="contains task"):
        validate_loaded_nonadaptive_metadata(
            _metadata_args(),
            _MetadataTask(),
            state,
            final_max_size=12,
            frontier_min_size=None,
            reset_each_round=False,
            dynamic_composed=True,
        )


def test_validate_loaded_nonadaptive_metadata_rejects_frontier_mismatch(tmp_path):
    state = _run_state(
        tmp_path,
        {
            "task": "dummy",
            "initial_min_size": 4,
            "initial_max_size": 8,
            "frontier_min_size": 11,
            "composed_max_size": 12,
            "composed_refresh_mode": "dynamic",
        },
    )

    with pytest.raises(ValueError, match="frontier_min_size mismatch"):
        validate_loaded_nonadaptive_metadata(
            _metadata_args(composed_eval_per_size=0),
            _MetadataTask(),
            state,
            final_max_size=12,
            frontier_min_size=10,
            reset_each_round=False,
            dynamic_composed=True,
        )

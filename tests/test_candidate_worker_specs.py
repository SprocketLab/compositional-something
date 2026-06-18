from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from self.adaptive.candidates import candidate_worker_specs, candidate_workers
from self.adaptive.proposals import PromptBundle


class _Task:
    @staticmethod
    def serialize_example(example):
        return {"value": example}


class _Trace:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_json_dict(self):
        return {"trace": self.value}


class _Proposal:
    def __init__(self, value: str) -> None:
        self.value = value

    def to_json_dict(self):
        return {"proposal": self.value}


def _item(index: int):
    return SimpleNamespace(
        index=index,
        row_id=f"row-{index}",
        proposal=_Proposal(f"proposal-{index}"),
        completion=f"completion-{index}",
        raw_output={"raw": index},
        proposal_prediction={"target": index},
        pseudo_diagnostics={"retained_total": index + 1},
    )


def _args(tmp_path: Path):
    return SimpleNamespace(
        seed=17,
        output_dir=tmp_path / "run",
        run_candidate_worker=True,
        candidate_worker_spec=tmp_path / "old-spec.json",
        run_candidate_pack_worker=True,
        candidate_worker_pack_spec=tmp_path / "old-pack.json",
    )


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_prepare_candidate_worker_specs_writes_inputs_specs_and_manifest(tmp_path: Path):
    round_dir = tmp_path / "attempt_0001"
    work_items = [_item(0), _item(2)]
    prompt = PromptBundle(system="system", user="user")

    spec_paths = candidate_worker_specs.prepare_candidate_worker_specs(
        args=_args(tmp_path),
        task=_Task(),
        current_checkpoint="checkpoint",
        source_examples=["source-a"],
        proposal_trace_buffer=[_Trace("proposal")],
        outcome_trace_buffer=[_Trace("outcome")],
        proposal_prompt=prompt,
        round_index=3,
        work_items=work_items,
        round_dir=round_dir,
        eval_examples=["eval-a"],
        current_final_accuracy=0.4,
        current_per_size_accuracy={5: 0.25},
        init_final_accuracy=0.1,
        attempt_index=4,
    )

    assert spec_paths == [
        round_dir / "candidate_jobs" / "specs" / "candidate_0.json",
        round_dir / "candidate_jobs" / "specs" / "candidate_1.json",
    ]
    assert (round_dir / "candidate_jobs" / "inputs" / "source_examples.jsonl").read_text(encoding="utf-8").strip() == '{"value": "source-a"}'
    assert (round_dir / "candidate_jobs" / "inputs" / "eval_examples.jsonl").read_text(encoding="utf-8").strip() == '{"value": "eval-a"}'
    assert _load_json(round_dir / "candidate_jobs" / "inputs" / "proposal_prompt.json") == {
        "system": "system",
        "user": "user",
    }
    assert (round_dir / "candidate_jobs" / "inputs" / "proposal_trace_buffer.jsonl").read_text(encoding="utf-8").strip() == '{"trace": "proposal"}'
    assert (round_dir / "candidate_jobs" / "inputs" / "outcome_trace_buffer.jsonl").read_text(encoding="utf-8").strip() == '{"trace": "outcome"}'

    first_spec = _load_json(spec_paths[0])
    assert first_spec["args"]["run_candidate_worker"] is False
    assert first_spec["args"]["candidate_worker_spec"] is None
    assert first_spec["args"]["run_candidate_pack_worker"] is False
    assert first_spec["args"]["candidate_worker_pack_spec"] is None
    assert first_spec["array_index"] == 0
    assert first_spec["candidate_index"] == 0
    assert first_spec["round_index"] == 3
    assert first_spec["attempt_index"] == 4
    assert first_spec["current_checkpoint"] == "checkpoint"
    assert first_spec["current_per_size_accuracy"] == {"5": 0.25}
    assert first_spec["seed"] == 17 + 4 * 1009
    assert first_spec["candidate"]["row_id"] == "row-0"
    assert first_spec["candidate"]["proposal"] == {"proposal": "proposal-0"}
    assert first_spec["candidate"]["completion"] == "completion-0"
    assert first_spec["candidate"]["raw_output"] == {"raw": 0}
    assert first_spec["candidate"]["proposal_prediction"] == {"target": 0}
    assert first_spec["candidate"]["pseudo_diagnostics"] == {"retained_total": 1}

    second_spec = _load_json(spec_paths[1])
    assert second_spec["array_index"] == 1
    assert second_spec["candidate_index"] == 2
    assert second_spec["seed"] == 17 + 4 * 1009 + 2

    manifest = _load_json(round_dir / "candidate_jobs" / "manifest.json")
    assert manifest == [
        {
            "array_index": 0,
            "candidate_index": 0,
            "metrics_path": str(round_dir / "candidates" / "candidate_00" / "candidate_metrics.json"),
            "spec_path": str(spec_paths[0]),
            "worker_failure_path": str(round_dir / "candidates" / "candidate_00" / "worker_failure.json"),
        },
        {
            "array_index": 1,
            "candidate_index": 2,
            "metrics_path": str(round_dir / "candidates" / "candidate_02" / "candidate_metrics.json"),
            "spec_path": str(spec_paths[1]),
            "worker_failure_path": str(round_dir / "candidates" / "candidate_02" / "worker_failure.json"),
        },
    ]


def test_prepare_candidate_worker_pack_specs_chunks_specs_and_writes_manifest(tmp_path: Path):
    round_dir = tmp_path / "attempt_0001"
    work_items = [_item(index) for index in range(5)]
    spec_paths = [tmp_path / f"spec-{index}.json" for index in range(5)]

    packs = candidate_worker_specs.prepare_candidate_worker_pack_specs(
        round_dir=round_dir,
        work_items=work_items,
        spec_paths=spec_paths,
        pack_size=2,
    )

    assert [(pack_index, [item.index for item in items], path.name) for pack_index, items, path in packs] == [
        (0, [0, 1], "pack_0.json"),
        (1, [2, 3], "pack_1.json"),
        (2, [4], "pack_2.json"),
    ]
    assert _load_json(round_dir / "candidate_jobs" / "pack_specs" / "pack_0.json") == {
        "candidates": [
            {
                "candidate_index": 0,
                "metrics_path": str(round_dir / "candidates" / "candidate_00" / "candidate_metrics.json"),
                "spec_path": str(spec_paths[0]),
                "worker_failure_path": str(round_dir / "candidates" / "candidate_00" / "worker_failure.json"),
            },
            {
                "candidate_index": 1,
                "metrics_path": str(round_dir / "candidates" / "candidate_01" / "candidate_metrics.json"),
                "spec_path": str(spec_paths[1]),
                "worker_failure_path": str(round_dir / "candidates" / "candidate_01" / "worker_failure.json"),
            },
        ],
        "pack_index": 0,
        "spec_paths": [str(spec_paths[0]), str(spec_paths[1])],
    }
    assert _load_json(round_dir / "candidate_jobs" / "pack_manifest.json") == [
        {
            "candidate_indices": [0, 1],
            "pack_index": 0,
            "pack_spec_path": str(round_dir / "candidate_jobs" / "pack_specs" / "pack_0.json"),
            "spec_paths": [str(spec_paths[0]), str(spec_paths[1])],
        },
        {
            "candidate_indices": [2, 3],
            "pack_index": 1,
            "pack_spec_path": str(round_dir / "candidate_jobs" / "pack_specs" / "pack_1.json"),
            "spec_paths": [str(spec_paths[2]), str(spec_paths[3])],
        },
        {
            "candidate_indices": [4],
            "pack_index": 2,
            "pack_spec_path": str(round_dir / "candidate_jobs" / "pack_specs" / "pack_2.json"),
            "spec_paths": [str(spec_paths[4])],
        },
    ]


def test_prepare_candidate_worker_pack_specs_rejects_nonpositive_pack_size(tmp_path: Path):
    with pytest.raises(ValueError, match="pack_size must be positive"):
        candidate_worker_specs.prepare_candidate_worker_pack_specs(
            round_dir=tmp_path,
            work_items=[],
            spec_paths=[],
            pack_size=0,
        )


def test_candidate_workers_reexports_canonical_spec_helpers():
    assert candidate_workers.prepare_candidate_worker_specs is candidate_worker_specs.prepare_candidate_worker_specs
    assert candidate_workers.prepare_candidate_worker_pack_specs is candidate_worker_specs.prepare_candidate_worker_pack_specs
    assert candidate_workers.candidate_metric_path is candidate_worker_specs.candidate_metric_path
    assert candidate_workers.candidate_worker_failure_path is candidate_worker_specs.candidate_worker_failure_path

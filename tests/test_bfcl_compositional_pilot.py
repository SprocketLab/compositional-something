from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from self.coding.atomic_data import AtomicExample, canonical_json, read_examples
from self.experiments.bfcl_compositional_pilot import (
    PRIMARY_CONDITION,
    TRAINED_CONDITIONS,
    _load_public_candidates,
    _materialize_training,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers/self/submit_bfcl_compositional_pilot_ailab.sh"


def atomic(source_id: str) -> AtomicExample:
    function = {
        "name": f"f_{source_id}",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    target = canonical_json([{"name": function["name"], "arguments": {"x": 1}}])
    return AtomicExample(
        task="bfcl",
        source_id=source_id,
        source_group_id=source_id,
        split="train",
        messages=({"role": "user", "content": source_id},),
        target=target,
        evaluator={"functions": [function], "accepted_calls": []},
        source_component_ids=(source_id,),
        metadata={"question": source_id},
    )


def accepted_record(index: int) -> dict:
    function = {
        "name": f"composed_{index}",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    target = canonical_json([{"name": function["name"], "arguments": {"x": index}}])
    candidate = {
        "candidate_id": f"candidate-{index}",
        "source_group_id": f"candidate-{index}",
        "split": "hidden_composition",
        "messages": [{"role": "user", "content": f"candidate {index}"}],
        "functions": [function],
        "component_count": 1,
        "source_component_ids": [f"source-{index}"],
        "question": f"candidate {index}",
        "template_id": "also",
        "template_partition": "train",
        "composition_family": "cross_function",
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "composed_target": target,
        "decision": {"accepted": True},
        "audit": {"oracle_exact": True},
    }


def test_round1_materialization_persists_unique_and_replay_expanded_data(tmp_path: Path):
    records = [accepted_record(index) for index in range(6)]
    mix = _materialize_training(
        run_root=tmp_path,
        round_index=1,
        condition="compose_g4",
        selected_records=records,
        atomic_train=[atomic("a"), atomic("b")],
    )
    assert mix == {
        "new": 6,
        "previous_frontier_replay": 0,
        "atomic_replay": 4,
        "total": 10,
        "trainer_seed": 7,
    }
    condition = tmp_path / "round_01/conditions/compose_g4"
    assert len(read_examples(condition / "composed_unique/selected_new.jsonl")) == 6
    materialized = read_examples(condition / "training_materialized/train.jsonl")
    assert len(materialized) == 10
    assert {row.metadata["training_origin"] for row in materialized} == {
        "new_composed",
        "atomic_replay",
    }


def test_materialized_training_order_is_condition_invariant(tmp_path: Path):
    records = [accepted_record(index) for index in range(12)]
    atomic_train = [atomic("a"), atomic("b"), atomic("c")]
    previous_new = [atomic("previous-a"), atomic("previous-b")]
    for round_index in (1, 2):
        for condition in ("compose_g1", "compose_g4"):
            _materialize_training(
                run_root=tmp_path,
                round_index=round_index,
                condition=condition,
                selected_records=records,
                atomic_train=atomic_train,
                previous_new=previous_new if round_index == 2 else None,
            )

    def model_facing_sequence(round_index: int, condition: str) -> list[tuple]:
        examples = read_examples(
            tmp_path
            / f"round_{round_index:02d}/conditions"
            / condition
            / "training_materialized/train.jsonl"
        )
        return [
            (
                example.source_id,
                example.messages,
                example.target,
                example.metadata["training_origin"],
                example.metadata.get("replay_instance", -1),
            )
            for example in examples
        ]

    for round_index in (1, 2):
        assert model_facing_sequence(round_index, "compose_g1") == model_facing_sequence(
            round_index, "compose_g4"
        )


def test_condition_matrix_contains_main_auxiliary_and_oracle_arms():
    assert PRIMARY_CONDITION == "compose_g1"
    assert TRAINED_CONDITIONS == (
        "direct_g4",
        "compose_g1",
        "compose_g4",
        "compose_g4_repeat20",
        "oracle",
    )


def test_generation_side_public_loader_does_not_require_oracle_file(tmp_path: Path):
    public_path = tmp_path / "data/public_candidates/round_01_cross.jsonl"
    public_path.parent.mkdir(parents=True)
    public_path.write_text('{"candidate_id":"public-only"}\n', encoding="utf-8")
    assert _load_public_candidates(tmp_path, 1, "cross") == [
        {"candidate_id": "public-only"}
    ]
    assert not (tmp_path / "data/oracle/round_01_cross.jsonl").exists()


def test_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_launcher_dry_run_emits_short_h200_pipeline(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "manifest.json").write_text(
        json.dumps({"jobs": {}, "status": "prepared"}),
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "ROOT_DIR": str(ROOT),
        "RUN_ROOT": str(run_root),
        "PREPARE": "0",
        "DRY_RUN": "1",
        "PYTHON_BIN": "/home/cs1095/.conda/envs/torch-env/bin/python",
    }
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    assert "bfcl-csi-r1-generate" in output
    assert "bfcl-csi-r1-gate" not in output
    assert "bfcl-csi-r2-gen-compose_g1" in output
    assert "bfcl-csi-r2-materialize" in output
    assert "bfcl-csi-r2-compose_g1" in output
    assert "bfcl-csi-collect" in output
    assert "--time=01:00:00" in output
    assert "--gres=gpu:h200:1" in output
    assert "afterok:" in output
    command_lines = [line for line in output.splitlines() if line.startswith("[INFO] Command: sbatch")]
    assert len(command_lines) == 19
    round2_generation = [
        line for line in command_lines if "--job-name=bfcl-csi-r2-gen-" in line
    ]
    assert len(round2_generation) == 4
    assert all("--dependency=afterok:" in line for line in round2_generation)

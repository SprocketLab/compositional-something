from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from self.coding.atomic_data import AtomicExample, canonical_json, read_examples, write_examples
from self.experiments.bfcl_cumulative_size_sweep import (
    CONDITIONS,
    SIZES,
    _cell_grid,
    _family_quota,
    _materialize,
    _run_grid,
    _training_flags,
)
from self.experiments.bfcl_compositional_pilot import _evaluation_sets


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers/self/submit_bfcl_cumulative_size_sweep_ailab.sh"


def atomic(source_id: str) -> AtomicExample:
    function = {
        "name": f"f_{source_id}",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
    }
    return AtomicExample(
        task="bfcl",
        source_id=source_id,
        source_group_id=source_id,
        split="train",
        messages=({"role": "user", "content": source_id},),
        target=canonical_json([{"name": function["name"], "arguments": {"x": 1}}]),
        evaluator={"functions": [function], "accepted_calls": []},
        source_component_ids=(source_id,),
        metadata={"question": source_id},
    )


def accepted_record(calls: int, index: int, family: str = "cross_function") -> dict:
    target = canonical_json(
        [{"name": f"f_{calls}_{part}", "arguments": {"x": index}} for part in range(calls)]
    )
    candidate = {
        "candidate_id": f"calls-{calls}-{family}-{index}",
        "source_group_id": f"group-{calls}-{family}-{index}",
        "split": "hidden_composition",
        "messages": [{"role": "user", "content": f"calls {calls} item {index}"}],
        "functions": [],
        "component_count": calls,
        "source_component_ids": [f"source-{calls}-{index}-{part}" for part in range(calls)],
        "question": f"calls {calls} item {index}",
        "template_id": "also",
        "template_partition": "train",
        "composition_family": family,
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate": candidate,
        "composed_target": target,
        "decision": {"accepted": True},
        "audit": {"oracle_exact": True},
    }


def test_grid_crosses_three_sizes_and_five_conditions():
    grid = _cell_grid()
    assert len(grid) == 15
    assert {(row["size"], row["condition"]) for row in grid} == {
        (size, condition) for size in SIZES for condition in CONDITIONS
    }
    assert [row["cell_index"] for row in grid] == list(range(15))


def test_repeat_arm_uses_exact_eighty_twenty_quota():
    for size in SIZES:
        assert _family_quota(size, "compose_g4_repeat20", "cross") == round(size * 0.80)
        assert _family_quota(size, "compose_g4_repeat20", "repeat") == round(size * 0.20)
        assert _family_quota(size, "compose_g1", "cross") == size
        assert _family_quota(size, "compose_g1", "repeat") == 0


def test_round3_materialization_is_equal_and_cumulative(tmp_path: Path):
    data = tmp_path / "atomic"
    write_examples(data / "train.jsonl", [atomic("a"), atomic("b")])
    args = SimpleNamespace(run_root=tmp_path / "run", atomic_data_dir=data)
    cell = {"cell_index": 0, "size": 3, "condition": "compose_g1", "cell_id": "n0003-compose_g1"}
    records = {
        (calls, "cross"): [accepted_record(calls, index) for index in range(3)]
        for calls in (2, 4, 8)
    }
    mix = _materialize(
        args=args,
        cell=cell,
        round_index=3,
        records=records,
        summaries={(calls, "cross"): {"accepted_count": 3} for calls in (2, 4, 8)},
    )
    assert mix == {
        "atomic_1_call": 3,
        "composed_2_call": 3,
        "composed_4_call": 3,
        "composed_8_call": 3,
        "total": 12,
        "size_per_regime": 3,
        "regimes": [1, 2, 4, 8],
        "trainer_seed": 7,
    }
    materialized = read_examples(
        tmp_path
        / "run/cells/n0003-compose_g1/round_03/training_materialized/train.jsonl"
    )
    assert len(materialized) == 12
    assert {example.metadata["training_origin"] for example in materialized} == {
        "atomic_gold_replay",
        "composed_2_call",
        "composed_4_call",
        "composed_8_call",
    }


def test_launcher_dry_run_uses_arrays_afterany_and_short_h200_jobs(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "manifest.json").write_text(
        json.dumps({"jobs": {}, "status": "prepared"}), encoding="utf-8"
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
    commands = [line for line in output.splitlines() if line.startswith("[INFO] Command: sbatch")]
    assert len(commands) == 4
    assert sum("--array=0-14%4" in line for line in commands) == 1
    assert sum("--gres=gpu:h200:1" in line for line in commands) == 2
    assert sum("--dependency=afterany:" in line for line in commands) == 3
    assert "bfcl-size-r1-mat" in output
    assert "bfcl-size-stage-r2-gen" in output
    assert "continue-submit" in output
    assert "DependencyNeverSatisfied" not in output


def test_requested_grid_is_persisted_and_read_back(tmp_path: Path):
    grid = _cell_grid((1000,), ("oracle", "direct_g1", "compose_g1"))
    assert [cell["cell_id"] for cell in grid] == [
        "n1000-oracle",
        "n1000-direct_g1",
        "n1000-compose_g1",
    ]
    assert [cell["cell_index"] for cell in grid] == [0, 1, 2]
    # Archived runs keep their own grid even after the defaults change.
    (tmp_path / "grid.json").write_text(json.dumps(grid), encoding="utf-8")
    assert _run_grid(tmp_path) == grid
    assert _run_grid(tmp_path / "missing") == _cell_grid()


def test_validation_cells_are_reported_separately_from_test_cells(tmp_path: Path):
    sets_root = tmp_path / "data/evaluation/sets"
    validation_root = tmp_path / "data/evaluation/validation_sets"
    examples = [atomic("a"), atomic("b")]
    write_examples(sets_root / "controlled_heldout_2.jsonl", examples)
    write_examples(validation_root / "controlled_heldout_2.jsonl", examples)
    names = [name for name, _rows in _evaluation_sets(tmp_path)]
    assert names == ["controlled_heldout_2", "validation_controlled_heldout_2"]
    # A run prepared before validation cells existed still evaluates.
    assert [name for name, _rows in _evaluation_sets(tmp_path / "missing")] == []


def test_training_budget_flags_reach_generated_commands():
    args = SimpleNamespace(max_steps=50, checkpoint_steps=(10, 20, 50))
    assert _training_flags(args) == ["--max-steps", "50", "--checkpoint-steps", "10,20,50"]
    assert _training_flags(SimpleNamespace(max_steps=None, checkpoint_steps=())) == []

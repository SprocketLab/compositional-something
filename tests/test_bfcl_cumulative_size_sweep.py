from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from self.coding.atomic_data import (
    AtomicExample,
    canonical_json,
    read_examples,
    write_examples,
    write_json,
)
from self.experiments.bfcl_cumulative_size_sweep import (
    CONDITIONS,
    SIZES,
    _cell_grid,
    _family_quota,
    _materialize,
    _checkpoint_adapters,
    _learned_conditions,
    _run_grid,
    _starting_adapter,
    _training_flags,
    SELECTION_FILE,
    select_checkpoint,
)
from self.coding.bfcl_composition import (
    build_controlled_evaluation_candidates,
    compose_component_predictions,
    oracle_example,
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
        evaluator={
            "functions": [function],
            "accepted_calls": [{"name": function["name"], "arguments": {"x": [1]}}],
        },
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


def test_learning_rate_axis_creates_one_cell_per_rate():
    grid = _cell_grid((1000,), ("oracle",), (5e-5, 2e-4))
    assert [cell["cell_id"] for cell in grid] == [
        "n1000-oracle-lr5em05",
        "n1000-oracle-lr2em04",
    ]
    assert [cell["learning_rate"] for cell in grid] == [5e-5, 2e-4]
    assert [cell["cell_index"] for cell in grid] == [0, 1]


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
    cell = {
        "cell_index": 0,
        "size": 3,
        "condition": "compose_g1",
        "learning_rate": 2e-4,
        "cell_id": "n0003-compose_g1-lr2em04",
    }
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
        / "run/cells/n0003-compose_g1-lr2em04/round_03/training_materialized/train.jsonl"
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
    assert len(commands) == 5
    assert sum("--array=0-14%4" in line for line in commands) == 1
    assert sum("--gres=gpu:h200:1" in line for line in commands) == 3
    assert sum("--dependency=afterany:" in line for line in commands) == 3
    # Seed and frozen composition need only the seed adapter, so the baseline
    # job runs unblocked alongside round 1.
    baseline = [line for line in commands if "bfcl-size-baselines" in line]
    assert len(baseline) == 1 and "--dependency" not in baseline[0]
    assert "evaluate-baselines" in output
    assert "bfcl-size-r1-mat" in output
    assert "bfcl-size-stage-r2-gen" in output
    assert "continue-submit" in output
    assert "DependencyNeverSatisfied" not in output


def test_requested_grid_is_persisted_and_read_back(tmp_path: Path):
    grid = _cell_grid((1000,), ("oracle", "direct_g1", "compose_g1"))
    assert [cell["cell_id"] for cell in grid] == [
        "n1000-oracle-lr2em04",
        "n1000-direct_g1-lr2em04",
        "n1000-compose_g1-lr2em04",
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


def test_checkpoint_adapters_are_ordered_and_include_the_final_step(tmp_path: Path):
    for step in (10, 50, 20):
        (tmp_path / f"adapter_step_{step:04d}").mkdir(parents=True)
    (tmp_path / "adapter").mkdir()
    (tmp_path / "metrics.json").write_text(json.dumps({"max_steps": 125}), encoding="utf-8")
    found = _checkpoint_adapters(tmp_path)
    assert [step for step, _path in found] == [10, 20, 50, 125]
    assert found[-1][1].name == "adapter"
    # A checkpoint requested at max_steps is the final adapter; not both.
    same = tmp_path / "same"
    (same / "adapter_step_0050").mkdir(parents=True)
    (same / "adapter").mkdir()
    (same / "metrics.json").write_text(json.dumps({"max_steps": 50}), encoding="utf-8")
    assert [step for step, _path in _checkpoint_adapters(same)] == [50]
    # A round that trained without intermediate checkpoints still resolves.
    bare = tmp_path / "bare"
    (bare / "adapter").mkdir(parents=True)
    (bare / "metrics.json").write_text(json.dumps({"max_steps": 30}), encoding="utf-8")
    assert [step for step, _path in _checkpoint_adapters(bare)] == [30]
    assert _checkpoint_adapters(tmp_path / "empty") == []


def test_frozen_composition_recomposes_evaluation_items(tmp_path: Path):
    items = [atomic(f"s{index}") for index in range(4)]
    cells = build_controlled_evaluation_candidates(
        items, component_counts=(2,), examples_per_cell=4, seed=7
    )
    public, oracle = cells["controlled_seen_2"]
    examples = [oracle_example(row, hidden) for row, hidden in zip(public, oracle)]
    raw = {
        str(spec["component_id"]): canonical_json(
            [{"name": f"f_{spec['source_component_ids'][0]}", "arguments": {"x": 1}}]
        )
        for candidate in public
        for spec in candidate["component_specs"]
    }
    predictions = [
        compose_component_predictions(candidate, raw, level="g1")["composed_target"]
        for candidate in public
    ]
    for candidate, prediction, example in zip(public, predictions, examples):
        assert json.loads(prediction) == json.loads(example.target)
        assert len(json.loads(prediction)) == candidate["component_count"]


def test_oracle_only_grid_skips_shared_seed_generation(tmp_path: Path):
    (tmp_path / "grid.json").write_text(
        json.dumps(_cell_grid((1000,), ("oracle",))), encoding="utf-8"
    )
    assert _learned_conditions(tmp_path) == []
    (tmp_path / "grid.json").write_text(
        json.dumps(_cell_grid((1000,), ("oracle", "compose_g1"))), encoding="utf-8"
    )
    assert _learned_conditions(tmp_path) == ["compose_g1"]


def test_training_budget_survives_the_staged_continuation_chain(tmp_path: Path):
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
        "MAX_STEPS": "50",
        "CHECKPOINT_STEPS": "10,20,50",
        "PYTHON_BIN": "/home/cs1095/.conda/envs/torch-env/bin/python",
    }
    completed = subprocess.run(
        ["bash", str(LAUNCHER)], cwd=ROOT, env=environment,
        check=True, capture_output=True, text=True,
    )
    output = completed.stdout + completed.stderr
    commands = [line for line in output.splitlines() if line.startswith("[INFO] Command: sbatch")]
    # Rounds 2 and 3 are built by a continuation job, so it must carry the
    # budget too or those rounds silently fall back to one epoch.
    for kind in ("train-evaluate", "continue-submit"):
        carrying = [
            line for line in commands
            if kind in line and "--max-steps 50" in line and "--checkpoint-steps 10,20,50" in line
        ]
        assert carrying, f"{kind} command lost the training budget"


def _round_with_checkpoints(root: Path, round_index: int, steps, max_steps: int):
    d = root / "cells/c" / f"round_{round_index:02d}"
    for step in steps:
        (d / f"adapter_step_{step:04d}").mkdir(parents=True)
    (d / "adapter").mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"max_steps": max_steps}), encoding="utf-8")
    return d


def test_round_continues_from_the_selected_checkpoint_when_one_is_recorded(tmp_path: Path):
    cell = {"cell_id": "c", "size": 1, "condition": "oracle", "learning_rate": 2e-4}
    args = SimpleNamespace(run_root=tmp_path, seed_adapter=tmp_path / "seed")
    d1 = _round_with_checkpoints(tmp_path, 1, (10, 20), max_steps=50)

    assert _starting_adapter(args, cell, 1) == tmp_path / "seed"
    # With no selection recorded, the final adapter is still the default.
    assert _starting_adapter(args, cell, 2) == d1 / "adapter"

    (d1 / SELECTION_FILE).write_text(
        json.dumps({"step": 20, "adapter": str(d1 / "adapter_step_0020")}), encoding="utf-8"
    )
    assert _starting_adapter(args, cell, 2) == d1 / "adapter_step_0020"


def test_selection_refuses_a_missing_adapter(tmp_path: Path):
    cell = {"cell_id": "c", "size": 1, "condition": "oracle", "learning_rate": 2e-4}
    args = SimpleNamespace(run_root=tmp_path, seed_adapter=tmp_path / "seed")
    d1 = _round_with_checkpoints(tmp_path, 1, (10,), max_steps=50)
    (d1 / SELECTION_FILE).write_text(
        json.dumps({"step": 99, "adapter": str(d1 / "adapter_step_0099")}), encoding="utf-8"
    )
    try:
        _starting_adapter(args, cell, 2)
    except FileNotFoundError as error:
        assert "adapter_step_0099" in str(error)
    else:
        raise AssertionError("a missing selected checkpoint must not fall back silently")


def test_select_checkpoint_scores_validation_cells_only(tmp_path: Path):
    cell = {"cell_index": 0, "size": 1, "condition": "oracle", "learning_rate": 2e-4, "cell_id": "c"}
    (tmp_path / "grid.json").write_text(json.dumps([cell]), encoding="utf-8")
    d1 = _round_with_checkpoints(tmp_path, 1, (10, 20), max_steps=50)
    rows = []
    for step, val, test in ((10, 0.50, 0.90), (20, 0.80, 0.10), (50, 0.60, 0.99)):
        rows.append({"step": step, "dataset": "validation_controlled_heldout_2", "exact_accuracy": val})
        rows.append({"step": step, "dataset": "controlled_heldout_2", "exact_accuracy": test})
    write_json(d1 / "checkpoint_evaluation/summary.json", {"rows": rows})
    select_checkpoint(SimpleNamespace(
        run_root=tmp_path, round=1, cell_index=0, step=None, select_on=None,
    ))
    payload = json.loads((d1 / SELECTION_FILE).read_text())
    # Step 20 wins on validation; step 50 would win on the test cells.
    assert payload["step"] == 20
    assert payload["adapter"] == str(d1 / "adapter_step_0020")
    assert payload["validation_scores"] == {"10": 0.50, "20": 0.80, "50": 0.60}

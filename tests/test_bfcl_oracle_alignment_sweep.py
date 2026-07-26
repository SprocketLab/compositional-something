from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from self.coding.atomic_data import canonical_json
from self.coding.bfcl_composition import audit_decision
from self.experiments.bfcl_oracle_alignment_sweep import (
    CHECKPOINT_STEPS,
    LEARNING_RATES,
    TARGET_STYLES,
    build_aligned_oracle_decision,
    sweep_cells,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launchers/self/submit_bfcl_oracle_alignment_sweep_ailab.sh"


def fixture_candidate() -> tuple[dict, dict]:
    functions = [
        {
            "name": "distance",
            "parameters": {
                "type": "object",
                "properties": {"unit": {"type": "string"}},
                "required": ["unit"],
            },
        },
        {
            "name": "city",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    ]
    component_specs = [
        {
            "component_id": "left",
            "source_component_ids": ["left"],
            "question": "distance",
            "functions": [functions[0]],
            "messages": [{"role": "user", "content": "distance"}],
            "expected_call_count": 1,
            "allow_exact_duplicates": False,
            "metadata": {},
        },
        {
            "component_id": "right",
            "source_component_ids": ["right"],
            "question": "city",
            "functions": [functions[1]],
            "messages": [{"role": "user", "content": "city"}],
            "expected_call_count": 1,
            "allow_exact_duplicates": False,
            "metadata": {},
        },
    ]
    candidate = {
        "candidate_id": "candidate",
        "source_group_id": "candidate",
        "split": "hidden_composition",
        "messages": [{"role": "user", "content": "both"}],
        "functions": functions,
        "component_count": 2,
        "source_component_ids": ["left", "right"],
        "question": "both",
        "template_id": "also",
        "template_partition": "train",
        "composition_family": "cross_function",
        "component_specs": component_specs,
        "independent": True,
    }
    oracle = {
        "candidate_id": "candidate",
        "canonical_calls": [
            {"name": "distance", "arguments": {"unit": "mi"}},
            {"name": "city", "arguments": {"location": "Seattle"}},
        ],
        "accepted_calls": [
            {"name": "distance", "arguments": {"unit": ["mi", "miles"]}},
            {"name": "city", "arguments": {"location": ["Seattle", "Seattle, WA"]}},
        ],
        "component_oracles": [
            {
                "component_id": "left",
                "canonical_calls": [{"name": "distance", "arguments": {"unit": "mi"}}],
                "accepted_calls": [
                    {"name": "distance", "arguments": {"unit": ["mi", "miles"]}}
                ],
            },
            {
                "component_id": "right",
                "canonical_calls": [{"name": "city", "arguments": {"location": "Seattle"}}],
                "accepted_calls": [
                    {
                        "name": "city",
                        "arguments": {"location": ["Seattle", "Seattle, WA"]},
                    }
                ],
            },
        ],
    }
    return candidate, oracle


def test_aligned_oracle_prefers_exact_end_to_end_aliases_and_order():
    candidate, oracle = fixture_candidate()
    direct = canonical_json(
        [
            {"name": "city", "arguments": {"location": "Seattle, WA"}},
            {"name": "distance", "arguments": {"unit": "miles"}},
        ]
    )
    decision, trace = build_aligned_oracle_decision(
        candidate,
        oracle,
        raw_components={"left": "[]", "right": "[]"},
        raw_direct={"candidate": direct},
    )
    assert trace["alignment_source"] == "direct_exact"
    assert trace["differs_from_canonical"]
    assert decision["composed_target"] == direct
    assert audit_decision(candidate, oracle, decision)["oracle_exact"]


def test_aligned_oracle_uses_exact_components_and_canonical_fallbacks():
    candidate, oracle = fixture_candidate()
    decision, trace = build_aligned_oracle_decision(
        candidate,
        oracle,
        raw_components={
            "left": canonical_json(
                [{"name": "distance", "arguments": {"unit": "miles"}}]
            ),
            "right": canonical_json(
                [{"name": "city", "arguments": {"location": "Portland"}}]
            ),
        },
        raw_direct={"candidate": "[]"},
    )
    assert trace["alignment_source"] == "component_or_canonical"
    assert [row["source"] for row in trace["component_trace"]] == [
        "component_exact",
        "canonical_fallback",
    ]
    assert json.loads(decision["composed_target"]) == [
        {"name": "distance", "arguments": {"unit": "miles"}},
        {"name": "city", "arguments": {"location": "Seattle"}},
    ]
    assert audit_decision(candidate, oracle, decision)["oracle_exact"]


def test_sweep_grid_is_full_target_lr_checkpoint_product():
    cells = sweep_cells()
    assert len(cells) == len(TARGET_STYLES) * len(LEARNING_RATES) * len(CHECKPOINT_STEPS) == 18
    assert {cell.target_style for cell in cells} == set(TARGET_STYLES)
    assert {cell.learning_rate for cell in cells} == set(LEARNING_RATES)
    assert {cell.max_steps for cell in cells} == set(CHECKPOINT_STEPS)
    assert [cell.index for cell in cells] == list(range(18))


def test_launcher_dry_run_uses_short_bounded_h200_array(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "manifest.json").write_text(
        json.dumps({"status": "prepared", "jobs": {}}),
        encoding="utf-8",
    )
    (run_root / "grid.json").write_text(
        json.dumps([cell.to_dict() for cell in sweep_cells()]),
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
    assert "--array=0-17%4" in output
    assert "--time=00:45:00" in output
    assert "--gres=gpu:h200:1" in output
    assert "--dependency=afterany:dryrun-array" in output
    assert "SLURM_ARRAY_TASK_ID" in output


def test_launcher_has_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)

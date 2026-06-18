from __future__ import annotations

import json
from pathlib import Path

from self import adaptive_self_improvement


def test_adaptive_controller_config_dry_run_writes_selection_and_traces(tmp_path: Path):
    fixture_path = tmp_path / "fixtures.jsonl"
    rows = [
        {
            "id": "invalid",
            "raw_output": {
                "left": 8,
                "right": 8,
                "guard": "none",
            },
            "frontier_delta": 1.0,
            "final_accuracy": 1.0,
        },
        {
            "id": "valid-low",
            "raw_output": {
                "left": 8,
                "right": 9,
                "guard": "none",
            },
            "frontier_delta": 0.1,
            "final_accuracy": 0.8,
        },
        {
            "id": "valid-high",
            "raw_output": {
                "left": 9,
                "right": 9,
                "guard": "reject_boundary_continue",
            },
            "frontier_delta": 0.2,
            "final_accuracy": 0.9,
        },
    ]
    fixture_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Plan\n", encoding="utf-8")
    output_dir = tmp_path / "pilot"

    args = adaptive_self_improvement.build_parser().parse_args(
        [
            "--task",
            "run_length",
            "--condition",
            "config",
            "--proposal-fixture-jsonl",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
            "--dry-run-proposals",
            "--source-min-allowed",
            "8",
            "--source-max-allowed",
            "16",
            "--frontier-min-allowed",
            "17",
            "--frontier-max-allowed",
            "48",
            "--min-examples-per-size",
            "1",
            "--max-examples-per-size",
            "100",
            "--plan-log-path",
            str(plan_path),
        ]
    )

    summary = adaptive_self_improvement.run(args)
    selected = json.loads((output_dir / "selected_proposal.json").read_text(encoding="utf-8"))
    trace_rows = [
        json.loads(line)
        for line in (output_dir / "trace_examples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = json.loads((output_dir / "proposal_results.json").read_text(encoding="utf-8"))

    assert summary["valid_count"] == 2
    assert selected["id"] == "valid-high"
    assert selected["reward"] > results[1]["reward"]
    assert [row["metadata"]["proposal_id"] for row in trace_rows] == ["valid-high", "valid-low"]
    assert "Selected proposal: `valid-high`" in plan_path.read_text(encoding="utf-8")


def test_adaptive_controller_writes_weak_regime_frontier_and_quality_metrics(tmp_path: Path):
    fixture_path = tmp_path / "fixtures.jsonl"
    fixture_path.write_text(
        json.dumps(
            {
                "id": "valid-frontier",
                "raw_output": {
                    "left": 4,
                    "right": 4,
                    "guard": "reject_boundary_carry",
                },
                "frontier_delta": 0.1,
                "final_accuracy": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    diagnostics = {
        "per_size_accuracy": {"8": 0.92, "9": 0.72},
        "composed_eval_slices": {
            "boundary_carry": {
                "accuracy": 0.45,
                "count": 20,
                "per_size_accuracy": {"9": 0.45},
            }
        },
    }
    output_dir = tmp_path / "pilot"

    args = adaptive_self_improvement.build_parser().parse_args(
        [
            "--task",
            "addition",
            "--condition",
            "config",
            "--proposal-fixture-jsonl",
            str(fixture_path),
            "--output-dir",
            str(output_dir),
            "--dry-run-proposals",
            "--frontier-policy",
            "weak_regime",
            "--frontier-diagnostics-json",
            json.dumps(diagnostics),
            "--init-final-accuracy",
            "0.7",
            "--source-min-allowed",
            "3",
            "--source-max-allowed",
            "7",
            "--frontier-min-allowed",
            "8",
            "--frontier-max-allowed",
            "10",
            "--min-examples-per-size",
            "1",
            "--max-examples-per-size",
            "100",
            "--plan-log-path",
            str(tmp_path / "missing.md"),
        ]
    )

    summary = adaptive_self_improvement.run(args)
    frontier = json.loads((output_dir / "frontier_selection.json").read_text(encoding="utf-8"))
    metrics = json.loads((output_dir / "proposal_quality_metrics.json").read_text(encoding="utf-8"))
    prompt = json.loads((output_dir / "proposal_prompt.json").read_text(encoding="utf-8"))

    assert frontier["selected"]["slice_name"] == "boundary_carry"
    assert frontier["frontier_min"] == 9
    assert metrics["valid_rate"] == 1.0
    assert abs(summary["proposal_quality_metrics"]["best_reward"] - 0.11) < 1e-12
    selected = json.loads((output_dir / "selected_proposal.json").read_text(encoding="utf-8"))
    assert abs(selected["final_accuracy_delta"] - 0.1) < 1e-12
    assert selected["init_final_accuracy"] == 0.7
    assert "adaptive_frontier_selection" in prompt["user"]

import json
from pathlib import Path

from self.analysis.artifacts import (
    adaptive_attempt_records,
    adaptive_candidate_records,
    adaptive_proposal_records,
    adaptive_trace_rows,
    discover_adaptive_runs,
    load_adaptive_run,
    load_self_improvement_rounds,
    per_size_accuracy_records,
    read_json,
    read_jsonl,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_adaptive_artifact_loader_flattens_attempts_proposals_and_candidates(tmp_path: Path):
    run_dir = tmp_path / "root" / "addition-config"
    attempt_dir = run_dir / "attempt_0001"
    _write_json(
        run_dir / "summary.json",
        {
            "task": "addition",
            "condition": "config",
            "attempts_completed": 1,
            "selected_rounds_completed": 1,
            "init_final_accuracy": 0.4,
        },
    )
    _write_json(run_dir / "adaptive_candidate_training_results.json", [{"round": 0, "eval_accuracy": 0.4}])
    _write_json(run_dir / "round_00" / "metrics.json", {"eval_accuracy": 0.4})
    _write_json(
        attempt_dir / "attempt_summary.json",
        {
            "attempt": 1,
            "selected_round": 1,
            "no_selection": False,
            "candidate_count": 1,
            "selected": {"id": "model_candidate_0"},
            "trace_count": 1,
        },
    )
    _write_json(
        attempt_dir / "proposal_results.json",
        [
            {
                "id": "model_candidate_0",
                "proposal_index": 0,
                "valid": True,
                "validation_category": "valid",
                "parsed_proposal": {"left": 3, "right": 7, "target": 10, "guard": "none"},
            },
            {
                "id": "model_candidate_1",
                "proposal_index": 1,
                "valid": False,
                "validation_category": "parse_error",
            },
        ],
    )
    _write_json(
        attempt_dir / "candidate_metrics.json",
        [
            {
                "id": "model_candidate_0",
                "index": 0,
                "valid": True,
                "reward": 0.12,
                "final_accuracy": 0.52,
                "frontier_delta": 0.08,
                "parsed_proposal": {"left": 3, "right": 7, "target": 10, "guard": "none"},
                "per_size_accuracy": {"10": 0.8},
            }
        ],
    )
    _write_jsonl(attempt_dir / "trace_examples.jsonl", [{"prompt": "p", "target": "t"}])

    assert discover_adaptive_runs(tmp_path / "root") == [run_dir]
    run = load_adaptive_run(run_dir)
    assert run.summary["task"] == "addition"
    assert run.results[0]["round"] == 0
    assert run.seed_metrics["eval_accuracy"] == 0.4
    assert len(run.attempts) == 1

    attempt_rows = adaptive_attempt_records(run)
    assert attempt_rows == [
        {
            "run_dir": str(run_dir),
            "run_name": "addition-config",
            "task": "addition",
            "condition": "config",
            "selected_rounds_completed": 1,
            "attempts_completed": 1,
            "init_final_accuracy": 0.4,
            "attempt_dir": str(attempt_dir),
            "attempt": 1,
            "selected_round": 1,
            "no_selection": False,
            "candidate_count": 1,
            "proposal_count": 2,
            "valid_proposal_count": 1,
            "selected_id": "model_candidate_0",
            "trace_count": 1,
            "outcome_trace_count": None,
            "proposal_trace_buffer_size": None,
            "outcome_trace_buffer_size": None,
        }
    ]

    proposal_rows = adaptive_proposal_records(run)
    assert len(proposal_rows) == 2
    assert proposal_rows[0]["proposal_left"] == 3
    assert proposal_rows[0]["proposal_target"] == 10
    assert proposal_rows[1]["validation_category"] == "parse_error"

    candidate_rows = adaptive_candidate_records(run)
    assert candidate_rows[0]["selected_candidate"] is True
    assert candidate_rows[0]["proposal_guard"] == "none"
    assert candidate_rows[0]["per_size_accuracy"] == {"10": 0.8}
    assert adaptive_trace_rows(run.attempts[0]) == [{"prompt": "p", "target": "t"}]


def test_generic_json_and_self_improvement_round_helpers(tmp_path: Path):
    assert read_json(tmp_path / "missing.json", {"default": True}) == {"default": True}
    assert read_jsonl(tmp_path / "missing.jsonl") == []

    run_dir = tmp_path / "classic"
    _write_json(
        run_dir / "self_improvement_results.json",
        [
            {"round": 0, "per_size_accuracy": {"3": 0.9, "4": 0.8}},
            {"round": 1, "per_size_accuracy": {"5": 0.7}},
        ],
    )

    rounds = load_self_improvement_rounds(run_dir)
    assert rounds[1]["round"] == 1
    assert per_size_accuracy_records(rounds, run_name="classic") == [
        {"round": 0, "size": 3, "accuracy": 0.9, "run_name": "classic"},
        {"round": 0, "size": 4, "accuracy": 0.8, "run_name": "classic"},
        {"round": 1, "size": 5, "accuracy": 0.7, "run_name": "classic"},
    ]

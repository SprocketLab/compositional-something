import json
from pathlib import Path

from self.analysis import (
    adaptive_artifact_common,
    adaptive_artifacts,
    adaptive_candidate_artifacts,
    nonadaptive_artifacts,
)
from self.analysis.artifacts import (
    adaptive_attempt_records,
    adaptive_candidate_artifact_records,
    adaptive_candidate_per_size_records,
    adaptive_candidate_records,
    adaptive_candidate_train_mix_records,
    adaptive_prompt_records,
    adaptive_proposal_grpo_records,
    adaptive_proposal_records,
    adaptive_selected_per_size_timeline_records,
    adaptive_trace_records,
    adaptive_trace_rows,
    discover_adaptive_runs,
    iter_candidate_dirs,
    load_adaptive_candidates,
    load_adaptive_run,
    load_self_improvement_rounds,
    per_size_accuracy_records,
    read_json,
    read_jsonl,
    resolve_self_improvement_results_path,
)
from self.analysis.plot_self_improvement_figure import load_records, resolve_results_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_adaptive_artifact_loader_flattens_attempts_proposals_and_candidates(tmp_path: Path):
    assert adaptive_attempt_records is adaptive_artifacts.adaptive_attempt_records
    assert adaptive_artifacts.AdaptiveAttemptArtifacts is adaptive_artifact_common.AdaptiveAttemptArtifacts
    assert adaptive_candidate_artifacts.AdaptiveRunArtifacts is adaptive_artifact_common.AdaptiveRunArtifacts
    assert load_adaptive_run is adaptive_artifacts.load_adaptive_run
    assert load_adaptive_candidates is adaptive_candidate_artifacts.load_adaptive_candidates
    assert load_adaptive_candidates is adaptive_artifacts.load_adaptive_candidates
    assert adaptive_candidate_artifact_records is adaptive_candidate_artifacts.adaptive_candidate_artifact_records

    run_dir = tmp_path / "root" / "addition-config"
    attempt_dir = run_dir / "attempt_0001"
    candidate_dir = attempt_dir / "candidates" / "candidate_00"
    failed_candidate_dir = attempt_dir / "candidates" / "candidate_01"
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
        attempt_dir / "proposal_prompt.json",
        {"system": "choose a config", "user": "state: addition"},
    )
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
    _write_json(
        candidate_dir / "candidate_metrics.json",
        {
            "id": "model_candidate_0",
            "index": 0,
            "valid": True,
            "reward": 0.12,
            "final_accuracy": 0.52,
            "frontier_delta": 0.08,
            "parsed_proposal": {"left": 3, "right": 7, "target": 10, "guard": "none"},
        },
    )
    _write_json(
        candidate_dir / "train_mix_summary.json",
        {
            "source_examples": 5,
            "pseudo_examples": 7,
            "task_train_examples": 12,
            "outcome_trace_replay_examples": 1,
            "candidate_proposal_trace_examples": 1,
            "total_train_examples": 14,
        },
    )
    _write_json(
        failed_candidate_dir / "worker_failure.json",
        {
            "error": "worker OOM",
        },
    )
    _write_jsonl(attempt_dir / "trace_examples.jsonl", [{"prompt": "p", "target": "t"}])
    _write_jsonl(attempt_dir / "selected_proposal_trace.jsonl", [{"prompt": "sp", "target": "st"}])
    _write_jsonl(attempt_dir / "outcome_trace_examples.jsonl", [{"prompt": "op", "target": "ot"}])
    _write_json(
        attempt_dir / "proposal_grpo" / "proposal_grpo_metrics.json",
        {
            "applied": True,
            "loss": 0.13,
            "reward_mean": 0.2,
        },
    )
    _write_jsonl(
        attempt_dir / "proposal_grpo" / "proposal_grpo_traces.jsonl",
        [{"completion": "{}", "reward": 1.0}],
    )

    assert discover_adaptive_runs(tmp_path / "root") == [run_dir]
    run = load_adaptive_run(run_dir)
    assert run.summary["task"] == "addition"
    assert run.results[0]["round"] == 0
    assert run.seed_metrics["eval_accuracy"] == 0.4
    assert len(run.attempts) == 1
    assert iter_candidate_dirs(run.attempts[0]) == [candidate_dir, failed_candidate_dir]

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

    prompt_rows = adaptive_prompt_records(run)
    assert prompt_rows == [
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
            "proposal_prompt_path": str(attempt_dir / "proposal_prompt.json"),
            "system": "choose a config",
            "user": "state: addition",
            "system_chars": 15,
            "user_chars": 15,
        }
    ]

    candidate_rows = adaptive_candidate_records(run)
    assert candidate_rows[0]["selected_candidate"] is True
    assert candidate_rows[0]["proposal_guard"] == "none"
    assert candidate_rows[0]["per_size_accuracy"] == {"10": 0.8}
    candidate_artifacts = load_adaptive_candidates(run.attempts[0])
    assert [candidate.candidate_index for candidate in candidate_artifacts] == [0, 1]
    assert candidate_artifacts[0].candidate_id == "model_candidate_0"
    assert candidate_artifacts[0].train_mix_summary["total_train_examples"] == 14
    assert candidate_artifacts[1].candidate_id == "candidate_01"
    assert candidate_artifacts[1].worker_failure == {"error": "worker OOM"}
    artifact_rows = adaptive_candidate_artifact_records(run)
    assert [
        (
            row["candidate_id"],
            row["candidate_index"],
            row["has_metrics"],
            row["has_train_mix_summary"],
            row["has_worker_failure"],
            row["worker_failure"],
        )
        for row in artifact_rows
    ] == [
        ("model_candidate_0", 0, True, True, False, None),
        ("candidate_01", 1, False, False, True, {"error": "worker OOM"}),
    ]
    candidate_size_rows = adaptive_candidate_per_size_records(run)
    assert candidate_size_rows == [
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
            "candidate_id": "model_candidate_0",
            "candidate_index": 0,
            "selected_candidate": True,
            "valid": True,
            "reward": 0.12,
            "metric_key": "per_size_accuracy",
            "proposal_left": 3,
            "proposal_right": 7,
            "proposal_target": 10,
            "proposal_guard": "none",
            "size": 10,
            "accuracy": 0.8,
        }
    ]
    train_mix_rows = adaptive_candidate_train_mix_records(run)
    assert train_mix_rows == [
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
            "candidate_dir": str(candidate_dir),
            "candidate_index": 0,
            "candidate_id": "model_candidate_0",
            "selected_candidate": True,
            "valid": True,
            "reward": 0.12,
            "final_accuracy": 0.52,
            "frontier_delta": 0.08,
            "train_mix_summary_path": str(candidate_dir / "train_mix_summary.json"),
            "proposal_left": 3,
            "proposal_right": 7,
            "proposal_target": 10,
            "proposal_guard": "none",
            "source_examples": 5,
            "pseudo_examples": 7,
            "task_train_examples": 12,
            "outcome_trace_replay_examples": 1,
            "candidate_proposal_trace_examples": 1,
            "total_train_examples": 14,
        }
    ]
    grpo_rows = adaptive_proposal_grpo_records(run)
    assert grpo_rows == [
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
            "proposal_grpo_metrics_path": str(
                attempt_dir / "proposal_grpo" / "proposal_grpo_metrics.json"
            ),
            "applied": True,
            "loss": 0.13,
            "reward_mean": 0.2,
        }
    ]
    assert adaptive_trace_rows(run.attempts[0]) == [{"prompt": "p", "target": "t"}]
    trace_rows = adaptive_trace_records(run)
    assert [
        (row["trace_name"], row["trace_index"], row["trace"])
        for row in trace_rows
    ] == [
        ("trace_examples.jsonl", 0, {"prompt": "p", "target": "t"}),
        ("selected_proposal_trace.jsonl", 0, {"prompt": "sp", "target": "st"}),
        ("outcome_trace_examples.jsonl", 0, {"prompt": "op", "target": "ot"}),
        ("proposal_grpo/proposal_grpo_traces.jsonl", 0, {"completion": "{}", "reward": 1.0}),
    ]


def test_generic_json_and_self_improvement_round_helpers(tmp_path: Path):
    assert load_self_improvement_rounds is nonadaptive_artifacts.load_self_improvement_rounds
    assert per_size_accuracy_records is nonadaptive_artifacts.per_size_accuracy_records
    assert resolve_self_improvement_results_path is nonadaptive_artifacts.resolve_self_improvement_results_path

    assert read_json(tmp_path / "missing.json", {"default": True}) == {"default": True}
    assert read_jsonl(tmp_path / "missing.jsonl") == []

    run_dir = tmp_path / "classic"
    expected_results_path = run_dir / "self_improvement_results.json"
    _write_json(
        expected_results_path,
        [
            {"round": 0, "per_size_accuracy": {"3": 0.9, "4": 0.8}},
            {"round": 1, "per_size_accuracy": {"5": 0.7}},
        ],
    )

    assert resolve_self_improvement_results_path(run_dir) == expected_results_path
    assert resolve_self_improvement_results_path(expected_results_path) == expected_results_path
    rounds = load_self_improvement_rounds(run_dir)
    assert rounds[1]["round"] == 1
    assert per_size_accuracy_records(rounds, run_name="classic") == [
        {"round": 0, "size": 3, "accuracy": 0.9, "run_name": "classic"},
        {"round": 0, "size": 4, "accuracy": 0.8, "run_name": "classic"},
        {"round": 1, "size": 5, "accuracy": 0.7, "run_name": "classic"},
    ]


def test_plot_self_improvement_figure_uses_stable_round_loader(tmp_path: Path):
    run_dir = tmp_path / "classic"
    results_path = run_dir / "self_improvement_results.json"
    _write_json(
        results_path,
        [
            {"round": 2, "eval_accuracy": 0.7},
            {"round": 0, "eval_accuracy": 0.4},
            {"round": 1, "eval_accuracy": 0.6},
        ],
    )

    assert resolve_results_path(str(run_dir)) == results_path
    assert resolve_results_path(str(results_path)) == results_path
    assert [record["round"] for record in load_records(results_path)] == [0, 1, 2]

    _write_json(tmp_path / "bad.json", {"round": 0})
    try:
        load_records(tmp_path / "bad.json")
    except ValueError as exc:
        assert "Expected list of round records" in str(exc)
    else:
        raise AssertionError("load_records should reject non-list payloads")


def test_adaptive_selected_per_size_timeline_records_carries_forward(tmp_path: Path):
    run_dir = tmp_path / "root" / "addition-config"
    attempt1 = run_dir / "attempt_0001"
    attempt2 = run_dir / "attempt_0002"
    _write_json(
        run_dir / "summary.json",
        {
            "task": "addition",
            "condition": "config",
            "attempts_completed": 2,
            "selected_rounds_completed": 1,
            "init_final_accuracy": 0.4,
        },
    )
    _write_json(
        run_dir / "round_00" / "metrics.json",
        {
            "eval_accuracy": 0.4,
            "per_size_accuracy": {"9": 0.5, "10": 0.1},
        },
    )
    _write_json(
        attempt1 / "attempt_summary.json",
        {
            "attempt": 1,
            "selected_round": 1,
            "selected": {"id": "model_candidate_0"},
        },
    )
    _write_json(
        attempt1 / "selected_candidate.json",
        {
            "id": "model_candidate_0",
            "final_accuracy": 0.52,
            "parsed_proposal": {"left": 3, "right": 7, "target": 10, "guard": "none"},
            "per_size_accuracy": {"10": 0.8, "11": 0.2},
        },
    )
    _write_json(
        attempt2 / "attempt_summary.json",
        {
            "attempt": 2,
            "selected_round": 1,
            "no_selection": True,
        },
    )

    rows = adaptive_selected_per_size_timeline_records(run_dir)
    compact = [
        (
            row["attempt"],
            row["selected_round"],
            row["selected_this_attempt"],
            row["checkpoint_source"],
            row["selected_id"],
            row.get("proposal_target"),
            row["size"],
            row["accuracy"],
            row["checkpoint_final_accuracy"],
        )
        for row in rows
    ]
    assert compact == [
        (0, 0, False, "seed", None, None, 9, 0.5, 0.4),
        (0, 0, False, "seed", None, None, 10, 0.1, 0.4),
        (1, 1, True, "selected_candidate", "model_candidate_0", 10, 10, 0.8, 0.52),
        (1, 1, True, "selected_candidate", "model_candidate_0", 10, 11, 0.2, 0.52),
        (2, 1, False, "carried_forward", None, None, 10, 0.8, 0.52),
        (2, 1, False, "carried_forward", None, None, 11, 0.2, 0.52),
    ]

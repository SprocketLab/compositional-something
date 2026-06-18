from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

from self.multiplication_rectangular import RectangularMultiplicationExample, iter_partition_grid
from self.multiplication_rectangular_tune import (
    build_stage1_manifest,
    build_stage2_followups,
    build_stage3_followups,
    choose_final_stage2_candidate,
    choose_stage1_top2,
    score_stage1_compose_candidate,
    score_stage2_candidate,
)
from self.rectangular_multiplication_self_improvement import (
    build_composed_pseudo_examples,
    build_frontier_partition_train_counts,
    summarize_accuracy_by_a_digits,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "launchers" / "self" / "submit_multiplication_rectangular_tune_mig.sh"


def _write_results(
    path: Path,
    *,
    seed_test_min: float,
    frontier_train_accuracy: float,
    frontier_test_accuracy: float,
    row_234: tuple[float, float, float],
) -> str:
    row_means = {"2": row_234[0], "3": row_234[1], "4": row_234[2]}
    payload = {
        "task": "rectangular_multiplication_self_improvement",
        "rounds": [
            {
                "round": 8,
                "max_b_digits": 16,
                "results": {
                    "seed_test": {
                        "accuracy": seed_test_min,
                        "min_partition_accuracy": seed_test_min,
                        "per_partition_accuracy": {},
                    },
                    "frontier_train": {
                        "accuracy": frontier_train_accuracy,
                        "min_partition_accuracy": 0.0,
                        "per_partition_accuracy": {},
                        "mean_accuracy_by_a_digits": row_means,
                        "min_accuracy_by_a_digits": row_means,
                    },
                    "frontier_test": {
                        "accuracy": frontier_test_accuracy,
                        "min_partition_accuracy": 0.0,
                        "per_partition_accuracy": {},
                        "mean_accuracy_by_a_digits": row_means,
                        "min_accuracy_by_a_digits": row_means,
                    },
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_build_frontier_partition_train_counts_hard_rows_v1():
    counts = build_frontier_partition_train_counts(
        partitions=[(1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2)],
        base_count=100,
        frontier_row_profile="hard_rows_v1",
    )

    assert counts[(1, 2)] == 100
    assert counts[(2, 2)] == 400
    assert counts[(3, 2)] == 400
    assert counts[(4, 2)] == 200
    assert counts[(5, 2)] == 100
    assert counts[(6, 2)] == 100


def test_build_composed_pseudo_examples_shortcuts_zero_components():
    example = RectangularMultiplicationExample(
        a=631,
        b=203,
        a_digits=3,
        b_digits=3,
        format_version="symbolic_v1",
    )
    component_predictions = {
        (3, 1, 631, 2): "1262",
        (3, 1, 631, 3): "1893",
    }

    pseudo_examples, diagnostics = build_composed_pseudo_examples(
        train_examples=[example],
        raw_output_map=component_predictions,
        supported_partitions=[(3, 1)],
        corruption_rate=0.0,
        rng=random.Random(0),
        diagnostics_mode="compose",
    )

    assert len(pseudo_examples) == 1
    assert pseudo_examples[0].target_override == "128093"
    assert diagnostics["retained_total"] == 1
    assert diagnostics["correct_value_total"] == 1
    assert diagnostics["missing_total"] == 0
    assert diagnostics["zero_shortcut_component_total"] == 1


def test_build_composed_pseudo_examples_supports_square_seed_partitions():
    example = RectangularMultiplicationExample(
        a=12345678,
        b=87654321,
        a_digits=8,
        b_digits=8,
        format_version="symbolic_v1",
    )
    square_seed = iter_partition_grid(1, 3, 1, 3)
    component_predictions = {
        (3, 3, 678, 321): "217638",
        (3, 3, 678, 654): "443412",
        (3, 2, 678, 87): "58986",
        (3, 3, 345, 321): "110745",
        (3, 3, 345, 654): "225630",
        (3, 2, 345, 87): "30015",
        (2, 3, 12, 321): "3852",
        (2, 3, 12, 654): "7848",
        (2, 2, 12, 87): "1044",
    }

    pseudo_examples, diagnostics = build_composed_pseudo_examples(
        train_examples=[example],
        raw_output_map=component_predictions,
        supported_partitions=square_seed,
        corruption_rate=0.0,
        rng=random.Random(0),
        diagnostics_mode="compose",
    )

    assert len(pseudo_examples) == 1
    assert pseudo_examples[0].target_override == example.target()
    assert diagnostics["retained_total"] == 1
    assert diagnostics["correct_value_total"] == 1
    assert diagnostics["missing_total"] == 0


def test_summarize_accuracy_by_a_digits_adds_mean_and_min():
    summary = summarize_accuracy_by_a_digits(
        {
            "count": 5,
            "accuracy": 0.5,
            "per_partition_accuracy": {
                "1x8": 0.9,
                "1x9": 0.7,
                "2x2": 0.3,
                "2x3": 0.5,
                "3x2": 0.1,
            },
            "min_partition_accuracy": 0.1,
        }
    )

    assert summary["mean_accuracy_by_a_digits"] == {"1": 0.8, "2": 0.4, "3": 0.1}
    assert summary["min_accuracy_by_a_digits"] == {"1": 0.7, "2": 0.3, "3": 0.1}


def test_build_stage1_manifest_has_expected_job_counts(tmp_path: Path):
    manifest = build_stage1_manifest(
        out_root=tmp_path / "tune",
        log_dir=tmp_path / "logs",
        python_bin="python",
        seed_model=tmp_path / "seed_model",
    )

    stage1_jobs = manifest["stage1_jobs"]
    assert len(stage1_jobs) == 6
    assert all(entry["baseline"] == "compose" for entry in stage1_jobs)
    assert {entry["frontier_row_profile"] for entry in stage1_jobs} == {"uniform", "hard_rows_v1"}
    assert {entry["expand_b_digits"] for entry in stage1_jobs} == {1}
    assert {entry["num_expand_rounds"] for entry in stage1_jobs} == {8}


def test_build_stage2_followups_emits_two_direct_jobs(tmp_path: Path):
    followups = build_stage2_followups(
        [
            {"schedule_label": "sched_a", "results_path": "/tmp/a.json"},
            {"schedule_label": "sched_b", "results_path": "/tmp/b.json"},
        ],
        out_root=tmp_path / "tune",
    )

    assert len(followups) == 2
    assert {entry["baseline"] for entry in followups} == {"direct"}
    assert {entry["schedule_label"] for entry in followups} == {"sched_a", "sched_b"}


def test_build_stage3_followups_emits_two_backfill_jobs(tmp_path: Path):
    followups = build_stage3_followups(
        {"schedule_label": "sched_a", "results_path": "/tmp/compose.json", "direct_results_path": "/tmp/direct.json"},
        out_root=tmp_path / "tune",
    )

    assert len(followups) == 2
    assert {entry["baseline"] for entry in followups} == {"short_only", "compose_corrupt"}


def test_choose_stage1_top2_prefers_viable_row_strength(tmp_path: Path):
    candidate_a = {
        "schedule_label": "sched_a",
        "results_path": _write_results(
            tmp_path / "a.json",
            seed_test_min=0.97,
            frontier_train_accuracy=0.13,
            frontier_test_accuracy=0.13,
            row_234=(0.06, 0.06, 0.06),
        ),
    }
    candidate_b = {
        "schedule_label": "sched_b",
        "results_path": _write_results(
            tmp_path / "b.json",
            seed_test_min=0.96,
            frontier_train_accuracy=0.14,
            frontier_test_accuracy=0.12,
            row_234=(0.08, 0.08, 0.08),
        ),
    }
    candidate_c = {
        "schedule_label": "sched_c",
        "results_path": _write_results(
            tmp_path / "c.json",
            seed_test_min=0.94,
            frontier_train_accuracy=0.11,
            frontier_test_accuracy=0.10,
            row_234=(0.03, 0.03, 0.03),
        ),
    }

    scored = [
        score_stage1_compose_candidate(candidate_a),
        score_stage1_compose_candidate(candidate_b),
        score_stage1_compose_candidate(candidate_c),
    ]
    selected = choose_stage1_top2(scored)

    assert [entry["schedule_label"] for entry in selected] == ["sched_b", "sched_a"]
    assert all(entry["selection_mode"] == "paper_viable" for entry in selected)


def test_choose_final_stage2_candidate_prefers_compose_direct_gap(tmp_path: Path):
    compose_a = score_stage1_compose_candidate(
        {
            "schedule_label": "sched_a",
            "results_path": _write_results(
                tmp_path / "compose_a.json",
                seed_test_min=0.97,
                frontier_train_accuracy=0.13,
                frontier_test_accuracy=0.16,
                row_234=(0.07, 0.07, 0.07),
            ),
        }
    )
    compose_b = score_stage1_compose_candidate(
        {
            "schedule_label": "sched_b",
            "results_path": _write_results(
                tmp_path / "compose_b.json",
                seed_test_min=0.98,
                frontier_train_accuracy=0.13,
                frontier_test_accuracy=0.15,
                row_234=(0.09, 0.09, 0.09),
            ),
        }
    )

    scored = [
        score_stage2_candidate(
            compose_a,
            direct_results_path=_write_results(
                tmp_path / "direct_a.json",
                seed_test_min=0.98,
                frontier_train_accuracy=0.05,
                frontier_test_accuracy=0.08,
                row_234=(0.01, 0.01, 0.01),
            ),
        ),
        score_stage2_candidate(
            compose_b,
            direct_results_path=_write_results(
                tmp_path / "direct_b.json",
                seed_test_min=0.98,
                frontier_train_accuracy=0.05,
                frontier_test_accuracy=0.10,
                row_234=(0.01, 0.01, 0.01),
            ),
        ),
    ]

    selected = choose_final_stage2_candidate(scored)
    assert selected["schedule_label"] == "sched_a"
    assert selected["compose_minus_direct_frontier_test_accuracy"] == 0.08


def test_multiplication_rectangular_tune_wrapper_has_valid_bash_syntax():
    assert WRAPPER.exists()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_submit_multiplication_rectangular_tune_wrapper_dry_run_prints_expected_counts(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "mult_tune")
    env["LOG_DIR"] = str(tmp_path / "logs")
    env["SEED_MODEL"] = str(tmp_path / "seed_model")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "self.multiplication_rectangular_tune submit" in stdout
    assert "Wrote stage-1 manifest" in stdout
    assert "Stage-1 counts: compose=6" in stdout
    assert "mult-rect-tune-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout

    manifest = json.loads((tmp_path / "mult_tune" / "stage1_manifest.json").read_text(encoding="utf-8"))
    stage1_jobs = manifest["stage1_jobs"]
    assert len(stage1_jobs) == 6
    assert {entry["expand_b_digits"] for entry in stage1_jobs} == {1}
    assert {entry["num_expand_rounds"] for entry in stage1_jobs} == {8}
    assert {entry["initial_max_b_digits"] for entry in stage1_jobs} == {8}
    assert {entry["frontier_row_profile"] for entry in stage1_jobs} == {"uniform", "hard_rows_v1"}
    assert {entry["learning_rate"] for entry in stage1_jobs} == {5e-5, 1e-4}
    assert {entry["max_steps"] for entry in stage1_jobs} == {3000, 6000}

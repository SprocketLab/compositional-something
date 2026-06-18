from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from self.experiments.figure2_condition_sweep import (
    build_addition_stage2_followups,
    build_stage1_manifest,
    finalize_selection_payload,
)


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "launchers" / "self" / "submit_figure2_condition_sweep_mig.sh"


def _write_results(path: Path, rows: list[dict]) -> str:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return str(path)


def test_figure2_condition_sweep_wrapper_has_valid_bash_syntax():
    assert WRAPPER.exists()
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)


def test_build_stage1_manifest_has_expected_job_counts(tmp_path: Path):
    manifest = build_stage1_manifest(
        out_root=tmp_path / "sweep",
        figure_dir=tmp_path / "figures",
        selection_json=tmp_path / "paper_schedule_selection.json",
        paper_schedule_env=tmp_path / "paper_schedule_selection.env",
        python_bin="python",
        run_length_seed_model=tmp_path / "run_length_seed",
        addition_seed_model=tmp_path / "addition_seed",
    )

    stage1_jobs = manifest["stage1_jobs"]
    assert len(stage1_jobs) == 13
    assert sum(1 for entry in stage1_jobs if entry["task"] == "run_length") == 4
    assert sum(1 for entry in stage1_jobs if entry["task"] == "addition") == 9


def test_build_addition_stage2_followups_only_adds_missing_baselines():
    followups = build_addition_stage2_followups(
        [
            {
                "schedule_label": "expand2_replay5000_train10000",
                "output_root": "/tmp/sched_a",
                "expand_num_digits": 2,
                "seed_replay_train_per_digit": 5000,
                "expand_train_per_digit": 10000,
            },
            {
                "schedule_label": "expand3_replay5000_train10000",
                "output_root": "/tmp/sched_b",
                "expand_num_digits": 3,
                "seed_replay_train_per_digit": 5000,
                "expand_train_per_digit": 10000,
            },
        ]
    )

    assert len(followups) == 4
    assert {entry["baseline"] for entry in followups} == {"short_only", "compose_corrupt"}
    assert {entry["schedule_label"] for entry in followups} == {
        "expand2_replay5000_train10000",
        "expand3_replay5000_train10000",
    }


def test_finalize_selection_payload_reuses_stage1_addition_results(tmp_path: Path):
    direct = _write_results(tmp_path / "direct.json", [{"round": 8, "max_digits": 31, "expanded_eval_accuracy": 0.15}])
    with_carry = _write_results(
        tmp_path / "with_carry.json",
        [{"round": 8, "max_digits": 31, "expanded_eval_accuracy": 0.33}],
    )
    filtered = _write_results(
        tmp_path / "filtered.json",
        [{"round": 8, "max_digits": 31, "expanded_eval_accuracy": 0.41, "frontier_train_accuracy": 0.29}],
    )
    short_only = _write_results(tmp_path / "short_only.json", [{"round": 8, "max_digits": 31}])
    corrupt = _write_results(tmp_path / "corrupt.json", [{"round": 8, "max_digits": 31}])
    run_length = _write_results(
        tmp_path / "run_length.json",
        [{"round": 8, "max_bits": 40, "per_bit_accuracy": {str(bits): 0.95 for bits in range(8, 41)}}],
    )

    stage2_selection = {
        "run_length_candidates": [],
        "addition_stage1_candidates": [],
        "selected_run_length": {
            "results_path": run_length,
            "expand_num_bits": 4,
            "expand_train_per_bit": 1200,
        },
        "selected_addition_topk": [
            {
                "schedule_label": "expand3_replay5000_train10000",
                "output_root": str(tmp_path / "addition_sched"),
                "expand_num_digits": 3,
                "seed_replay_train_per_digit": 5000,
                "expand_train_per_digit": 10000,
                "baseline_results": {
                    "direct": direct,
                    "with_carry": with_carry,
                    "with_carry_filtered": filtered,
                },
            }
        ],
        "addition_stage2_followups": [
            {
                "schedule_label": "expand3_replay5000_train10000",
                "baseline": "short_only",
                "results_path": short_only,
            },
            {
                "schedule_label": "expand3_replay5000_train10000",
                "baseline": "compose_corrupt",
                "results_path": corrupt,
            },
        ],
    }

    payload = finalize_selection_payload(
        stage2_selection=stage2_selection,
        run_length_seed_model="run_length_seed",
        addition_seed_model="addition_seed",
    )

    addition_candidates = payload["addition_stage2_fullpack_candidates"]
    assert len(addition_candidates) == 1
    assert addition_candidates[0]["baseline_results"]["direct"] == direct
    assert addition_candidates[0]["baseline_results"]["with_carry"] == with_carry
    assert addition_candidates[0]["baseline_results"]["with_carry_filtered"] == filtered
    assert addition_candidates[0]["baseline_results"]["short_only"] == short_only
    assert addition_candidates[0]["baseline_results"]["compose_corrupt"] == corrupt
    assert payload["selected_schedules"]["addition"]["results_path"] == filtered


def test_submit_figure2_condition_sweep_wrapper_dry_run_prints_expected_counts(tmp_path: Path):
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["OUT_ROOT"] = str(tmp_path / "figure2_condition_sweep")
    env["SELECTION_JSON"] = str(tmp_path / "paper_schedule_selection.json")
    env["PAPER_SCHEDULE_ENV"] = str(tmp_path / "paper_schedule_selection.env")
    env["FIGURE_DIR"] = str(tmp_path / "figures")
    env["LOG_DIR"] = str(tmp_path / "logs")

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout = result.stdout
    assert "self.experiments.figure2_condition_sweep submit" in stdout
    assert "Wrote stage-1 manifest" in stdout
    assert "Stage-1 counts: run_length=4 addition=9" in stdout
    assert "fig2-sweep-stage2" in stdout
    assert "DRY_RUN=1; stage-2 selector job not submitted." in stdout
